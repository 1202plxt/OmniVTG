import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import warnings
import cv2
import numpy as np
import argparse
import os
import gc
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")


# ========================================================================
# 🌟 核心创新：基于查询的时空注意力 (Query-Conditioned Spatio-Temporal Attention)
# ------------------------------------------------------------------------
# 工作流程：
#   input:  视觉编码器输出的扁平视频 token 序列 V (N_v, D_v)
#           与当前时刻对应的「自然语言查询」 token 序列 T (N_t, D_t)
#   step 1: V 通过一个可学习线性层，对齐到 LLM 文本隐藏空间 D_hid
#   step 2: 从 T 中池化得到 q_text ∈ R^(D_hid)  —— 「空间探照灯」
#   step 3: 对每帧做 Cross-Attention:   Attn(q_text, V_frame)
#           => 使与查询语义相关的视觉 patch 被放大，无关区域被压低
#   step 4: 跨帧做 Temporal Self-Attention，把帧间动作演化串起来
#   output: 重新加权后的视觉 token 序列，形状与输入一致，直接替换原 V
# ========================================================================
class QueryConditionedSTAttention(nn.Module):
    def __init__(
        self,
        vision_dim: int,        # 视觉编码器输出维度（例如 1536）
        text_dim: int,          # LLM 文本/词嵌入维度（例如 4096）
        hidden_dim: int = 1024, # 中间注意力维度
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.vision_proj = nn.Linear(vision_dim, hidden_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, vision_dim, bias=False)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.zeros(1))  # 残差门控，初始保持原行为

    def forward(
        self,
        visual_tokens: torch.Tensor,    # (total_vis_tokens, vision_dim)
        text_tokens: torch.Tensor,      # (1, text_len, text_dim)
        num_frames: int,                # 本段视频的帧数
    ) -> torch.Tensor:
        """
        visual_tokens 必须是扁平序列，按「帧1_tokens, 帧2_tokens, ...」顺序排列，
        且每帧 token 数相同；Qwen3-VL 的 video_grid_thw 正满足此约定。
        """
        total_tokens, v_dim = visual_tokens.shape
        tokens_per_frame = total_tokens // num_frames
        assert tokens_per_frame * num_frames == total_tokens, (
            f"视觉 token 总数 {total_tokens} 无法被帧数 {num_frames} 整除；"
            "请确认 Qwen3-VL visual 输出与 grid_thw 约定一致。"
        )

        # --- 1) 线性投影到共享注意力空间 ---
        v_hid = self.vision_proj(visual_tokens)          # (T, H)
        q_text_seq = self.text_proj(text_tokens)         # (1, L, H)
        # 文本隐层做 mean-pool，得到单向量「探照灯」
        q_text = q_text_seq.mean(dim=1, keepdim=True)    # (1, 1, H)

        # --- 2) 空间 Cross-Attention（每帧内部） ---
        v_frames = v_hid.view(num_frames, tokens_per_frame, -1)  # (F, P, H)
        q_spatial = q_text.expand(num_frames, 1, -1)             # (F, 1, H)

        # 对每个帧，把 q_text 作为 query，该帧所有 patch token 作为 key/value
        spatial_out, _ = self.spatial_attn(
            query=q_spatial, key=v_frames, value=v_frames
        )  # (F, 1, H) —— 每帧得到 1 个「文本对齐的帧特征」

        # 用该帧特征对原 patch 做标量重加权（sigmoid 门控）—— 不改变 token 数量
        frame_gate = torch.sigmoid((v_frames * spatial_out).sum(dim=-1, keepdim=True))
        v_spatial = v_frames * frame_gate                      # (F, P, H)
        v_spatial = self.norm1(v_spatial.reshape(total_tokens, -1))  # (T, H)

        # --- 3) 时间 Self-Attention（沿时间轴交换信息） ---
        # 取每帧的 [CLS] 风格代表（此处取均值）
        frame_reps = v_spatial.view(num_frames, tokens_per_frame, -1).mean(dim=1)  # (F, H)
        frame_reps = frame_reps.unsqueeze(0)                                       # (1, F, H)
        temp_out, _ = self.temporal_attn(
            query=frame_reps, key=frame_reps, value=frame_reps
        )  # (1, F, H)

        # 将时间聚合结果广播回每帧的每个 patch 做残差
        temp_broad = temp_out.squeeze(0).unsqueeze(1).expand(num_frames, tokens_per_frame, -1)
        temp_broad = temp_broad.reshape(total_tokens, -1)

        v_out = v_spatial + self.scale.tanh() * temp_broad
        v_out = self.norm2(v_out)

        # --- 4) 投影回 vision_dim，保证下游管线不变 ---
        refined = self.out_proj(v_out) + visual_tokens  # 残差保形
        return refined


# ========================================================================
# Dataset：1 秒滑窗 + 软标签 1-10（保持原逻辑，但 __getitem__ 改为 tensor 输出）
# ========================================================================
class CharadesOnlineTrainDataset(Dataset):
    def __init__(self, annotation_json, video_dir, num_frames=2, step_sec=2.0,
                 window_sec=2.0, binary_threshold=5):
        """
        num_frames   : 每个滑窗抽几帧（原 6 → 改 2，视觉 token 减少 3 倍）
        step_sec     : 滑窗步长秒（原 1.0 → 改 2.0，样本数减半）
        window_sec   : 单滑窗持续秒数（与 step_sec 保持一致即可）
        binary_threshold: overlap 映射到 1-10 后，>= 此值视为 positive
        """
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.step_sec = float(step_sec)
        self.window_sec = float(window_sec)
        self.binary_threshold = int(binary_threshold)

        with open(annotation_json, "r", encoding="utf-8") as f:
            if annotation_json.endswith(".jsonl"):
                self.samples = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                self.samples = json.load(f)

        self.flattened_windows = []
        self._prepare_online_stream_samples()

    def _prepare_online_stream_samples(self):
        print(f"⏳ 正在构建滑窗样本 (step={self.step_sec}s, window={self.window_sec}s, frames={self.num_frames})...")
        for sample in self.samples:
            video_file = self.video_dir / os.path.basename(sample["video_file"])
            if not video_file.exists():
                continue

            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            if duration < self.window_sec * 0.5:
                continue

            gt_windows = sample.get("relevant_windows", [])
            start_time = 0.0

            while start_time < duration:
                end_time = min(start_time + self.window_sec, duration)
                max_overlap = 0.0
                for gt_s, gt_e in gt_windows:
                    inter_s = max(start_time, gt_s)
                    inter_e = min(end_time, gt_e)
                    inter_len = max(0.0, inter_e - inter_s)
                    if inter_len > 0:
                        max_overlap = max(max_overlap, inter_len / self.window_sec)

                score_1_10 = int(np.round(max_overlap * 9 + 1))
                # 二值化：直接训练模型做「是否命中 relevant window」的判断
                label = "Yes" if score_1_10 >= self.binary_threshold else "No"

                self.flattened_windows.append({
                    "video_path": video_file,
                    "start": start_time,
                    "end":   end_time,
                    "query": sample["query"],
                    "label": label,
                    "score": score_1_10,
                })
                start_time += self.step_sec

        n_pos = sum(1 for w in self.flattened_windows if w["label"] == "Yes")
        print(f"✅ 样本构建完成：总计 {len(self.flattened_windows)} 个滑窗 "
              f"(positive {n_pos} / negative {len(self.flattened_windows) - n_pos})")

    def __len__(self):
        return len(self.flattened_windows)

    def __getitem__(self, idx):
        meta = self.flattened_windows[idx]
        cap = cv2.VideoCapture(str(meta["video_path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        time_points = np.linspace(meta["start"], meta["end"], self.num_frames)
        frames = []
        for t in time_points:
            frame_idx = min(int(t * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (336, 336))
            else:
                frame = np.zeros((336, 336, 3), dtype=np.uint8) if not frames else frames[-1]
            frames.append(frame)
        cap.release()

        frames_np = np.stack(frames, axis=0).astype(np.uint8)
        return frames_np, meta["query"], meta["label"]


# ========================================================================
# 训练主循环
# ========================================================================
SHORT_PROMPT = (
    'Does this video snippet match the query "{}"? '
    "Answer Yes or No only."
)


def build_prompt_and_messages(query_txt, label, num_frames):
    """
    极简版：instruction 只保留必要信息，assistant 只输出 "Yes" / "No" 一个单词。
    token 数量 ↓ → LLM 前向更快。
    """
    user_content = [{"type": "image"} for _ in range(num_frames)]
    user_content.append({"type": "text", "text": SHORT_PROMPT.format(query_txt)})

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": label},
    ]


def main():
    parser = argparse.ArgumentParser(description="极简版：基于查询的时空注意力 + 二元 Yes/No 判断微调")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=2,    help="每滑窗抽几帧（建议 2）")
    parser.add_argument("--step_sec",   type=float, default=2.0, help="滑窗步长秒（建议 2）")
    parser.add_argument("--window_sec", type=float, default=2.0, help="单滑窗持续秒数")
    parser.add_argument("--binary_threshold", type=int, default=5, help="1-10 score 超过则记 Yes")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--st_lr", type=float, default=1e-4, help="时空注意力模块的学习率（略大于基座 LoRA）")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/ST_Lora_V1")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) 加载基座
    # ------------------------------------------------------------
    print("📥 正在初始化大模型基座，自动分布至双卡显存池...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 视觉 / 文本 维度
    vision_dim = model.config.vision_config.hidden_size
    text_dim = model.config.hidden_size
    print(f"[dim] vision={vision_dim}, text={text_dim}")

    # ------------------------------------------------------------
    # 2) 刚性锁定基座参数
    # ------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------
    # 3) 部署 LoRA 旁路矩阵
    # ------------------------------------------------------------
    print("🔥 正在配置自注意力层 LoRA 适配器...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # ------------------------------------------------------------
    # 4) 部署「基于查询的时空注意力模块」
    # ------------------------------------------------------------
    st_module = QueryConditionedSTAttention(
        vision_dim=vision_dim,
        text_dim=text_dim,
        hidden_dim=1024,
        num_heads=8,
        dropout=0.1,
    ).to(model.device, dtype=torch.bfloat16)

    # 把它挂到 model 上，便于保存 / device_map 兼容
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.query_st_attention = st_module

    print("✅ 基于查询的时空注意力模块已挂载 (query_st_attention)")

    # ------------------------------------------------------------
    # 5) Monkey-patch：在 visual 编码器输出注入 q_text 驱动的重加权
    # ------------------------------------------------------------
    # 使用 Python 闭包存储当前 step 的「查询 tokens」和「帧数」。
    # 训练循环里每次 forward 前都需要设置它们。
    state = {
        "query_tokens": None,   # (1, L)
        "num_frames": args.num_frames,
    }

    original_visual_forward = model.visual.forward

    def patched_visual_forward(pixel_values, grid_thw, *args, **kwargs):
        # 先让原视觉编码器跑
        visual_out = original_visual_forward(pixel_values, grid_thw, *args, **kwargs)
        video_embeds, deepstack_video_embeds = visual_out  # (N, vision_dim), list[...]

        # 若当前 step 提供了基于查询的文本特征，则对视频 token 做「探照」
        if state["query_tokens"] is not None:
            with torch.set_grad_enabled(st_module.training and model.training):
                # 获取文本嵌入（使用 LLM 的词嵌入层，保证 q_text 与视觉对齐到同一语义空间）
                text_embeds = base_model.get_input_embeddings()(state["query_tokens"])  # (1, L, D_t)
                video_embeds_refined = st_module(
                    visual_tokens=video_embeds,          # (N, D_v)
                    text_tokens=text_embeds,             # (1, L, D_t)
                    num_frames=state["num_frames"],
                )
                return video_embeds_refined, deepstack_video_embeds

        return visual_out

    # 替换
    model.visual.forward = patched_visual_forward

    # ------------------------------------------------------------
    # 6) 优化器：LoRA 参数 + 时空注意力模块
    # ------------------------------------------------------------
    lora_params = [p for p in model.parameters() if p.requires_grad]
    st_params = list(st_module.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": st_params, "lr": args.st_lr, "weight_decay": 0.01},
        ]
    )

    # ------------------------------------------------------------
    # 7) 数据
    # ------------------------------------------------------------
    dataset = CharadesOnlineTrainDataset(
        args.annotation_json, args.video_dir,
        num_frames=args.num_frames,
        step_sec=args.step_sec, window_sec=args.window_sec,
        binary_threshold=args.binary_threshold,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # ------------------------------------------------------------
    # 8) 训练 loop —— 目标从 1-10 评分退化为 Yes/No 判断
    #    - 文本序列短 → 每个 step 的 LLM 前向显著变短
    # ------------------------------------------------------------
    print("\n==================== 🏁 启动有监督协同微调 ====================")
    for epoch in range(args.epochs):
        model.train()
        st_module.train()
        epoch_loss = 0.0
        n_steps = 0

        for step, (frames_batch, query_batch, label_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            # frames_batch: (1, T, H, W, C) -> (T, H, W, C)
            frames_np = frames_batch[0].numpy()
            query_txt = query_batch[0]
            label_txt = label_batch[0]  # "Yes" / "No"

            # 1) 构造对话格式 + 填图像
            messages = build_prompt_and_messages(query_txt, label_txt, args.num_frames)
            image_list = [frames_np[i] for i in range(args.num_frames)]

            for m_i, msg in enumerate(messages):
                if msg["role"] == "user":
                    new_content = []
                    img_idx = 0
                    for item in msg["content"]:
                        if item["type"] == "image":
                            new_content.append({"type": "image", "image": image_list[img_idx]})
                            img_idx += 1
                        else:
                            new_content.append(item)
                    messages[m_i]["content"] = new_content

            # 2) q_text：供 query-conditioned ST attention 使用的隐层特征
            q_ids = processor.tokenizer(
                query_txt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(model.device)
            state["query_tokens"] = q_ids
            state["num_frames"] = args.num_frames

            try:
                inputs = processor.apply_chat_template(
                    messages, tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    labels = inputs["input_ids"].clone()
                    # 极简 labels 只有 1–2 个 token（"Yes"/"No"）
                    # 稳妥起见保留最后 3 个位置不 mask（BOS/EOS 也可能多算但不影响梯度）
                    mask_len = max(1, labels.shape[1] - 3)
                    labels[:, :mask_len] = -100

                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

                if (step + 1) % 20 == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.epochs}] | Step [{step+1}/{len(dataloader)}] "
                        f"| gt={label_txt} | Loss={loss.item():.4f}", flush=True,
                    )
            except Exception as e:
                print(f"⚠️ 训练步长抖动保护: {str(e)}", flush=True)
                torch.cuda.empty_cache()
                continue
            finally:
                state["query_tokens"] = None
                if (step + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(1, n_steps)
        epoch_save_path = Path(args.save_dir) / f"epoch_{epoch+1}"
        model.save_pretrained(epoch_save_path)
        torch.save(st_module.state_dict(), epoch_save_path / "query_st_attention.pt")
        print(f"💾 Epoch {epoch+1} 完成 | 平均 Loss={avg_loss:.4f} | 保存至 {epoch_save_path}")


if __name__ == "__main__":
    main()
