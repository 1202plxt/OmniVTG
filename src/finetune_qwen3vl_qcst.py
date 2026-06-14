import torch
import torch.nn as nn
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
from PIL import Image  # 新增
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")


# ========================================================================
# 🌟 核心创新模块：基于查询的时空注意力 (Query-Conditioned ST-Attention)
# ========================================================================
class QueryConditionedSTAttention(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.vision_proj = nn.Linear(vision_dim, hidden_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, vision_dim, bias=False)

        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.zeros(1))  # 残差门控，保证初始稳定性

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        text_tokens: 来自 LLM embedding 层的输出，形状 (1, L, text_dim)
        """
        total_tokens, v_dim = visual_tokens.shape
        tokens_per_frame = total_tokens // num_frames

        # 1) 跨模态投影
        v_hid = self.vision_proj(visual_tokens)          # (N, H)
        q_text = self.text_proj(text_tokens.mean(dim=1, keepdim=True))  # (1, 1, text_dim) -> proj -> (1, 1, H)

        # 2) 空间 Cross-Attention (基于文本筛选视觉)
        v_frames = v_hid.view(num_frames, tokens_per_frame, -1)
        q_spatial = q_text.expand(num_frames, 1, -1)
        spatial_out, _ = self.spatial_attn(query=q_spatial, key=v_frames, value=v_frames)

        frame_gate = torch.sigmoid((v_frames * spatial_out).sum(dim=-1, keepdim=True))
        v_spatial = self.norm1((v_frames * frame_gate).reshape(total_tokens, -1))

        # 3) 时间 Self-Attention (建立帧间因果)
        frame_reps = v_spatial.view(num_frames, tokens_per_frame, -1).mean(dim=1).unsqueeze(0)
        temp_out, _ = self.temporal_attn(query=frame_reps, key=frame_reps, value=frame_reps)
        temp_broad = temp_out.squeeze(0).unsqueeze(1).expand(num_frames, tokens_per_frame, -1).reshape(total_tokens, -1)

        v_out = self.norm2(v_spatial + self.scale.tanh() * temp_broad)

        # 4) 投影回原视觉空间并添加残差
        return self.out_proj(v_out) + visual_tokens


# ========================================================================
# 📊 数据集构建：2秒滑窗 + IoU物理量化软标签
# ========================================================================
class CharadesOnlineTrainDataset(Dataset):
    def __init__(self, annotation_json, video_dir, num_frames=8, step_sec=2.0, window_sec=2.0):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.step_sec = float(step_sec)
        self.window_sec = float(window_sec)

        with open(annotation_json, "r", encoding="utf-8") as f:
            if annotation_json.endswith(".jsonl"):
                self.samples = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                self.samples = json.load(f)

        self.flattened_windows = []
        self._prepare_online_stream_samples()

    def _prepare_online_stream_samples(self):
        print(f"⏳ 正在构建滑窗样本 (window={self.window_sec}s, frames={self.num_frames})")

        n_videos_ok = 0
        n_videos_skip = 0
        total_wins = 0

        # 字段名兼容：支持多套注释格式
        VIDEO_KEYS = ["video_file", "vid", "video", "video_id"]
        QUERY_KEYS = ["query", "sentence", "description", "text"]
        WINDOW_KEYS = ["relevant_windows", "timestamps", "gt_windows", "window"]
        DURATION_KEYS = ["duration", "length", "len"]

        for sample in self.samples:
            video_fn = None
            for k in VIDEO_KEYS:
                if k in sample and sample[k]:
                    video_fn = sample[k]
                    break
            if video_fn is None:
                continue

            video_file = self.video_dir / os.path.basename(str(video_fn))
            if not video_file.exists():
                n_videos_skip += 1
                continue

            # 查询文本
            query_txt = None
            for k in QUERY_KEYS:
                if k in sample and sample[k]:
                    query_txt = str(sample[k])
                    break
            if query_txt is None:
                continue

            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps and fps > 0 else 0.0
            cap.release()

            # 兼容多种字段写法: relevant_windows / timestamps / [s, e]
            gt_windows = []
            for k in WINDOW_KEYS:
                if k not in sample or sample[k] is None:
                    continue
                val = sample[k]
                if isinstance(val, (list, tuple)) and len(val):
                    # 形如 [[s1,e1], [s2,e2]]
                    if isinstance(val[0], (list, tuple)):
                        gt_windows = [list(v) for v in val]
                        break
                    # 形如 [s, e] —— 单个窗口
                    if len(val) == 2 and isinstance(val[0], (int, float)):
                        gt_windows = [list(val)]
                        break

            if len(gt_windows) == 0 or duration <= 0.5:
                n_videos_skip += 1
                continue

            n_videos_ok += 1
            start_time = 0.0
            while start_time < duration:
                end_time = min(start_time + self.window_sec, duration)
                max_overlap = 0.0
                for gt_s, gt_e in gt_windows:
                    inter_s, inter_e = max(start_time, float(gt_s)), min(end_time, float(gt_e))
                    inter_len = max(0.0, inter_e - inter_s)
                    if inter_len > 0:
                        max_overlap = max(max_overlap, inter_len / self.window_sec)

                score_1_10 = int(np.round(max_overlap * 9 + 1))

                self.flattened_windows.append({
                    "video_path": video_file, "start": start_time, "end": end_time,
                    "query": query_txt, "score": score_1_10
                })
                total_wins += 1
                start_time += self.step_sec

        print(f"✅ 样本构建完成: 视频可用={n_videos_ok}, 视频跳过={n_videos_skip}, 滑窗样本={total_wins}")

    def __len__(self):
        return len(self.flattened_windows)

    def __getitem__(self, idx):
        meta = self.flattened_windows[idx]
        cap = cv2.VideoCapture(str(meta["video_path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        # 关键优化: 一次 seek 到窗口起点，顺序读取 num_frames * stride 帧，而不是每帧 seek
        stride = max(1, int(round(fps * (self.window_sec / max(1, self.num_frames)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(meta["start"] * fps)))

        frames = []
        target = set(range(0, self.num_frames * stride, stride))
        i = 0
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i in target:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (336, 336))
                frames.append(frame)
            i += 1
        cap.release()

        # 兜底：如果视频比预期短，用最后一帧/零帧补齐
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((336, 336, 3), dtype=np.uint8))
        return np.stack(frames, axis=0).astype(np.uint8), str(meta["query"]), str(meta["score"])


# ========================================================================
# 🚀 训练主控制器与 Prompt 注入
# ========================================================================
SCORE_PROMPT = (
    'Task: Evaluate the temporal overlap (IoU) between the action query "{}" and the current video window.\n'
    'Rule: Rate the overlap strictly on a scale from 1 to 10 based on the proportion of the action present:\n'
    '- 1: 0% overlap (Pure background)\n'
    '- 3: ~20% overlap (Action just starting or ending)\n'
    '- 6: ~50% overlap (Half action, half background)\n'
    '- 10: 100% overlap (The window is entirely filled with the action)\n'
    'Output ONLY the single integer score.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--st_lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/ST_Lora_V1")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("📥 正在初始化模型与双卡显存池...")
    # 修 1: torch_dtype -> dtype（同时保留 torch_dtype，避免 warning 报错
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 冻结基座
    for param in model.parameters():
        param.requires_grad = False

    print("🔥 配置 LoRA 与 QueryConditionedSTAttention...")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)

    # 修 2: 视觉编码器输出经过 merger 后维度等于 text_config.hidden_size（4096）
    # 所以 vision_proj 和 text_proj 的输入维度都是 4096
    st_module = QueryConditionedSTAttention(
        vision_dim=model.config.text_config.hidden_size,  # ← 改成 text_config.hidden_size
        text_dim=model.config.text_config.hidden_size,
        hidden_dim=1024, num_heads=8,
    ).to(model.device, dtype=torch.bfloat16)

    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.query_st_attention = st_module

    # Monkey-patch 动态注入时空注意力
    state = {"query_tokens": None, "num_frames": args.num_frames}
    original_visual_forward = model.visual.forward

    def patched_visual_forward(pixel_values, grid_thw, *v_args, **kwargs):
        visual_out = original_visual_forward(pixel_values, grid_thw, *v_args, **kwargs)
        video_embeds, deepstack_video_embeds = visual_out
        if state["query_tokens"] is not None:
            with torch.set_grad_enabled(st_module.training and model.training):
                # state["query_tokens"] 已经是 embedding 输出 (1, L, text_dim)，直接用，不再过 embedding 层
                video_embeds_refined = st_module(
                    visual_tokens=video_embeds,
                    text_tokens=state["query_tokens"],
                    num_frames=state["num_frames"],
                )
                return video_embeds_refined, deepstack_video_embeds
        return visual_out

    model.visual.forward = patched_visual_forward

    # 修 3: 联合优化器 —— 从 model.parameters() 里排除 st_module 的参数，避免重复
    st_param_ids = set(id(p) for p in st_module.parameters())
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in st_param_ids]
    st_params = list(st_module.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": st_params, "lr": args.st_lr, "weight_decay": 0.01},
    ])

    print(f"[debug] LoRA 参数数量 = {len(lora_params)}, ST-Attention 参数数量 = {len(st_params)}")
    print(f"[debug] LoRA 参数量 = {sum(p.numel() for p in lora_params) / 1e6:.2f}M, "
          f"ST-Attention 参数量 = {sum(p.numel() for p in st_params) / 1e6:.2f}M")

    dataset = CharadesOnlineTrainDataset(
        args.annotation_json, args.video_dir, num_frames=args.num_frames, step_sec=2.0, window_sec=2.0,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    print("\n==================== 🏁 启动有监督复合微调 ====================")
    for epoch in range(args.epochs):
        model.train()
        st_module.train()
        epoch_loss = 0.0

        for step, (frames_batch, query_batch, score_str_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            frames_np = frames_batch[0].numpy()
            query_txt = query_batch[0]
            score_str = score_str_batch[0]

            # 构造对话格式：多图像 + 文本
            # Qwen3-VL 的 apply_chat_template 会自动处理图像占位符
            eval_prompt = SCORE_PROMPT.format(query_txt)
            
            # 构造 content：先放所有图像，再放文本
            content = []
            for i in range(args.num_frames):
                content.append({
                    "type": "image",
                    "image": Image.fromarray(frames_np[i])
                })
            content.append({
                "type": "text",
                "text": eval_prompt
            })
            
            messages = [{
                "role": "user",
                "content": content
            }, {
                "role": "assistant",
                "content": score_str
            }]

            # 使用 apply_chat_template 自动处理图像占位符
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,  # ← 必须加这个参数，否则返回字符串
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            # 1) tokenizer 输出 token ids，2) 过 embedding 层得到隐层，3) 传给 ST-Attention
            q_ids = processor.tokenizer(
                query_txt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            text_embeds = base_model.get_input_embeddings()(q_ids)  # (1, L, text_dim)
            state["query_tokens"] = text_embeds  # ← 存的是 embedding，不是 token ids
            state["num_frames"] = args.num_frames

            try:
                # 对齐 labels：prompt 部分的 token 不参与 loss 计算
                # Qwen3VL processor 返回的 input_ids 顺序为：文本 tokens + 视觉 patch tokens
                # 视觉 token 在文本 tokens 之后，我们 mask 掉前面的文本部分
                labels = inputs["input_ids"].clone()
                text_len = q_ids.shape[1]  # query 长度
                # mask 掉前面的文本 token（prompt + query），只监督最后几个数字 token
                mask_len = max(2, labels.shape[1] - 3)
                labels[:, :mask_len] = -100

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (step + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.epochs}] | Step [{step+1}/{len(dataloader)}] | "
                        f"gt={score_str} | Loss={loss.item():.4f}", flush=True,
                    )

            except Exception as e:
                print(f"⚠️ 步长保护: {str(e)}", flush=True)
                import traceback; traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            finally:
                state["query_tokens"] = None
                if (step + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_save_path = Path(args.save_dir) / f"epoch_{epoch+1}"
        model.save_pretrained(epoch_save_path)
        torch.save(st_module.state_dict(), epoch_save_path / "query_st_attention.pt")
        print(f"💾 Epoch {epoch+1} 完毕 | 平均 Loss = {avg_loss:.4f} | 保存路径: {epoch_save_path}")


if __name__ == "__main__":
    main()
