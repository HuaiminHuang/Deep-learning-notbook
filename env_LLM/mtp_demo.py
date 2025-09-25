# requirements:
# pip install torch transformers accelerate einops

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# --------- 超参 / 配置（可调） ----------
base_model_name = "qwen-base"   # <- 替换为实际模型名
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2                  # 8GB GPU 下常用小 batch
accumulation_steps = 8          # 梯度累积，用来虚拟增大 batch
lr = 2e-5
weight_decay = 0.01
max_seq_len = 512
mtp_depths = [2, 3]             # 我们构造两个 MTP module：预测 t+2 和 t+3；可扩展
loss_weights = {"main": 1.0, "mtp": 0.5}  # mtp 的权重可调
# --------------------------------------

# ---------- 辅助：简单实现 RMSNorm ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, D)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

# --------- 加载 base model 和 tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,   # 加载为 fp16（节省显存） —— 但要确保显卡和 transformers 版本兼容
    low_cpu_mem_usage=True       # 尝试节省 CPU 内存加载（HF 支持）
)
base_model.eval()  # 可根据是否要微调调整
base_model.to(device)

# 找到 base 模型里的 transformer block（不同模型命名不同）
def extract_transformer_blocks(model):
    """
    尝试从常见结构中提取 transformer block 列表：
      - model.model.decoder.layers / model.model.layers / model.transformer.h / model.base_model.model.decoder.layers 等
    返回 list of nn.Module
    """
    # 常见名字尝试（按顺序尝试）
    possible_paths = [
        ["model", "decoder", "layers"],
        ["model", "layers"],
        ["transformer", "h"],
        ["base_model", "model", "decoder", "layers"],
        ["transformer", "layers"],
    ]
    for path in possible_paths:
        cur = model
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, tuple)) or hasattr(cur, "__len__"):
            try:
                # try to index the sequence-like module container
                return list(cur)
            except Exception:
                pass
    # fallback: search for modules that 'look like' a block by name
    blocks = []
    for name, mod in model.named_modules():
        # very heuristic: name contains 'block' or 'layer' and module has self-attention attributes
        if ("block" in name or "layer" in name or "h." in name) and hasattr(mod, "attention") or hasattr(mod, "self_attn") or "attn" in name:
            blocks.append(mod)
    return blocks

transformer_blocks = extract_transformer_blocks(base_model)
print(f"找到 {len(transformer_blocks)} transformer blocks（示例索引 0）")
# 为安全起见，如果没有找到，直接抛错提示用户检查模型结构
if len(transformer_blocks) == 0:
    raise RuntimeError("无法自动找到 base model 的 transformer blocks，检查模型结构并修改 extract_transformer_blocks。")

# --------- 定义 MTPModule ----------
class MTPModule(nn.Module):
    def __init__(self, shared_embedding: nn.Embedding, shared_lm_head: nn.Linear,
                 base_block: nn.Module, hidden_size: int, seq_len: int = 64):
        """
        shared_embedding: 共享 embedding 层（来自 base_model）
        shared_lm_head: 共享输出 head（来自 base_model.lm_head 或 tie_weights）
        base_block: 一个 transformer block（会复制，避免直接修改 base）
        hidden_size: 模型隐藏维度
        seq_len: MTP 模块处理的序列长度（窗口大小）
        """
        super().__init__()
        self.shared_embedding = shared_embedding
        self.rms1 = RMSNorm(hidden_size)
        self.rms2 = RMSNorm(hidden_size)
        # 用 linear projection 来做 "concatenation + 投影"（图中的 linear projection）
        # 如果使用拼接两份 hidden，输入维度 2*hidden_size，否则用 hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)  # 简化版：1->1 投影
        # 使用 base_block 的深拷贝作为我们的 transformer block（保持架构与参数）
        self.transformer_block = copy.deepcopy(base_block)
        # 输出头直接复用共享 lm head，不在此处创建新的参数
        self.shared_lm_head = shared_lm_head
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T_mtp)  MTP 模块的输入 token ids（例如：t2..t5）
        返回：logits (B, T_mtp, vocab_size)
        """
        # embedding -> rmsnorm -> proj -> transformer_block -> lm_head
        emb = self.shared_embedding(input_ids)            # (B, T, D)
        x = self.rms1(emb)                                # RMSNorm 处理
        x = self.proj(x)                                  # 保证维度一致
        # 运行 transformer block：不同模型的 block forward 签名不同。这里做通用尝试：
        try:
            # 一般 block 接受 hidden_states, attention_mask...
            x = self.transformer_block(x, attention_mask=attention_mask)[0]
        except Exception:
            # fallback: 有些 block 以 (hidden_states, ) 返回
            x = self.transformer_block(x)
        # logits via shared head: 假设 shared_lm_head 是线性层 (D -> vocab)
        logits = self.shared_lm_head(x)
        return logits

# --------- 构造 MTP 模块集合（用第 k 层初始化） ----------
hidden_size = base_model.config.hidden_size
vocab_size = base_model.config.vocab_size

# get shared embedding & lm_head (depends on model naming)
# common patterns: base_model.get_input_embeddings(), base_model.lm_head / base_model.get_output_embeddings()
shared_embedding = base_model.get_input_embeddings()
try:
    shared_lm_head = base_model.get_output_embeddings()
    if shared_lm_head is None:
        # some models tie weights: build linear from embedding weight
        shared_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        shared_lm_head.weight = shared_embedding.weight
except Exception:
    shared_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    shared_lm_head.weight = shared_embedding.weight

mtp_modules = nn.ModuleDict()
# 选择用 base model 的第 N 层（例如第 middle 层）作为初始 transformer block
init_layer_idx = min(4, len(transformer_blocks)-1)  # 示例：用第 4 层初始化
for d in mtp_depths:
    block_copy = transformer_blocks[init_layer_idx]
    mtp_modules[f"mtp_{d}"] = MTPModule(
        shared_embedding=shared_embedding,
        shared_lm_head=shared_lm_head,
        base_block=block_copy,
        hidden_size=hidden_size,
        seq_len=64
    )
mtp_modules.to(device)

# --------- 构造训练用的 wrapper（计算 main loss + mtp losses） ----------
class MTPTrainer(nn.Module):
    def __init__(self, base_model, mtp_modules: nn.ModuleDict, tokenizer):
        super().__init__()
        self.base = base_model
        self.mtp_modules = mtp_modules
        self.tokenizer = tokenizer

    def forward(self, batch):
        """
        batch: dict with 'input_ids' and 'attention_mask' (B, L)
        我们需要构造：
          - main logits: base_model(input_ids) -> logits
          - 对于每个 mtp depth d: 构造 shifted inputs and targets (窗口) -> mtp_logits -> loss
        返回 汇总损失和单项 loss dict
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # --- main forward: predict next token for each position (causal LM) ---
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits_main = outputs.logits  # (B, L, V)
        # construct main targets: standard causal next-token target (shift left)
        labels_main = batch.get("labels", None)
        if labels_main is None:
            labels_main = input_ids[..., 1:].clone()
            # pad final token with -100 to ignore
            labels_main = torch.cat([labels_main, torch.full((input_ids.size(0),1), -100, dtype=torch.long, device=device)], dim=1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_main = loss_fct(logits_main.view(-1, logits_main.size(-1)), labels_main.view(-1))

        # --- MTP modules loss ---
        mtp_losses = {}
        for key, mtp in self.mtp_modules.items():
            # key format "mtp_d"
            d = int(key.split("_")[1])
            # build MTP window: for each position i, we want the module to predict token at i+d
            # simplified batching: we create sub-sequences shifted by d and use them as inputs/targets
            # here choose windows of length W from anywhere—simple approach: feed suffix/prefix windows
            W = mtp.seq_len
            B, L = input_ids.size()
            # we will use last W tokens before the position where we predict future token
            # For demo: take windows [0:W] predict tokens at index W-1 + d (ensure within L)
            # In practice you would enumerate many windows across sequence.
            if L < W + d:
                # skip if not enough tokens
                mtp_losses[key] = torch.tensor(0.0, device=device)
                continue

            # take inputs as tokens from positions [0:W]
            mtp_input_ids = input_ids[:, :W].contiguous()
            # targets: token at position W-1 + d  (one token per sequence) -> we will expand to a sequence target by shifting
            mtp_target_pos = W - 1 + d
            mtp_targets = input_ids[:, mtp_target_pos].contiguous()    # (B,)
            # run mtp forward -> logits (B, T_mtp, V)
            mtp_logits = mtp(mtp_input_ids)  # (B, W, V)
            # simplify: take last position's logits as prediction for that future token
            logits_pred = mtp_logits[:, -1, :]  # (B, V)
            loss_m = F.cross_entropy(logits_pred, mtp_targets)
            mtp_losses[key] = loss_m

        total_mtp_loss = sum(mtp_losses.values())
        total_loss = loss_main * loss_weights["main"] + total_mtp_loss * loss_weights["mtp"]
        # return dict
        out = {"loss": total_loss, "loss_main": loss_main, "mtp_losses": mtp_losses}
        return out

trainer_model = MTPTrainer(base_model, mtp_modules, tokenizer).to(device)

# ---------- 简单 Dataset（示例） ----------
class MyTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len=512):
        self.examples = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt")
            self.examples.append(enc)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

# 示例数据
texts = [
    "今天天气很好，我们去公园散步。",
    "机器学习中的优化算法包括 SGD, Adam 等。",
    # 多放几条你的训练文本
]
dataset = MyTextDataset(texts * 200, tokenizer, max_len=max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Optimizer & AMP 设置 ----------
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trainer_model.parameters()), lr=lr, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()

# ---------- 训练循环（带梯度累积 & AMP） ----------
trainer_model.train()
num_epochs = 3
global_step = 0
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            out = trainer_model(batch)
            loss = out["loss"] / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        if global_step % 10 == 0:
            print(f"epoch {epoch} step {global_step} loss_main={out['loss_main'].item():.4f} mtp={sum(v.item() for v in out['mtp_losses'].values()):.4f}")

# 保存模型（只保存微调的参数）
torch.save({
    "mtp_state_dict": mtp_modules.state_dict(),
    "optimizer": optimizer.state_dict(),
}, "mtp_finetuned.pt")

