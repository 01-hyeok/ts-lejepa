"""
run_classification.py
─────────────────────────────────────────────────────────────────────────────
LeJEPA time-series classification entry-point.

Mirrors run_linear_probing_lejepa.py in:
  - argument parsing style
  - checkpoint loading / saving logic
  - TensorBoard logging style (writer_train / writer_val / writer_test)
  - experiment directory organisation  (runs/<exp_id>/  checkpoints/classification/<exp_id>/)
  - encoder instantiation + freeze pattern
  - training / validation / test loop structure
  - seed setting style
  - device handling style

What is NEW compared to run_linear_probing_lejepa.py:
  - ClassificationHead  (temporal mean-pool → dropout → Linear → logits [B, num_class])
  - CrossEntropyLoss instead of MSELoss
  - Accuracy / F1 metrics printed at test time
  - Dataset sourced from lejepa.data_ts_classification
"""

import os
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, encoder, head, adapter, path):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, encoder, head, adapter, path)
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, encoder, head, adapter, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, encoder, head, adapter, path):
        if self.verbose:
            print(f'Validation Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best model...')
        torch.save({
            "head": head.state_dict(),
            "adapter": adapter.state_dict() if adapter is not None else None,
            "encoder": encoder.state_dict(),
        }, path)
        self.val_loss_min = val_loss


from lejepa.data_ts_classification import get_classification_loaders
from lejepa.arch_registry import (
    ADAPTER_FREE_ARCHS,
    SUPPORTED_ARCHS,
    build_encoder,
    get_embed_dim,
    infer_pretrain_in_vars,
    needs_channel_adapter,
    validate_arch,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LeJEPAClassificationHead(nn.Module):
    """
    Pure Linear classifier on top of the frozen LeJEPA encoder.

    Pipeline:
        1. Token mean pooling   : [B*C, N, d_model] -> [B*C, d_model]  (done in extract_emb)
        2. Channel mean pooling : [B, C, d_model]   -> [B, d_model]    (done in extract_emb)
        3. Linear classifier    : [B, d_model]      -> [B, num_class]  (this module)
    """

    def __init__(self, embed_dim: int, num_class: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(emb))


def extract_emb(encoder, x_in: torch.Tensor, arch_key: str) -> torch.Tensor:
    """
    x_in: [B, C, T]  (already padded/truncated to seq_len)
    returns: [B, embed_dim]
    """
    B, C, T = x_in.shape

    if arch_key == "timevlm":
        emb = encoder._process(x_in, target_res=224, training=False, is_downstream=True)

    elif arch_key == "timesnet":
        seq = encoder(x_in.transpose(1, 2))
        emb = seq.mean(dim=1)

    elif arch_key in ("patchtst", "utica"):
        actual_enc = encoder.encoder if arch_key == "utica" else encoder
        seq = actual_enc._process(x_in.unsqueeze(1), max_len=x_in.shape[-1], return_seq=True)
        emb = seq.mean(dim=1)
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    elif arch_key == "tiling_ci":
        emb = encoder._process(x_in.unsqueeze(1))
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    else:
        emb = encoder._process(
            x_in.unsqueeze(1),
            target_res=224,
            training=False,
            is_downstream=True,
        )
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    if arch_key == "tivit_indep":
        emb = emb.mean(1)

    return emb


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.arch = validate_arch(args.arch)

    train_loader, val_loader, test_loader, num_classes, in_vars, resolved_seq_len = get_classification_loaders(
        data_root=args.data_root,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )
    args.seq_len = resolved_seq_len

    arch_key = args.arch

    state_dict = None
    if os.path.exists(args.pretrain_path):
        ckpt = torch.load(args.pretrain_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

    is_ci = arch_key in ADAPTER_FREE_ARCHS
    if args.pretrain_in_vars > 0:
        pretrain_in_vars = args.pretrain_in_vars
    else:
        pretrain_in_vars = infer_pretrain_in_vars(
            arch_key,
            state_dict,
            in_vars,
            args.pretrain_dataset,
        )

    print("=" * 50)
    print(f"🚀 Architecture      : {args.arch.upper()}")
    print(f"📊 Dataset           : {args.data_root}")
    print(f"📊 Channels          : {in_vars}   (Pretrain: {pretrain_in_vars})  CI={is_ci}")
    print(f"📊 Classes           : {num_classes}")
    print(f"📊 Pretrain dataset  : {args.pretrain_dataset}")
    print("=" * 50)

    ckpt_type = (os.path.basename(args.pretrain_path).split("_")[2]
                 if "best" in args.pretrain_path else "total")
    exp_id = f"classification_{args.pretrain_dataset}_to_{args.dataset_name}_{arch_key}_{ckpt_type}"

    runs_root = os.path.join("runs", exp_id)
    ckpt_root = os.path.join("checkpoints", "classification", exp_id)

    runs_dir_train = os.path.join(runs_root, "train")
    runs_dir_val = os.path.join(runs_root, "val")
    os.makedirs(runs_dir_train, exist_ok=True)
    os.makedirs(runs_dir_val, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    writer_train = SummaryWriter(runs_dir_train)
    writer_val = SummaryWriter(runs_dir_val)

    encoder = build_encoder(
        arch_key,
        in_vars=pretrain_in_vars,
        vit_model=args.vit_model,
        proj_dim=args.proj_dim,
        use_revin=False,
    ).to(device)

    if state_dict is not None:
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        matched = len(state_dict) - len(missing)
        print(f"✅ Loaded pretrained encoder from {args.pretrain_path}")
        print(f"   - {matched}/{len(state_dict)} keys matched.")
        if missing:
            print(f"   ⚠️  Missing keys (sample): {missing[:3]} …")
    else:
        print(f"⚠️  Pretrain checkpoint not found: {args.pretrain_path} — training from scratch.")

    if not args.fine_tune:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        print("✅ Encoder is FROZEN (Linear Probing mode).")
    else:
        encoder.train()
        print("🔥 Encoder is UN-FROZEN (Full Fine-Tuning mode).")

    if needs_channel_adapter(arch_key, pretrain_in_vars, in_vars):
        channel_adapter = nn.Linear(in_vars, pretrain_in_vars).to(device)
        with torch.no_grad():
            channel_adapter.weight.zero_()
            for j in range(pretrain_in_vars):
                channel_adapter.weight[j, j % in_vars] = 1.0
        print(f"✅ Channel Adapter (CD): {in_vars} → {pretrain_in_vars}")
    else:
        channel_adapter = None
        if is_ci:
            print("ℹ️  Channel Adapter: skipped (CI architecture)")
        else:
            print("ℹ️  Channel Adapter: skipped (same channel count)")

    embed_dim = get_embed_dim(encoder, arch_key)

    head = LeJEPAClassificationHead(embed_dim, num_classes, dropout=args.dropout).to(device)
    print(f"✅ ClassificationHead: [B, {embed_dim}] → [B, {num_classes}]")

    trainable = list(head.parameters())
    if channel_adapter is not None:
        trainable += list(channel_adapter.parameters())
    if args.fine_tune:
        trainable += list(encoder.parameters())

    opt = torch.optim.RAdam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        head.train()
        if channel_adapter is not None:
            channel_adapter.train()
        if args.fine_tune:
            encoder.train()

        train_losses, train_preds, train_labels = [], [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)

            opt.zero_grad()

            if channel_adapter is not None:
                batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

            x_in = batch_x
            if args.fine_tune:
                emb = extract_emb(encoder, x_in, arch_key)
            else:
                with torch.no_grad():
                    emb = extract_emb(encoder, x_in, arch_key)

            logits = head(emb)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=4.0)
            opt.step()

            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        avg_train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        head.eval()
        if channel_adapter is not None:
            channel_adapter.eval()
        if args.fine_tune:
            encoder.eval()

        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device)

                if channel_adapter is not None:
                    batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

                emb = extract_emb(encoder, batch_x, arch_key)
                logits = head(emb)
                loss = criterion(logits, batch_y)

                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        writer_train.add_scalar("Loss", avg_train_loss, epoch)
        writer_train.add_scalar("Accuracy", train_acc, epoch)
        writer_train.add_scalar("F1_macro", train_f1, epoch)
        writer_val.add_scalar("Loss", avg_val_loss, epoch)
        writer_val.add_scalar("Accuracy", val_acc, epoch)
        writer_val.add_scalar("F1_macro", val_f1, epoch)

        early_stopping(avg_val_loss, encoder, head, channel_adapter, os.path.join(ckpt_root, "best_model.pt"))
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    best_ckpt = torch.load(os.path.join(ckpt_root, "best_model.pt"), map_location=device)
    head.load_state_dict(best_ckpt["head"])
    if channel_adapter is not None and best_ckpt.get("adapter") is not None:
        channel_adapter.load_state_dict(best_ckpt["adapter"])
    if "encoder" in best_ckpt:
        encoder.load_state_dict(best_ckpt["encoder"])

    head.eval()
    if channel_adapter is not None:
        channel_adapter.eval()
    encoder.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)

            if channel_adapter is not None:
                batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

            emb = extract_emb(encoder, batch_x, arch_key)
            logits = head(emb)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("=" * 50)
    print(f"Classification Test Result ({args.dataset_name})")
    print(f"Accuracy : {test_acc:.6f}")
    print(f"F1 (macro): {test_f1:.6f}")
    print("=" * 50)

    summary_path = os.path.join(ckpt_root, "results_summary.txt")
    with open(summary_path, "a") as f:
        f.write(
            f"Dataset: {args.dataset_name} | Pretrain: {args.pretrain_dataset} | "
            f"Arch: {arch_key} | Acc: {test_acc:.6f} | F1: {test_f1:.6f}\n"
        )

    writer_train.close()
    writer_val.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LeJEPA Time-Series Classification")

    p.add_argument("--arch", default="basic", help="Supported: " + ", ".join(SUPPORTED_ARCHS))
    p.add_argument("--vit_model", default="vit_small_patch14_dinov2")
    p.add_argument("--proj_dim", type=int, default=128)

    p.add_argument("--pretrain_dataset", default="tsld")
    p.add_argument("--pretrain_path", required=True)

    p.add_argument("--data_root", required=True, help="Path to dataset directory (.ts) or CSV file")
    p.add_argument("--dataset_name", default="UCR_dataset", help="Human-readable name used in logs and checkpoint paths")

    p.add_argument("--seq_len", type=int, default=0, help="Sequence length (0 for auto-detect from dataset)")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fine_tune", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, help="Unfreeze and fine-tune the encoder")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain_in_vars", type=int, default=0, help="Override pretrain channel count (use only if auto-detect fails)")

    args = p.parse_args()
    train(args)
