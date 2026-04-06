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

# ── local imports (same pattern as run_linear_probing_lejepa.py) ──────────────
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, encoder, head, adapter, path):
        # Loss 기준: 낮을수록 좋음
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
            "head":    head.state_dict(),
            "adapter": adapter.state_dict() if adapter is not None else None,
            "encoder": encoder.state_dict()
        }, path)
        self.val_loss_min = val_loss

from lejepa.data_ts_classification import get_classification_loaders

from lejepa.model_ts_lejepa_basic   import MultiResViTEncoder
from lejepa.model_ts_lejepa_tiling  import MultiResViTTilingEncoder
from lejepa.model_ts_tivit          import TiViTIndependentEncoder, TiViTDependentEncoder
from lejepa.model_ts_timevlm        import TimeVLMEncoder
from lejepa.model_ts_conv2d         import Conv2DLearnableEncoder
from lejepa.model_ts_lejepa_ci      import MultiResViTCIEncoder
from lejepa.model_ts_lejepa_1d      import PatchTS1DEncoder
from lejepa.model_ts_utica          import UTICAEncoder
from lejepa.model_ts_conv           import MultiResViTConvEncoder
from lejepa.model_ts_timesblock     import LeJEPATimesModel

# Architectures that are Channel-Independent (CI):
# They process each channel separately, so no channel adapter is needed.
CI_ARCHS = {
    "timevlm", "tiling_ci", "patchtst", "utica",
    "timesnet", "timesnet_update",
}

# CD (Channel-Dependent) architectures need a channel adapter when the
# downstream dataset has a different number of channels than pretraining.
CD_ARCHS = {
    "basic", "tiling", "tiling_repeat", "tivit_indep", "tivit_dep",
    "conv2d", "conv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (mirrors run_linear_probing_lejepa.py)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Classification Head
# ─────────────────────────────────────────────────────────────────────────────

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
        # emb: [B, embed_dim]
        return self.fc(self.dropout(emb))   # [B, num_class]


# ─────────────────────────────────────────────────────────────────────────────
# Encoder extraction helper  (mirrors emb extraction in run_linear_probing)
# ─────────────────────────────────────────────────────────────────────────────

def extract_emb(encoder, x_in: torch.Tensor, arch_key: str) -> torch.Tensor:
    """
    x_in: [B, C, T]  (already padded/truncated to seq_len)
    returns: [B, embed_dim]
    """
    B, C, T = x_in.shape

    if arch_key == "timevlm":
        # TimeVLM internally maps [B, C, T] -> [B, 3, 224, 224] image (CD-like preserved B)
        emb = encoder._process(x_in, target_res=224, training=False, is_downstream=True)
        # Returns [B, D]

    elif arch_key in ("patchtst", "utica"):
        actual_enc = encoder.encoder if arch_key == "utica" else encoder

        # Step 1. Token sequence: [B*C, N, d_model]
        seq = actual_enc._process(x_in.unsqueeze(1), max_len=x_in.shape[-1], return_seq=True)

        # Step 2. Token mean pooling: [B*C, N, d_model] -> [B*C, d_model]
        emb = seq.mean(dim=1)

        # Step 3. Channel mean pooling: [B*C, d_model] -> [B, C, d_model] -> [B, d_model]
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    elif arch_key == "tiling_ci":
        # MultiResViTCIEncoder._process only takes (x)
        emb = encoder._process(x_in.unsqueeze(1))
        # Aggregate CI channels: [B*C, D] -> [B, C, D] -> [B, D]
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    elif arch_key in ("timesnet", "timesnet_update"):
        # Explicit Channel-Independent (CI) mode for TimesNet
        # [B, C, T] -> [B*C, T, 1]
        x_ci = x_in.reshape(B * C, 1, T).transpose(1, 2) # [B*C, T, 1]
        
        # Forward encoder (expects C=1 if pretrained as CI)
        seq_repr = encoder(x_ci)   # [B*C, T, D]
        emb_p = seq_repr.mean(dim=1) # [B*C, D]
        
        # Aggregate CI channels: [B*C, D] -> [B, C, D] -> [B, D]
        if C > 1:
            emb = emb_p.reshape(B, C, -1).mean(dim=1)
        else:
            emb = emb_p

    else:
        # Fallback for CD/Standard models (basic, tiling, conv, timevlm)
        # These typically take (x, target_res, training, is_downstream)
        emb = encoder._process(x_in.unsqueeze(1), target_res=224, training=False,
                                is_downstream=True)
        # If result is flattened [B*C, D], aggregate
        if emb.shape[0] == B * C and C > 1:
            emb = emb.reshape(B, C, -1).mean(dim=1)

    # TiViT Independent returns [B, C, Embed]
    if arch_key == "tivit_indep":
        emb = emb.mean(1)   # [B, Embed]

    return emb   # [B, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, num_classes, in_vars, resolved_seq_len = get_classification_loaders(
        data_root=args.data_root,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )
    
    # Overwrite args.seq_len with the dynamically discovered length so the rest of the script syncs to it
    args.seq_len = resolved_seq_len

    arch_key = args.arch.lower().replace("-", "_")

    # ── 2. Pretrained encoder — channel inference + load ────────────────────
    pretrain_in_vars = in_vars
    state_dict = None
    if os.path.exists(args.pretrain_path):
        ckpt = torch.load(args.pretrain_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

    # channel-independent: pretrain_in_vars is always in_vars (no adapter needed)
    # channel-dependent   : infer actual pretrain channels from checkpoint keys
    is_ci = arch_key in CI_ARCHS
    if is_ci:
        # TimeVLM is CI-arch but handles multiple channels internally in its image mapping.
        # Others (PatchTST, TimesNet CI) expect 1-channel input per sample.
        pretrain_in_vars = in_vars if arch_key == "timevlm" else 1
    elif state_dict is not None:
        if arch_key == "basic" and "proj.weight" in state_dict:
            pretrain_in_vars = state_dict["proj.weight"].shape[1]
        elif arch_key in ("tiling", "tiling_repeat", "tivit_dep") \
                and "backbone.patch_embed.proj.weight" in state_dict:
            pretrain_in_vars = state_dict["backbone.patch_embed.proj.weight"].shape[1]
        elif arch_key == "conv2d" and "canvas_encoder.proj.weight" in state_dict:
            pretrain_in_vars = state_dict["canvas_encoder.proj.weight"].shape[1]
        else:
            pretrain_in_vars = 321 if args.pretrain_dataset == "electricity" else in_vars
    else:
        pretrain_in_vars = 321 if args.pretrain_dataset == "electricity" else in_vars

    print("=" * 50)
    print(f"🚀 Architecture      : {args.arch.upper()}")
    print(f"📊 Dataset           : {args.data_root}")
    print(f"📊 Channels          : {in_vars}   (Pretrain: {pretrain_in_vars})  CI={is_ci}")
    print(f"📊 Classes           : {num_classes}")
    print(f"📊 Pretrain dataset  : {args.pretrain_dataset}")
    print("=" * 50)

    # ── 2.5 Experiment directories (mirrors run_linear_probing_lejepa.py) ───
    ckpt_type = (os.path.basename(args.pretrain_path).split("_")[2]
                 if "best" in args.pretrain_path else "total")
    exp_id    = f"classification_{args.pretrain_dataset}_to_{args.dataset_name}_{arch_key}_{ckpt_type}"

    runs_root = os.path.join("runs", exp_id)
    ckpt_root = os.path.join("checkpoints", "classification", exp_id)

    runs_dir_train = os.path.join(runs_root, "train")
    runs_dir_val   = os.path.join(runs_root, "val")
    os.makedirs(runs_dir_train, exist_ok=True)
    os.makedirs(runs_dir_val,   exist_ok=True)
    os.makedirs(ckpt_root,      exist_ok=True)

    writer_train = SummaryWriter(runs_dir_train)
    writer_val   = SummaryWriter(runs_dir_val)

    # ── 3. Encoder instantiation + freeze ───────────────────────────────────
    if arch_key == "basic":
        encoder = MultiResViTEncoder(in_vars=pretrain_in_vars,
                                     model_name=args.vit_model,
                                     proj_dim=args.proj_dim).to(device)
    elif arch_key in ("tiling", "tiling_repeat"):
        encoder = MultiResViTTilingEncoder(in_vars=pretrain_in_vars,
                                           model_name=args.vit_model,
                                           proj_dim=args.proj_dim).to(device)
    elif arch_key == "tivit_indep":
        encoder = TiViTIndependentEncoder(in_vars=pretrain_in_vars,
                                          model_name=args.vit_model,
                                          proj_dim=args.proj_dim).to(device)
    elif arch_key == "tivit_dep":
        encoder = TiViTDependentEncoder(in_vars=pretrain_in_vars,
                                        model_name=args.vit_model,
                                        proj_dim=args.proj_dim).to(device)
    elif arch_key == "timevlm":
        encoder = TimeVLMEncoder(in_vars=pretrain_in_vars,
                                 model_name=args.vit_model,
                                 proj_dim=args.proj_dim).to(device)
    elif arch_key == "conv2d":
        encoder = Conv2DLearnableEncoder(in_vars=pretrain_in_vars,
                                         model_name=args.vit_model,
                                         proj_dim=args.proj_dim).to(device)
    elif arch_key == "tiling_ci":
        encoder = MultiResViTCIEncoder(in_vars=pretrain_in_vars,
                                       model_name=args.vit_model,
                                       proj_dim=args.proj_dim).to(device)
    elif arch_key == "patchtst":
        encoder = PatchTS1DEncoder(in_vars=pretrain_in_vars, d_model=384,
                                   proj_dim=args.proj_dim, use_revin=False).to(device)
    elif arch_key == "utica":
        encoder = UTICAEncoder(in_vars=pretrain_in_vars, d_model=384,
                               proj_dim=args.proj_dim, use_revin=False).to(device)
    elif arch_key == "conv":
        encoder = MultiResViTConvEncoder(in_vars=pretrain_in_vars,
                                         model_name=args.vit_model,
                                         proj_dim=args.proj_dim).to(device)
    elif arch_key == "timesnet":
        encoder = LeJEPATimesModel(in_channels=pretrain_in_vars,
                                   d_model=128, proj_dim=args.proj_dim).to(device)
    elif arch_key == "timesnet_update":
        encoder = LeJEPATimesModel(in_channels=pretrain_in_vars,
                                   d_model=128, proj_dim=args.proj_dim,
                                   norm_type="instance").to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

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

    # ── 3.5 Channel adapter — only for CD architectures ─────────────────────
    if not is_ci and pretrain_in_vars != in_vars:
        channel_adapter = nn.Linear(in_vars, pretrain_in_vars).to(device)
        with torch.no_grad():
            channel_adapter.weight.zero_()
            for j in range(pretrain_in_vars):
                channel_adapter.weight[j, j % in_vars] = 1.0
        print(f"✅ Channel Adapter (CD): {in_vars} → {pretrain_in_vars}")
    else:
        channel_adapter = None
        if is_ci:
            print(f"ℹ️  Channel Adapter: skipped (CI architecture)")
        else:
            print(f"ℹ️  Channel Adapter: skipped (same channel count)")

    # ── 4. Embed dim ─────────────────────────────────────────────────────────
    if arch_key in ("patchtst", "utica"):
        embed_dim = encoder.encoder.d_model if arch_key == "utica" else encoder.d_model
    elif arch_key in ("timesnet", "timesnet_update"):
        embed_dim = 128
    else:
        embed_dim = encoder.backbone.num_features

    # ── 5. Classification head ───────────────────────────────────────────────
    head = LeJEPAClassificationHead(embed_dim, num_classes,
                                    dropout=args.dropout).to(device)
    print(f"✅ ClassificationHead: [B, {embed_dim}] → [B, {num_classes}]")

    # ── 6. Optimizer  (head + optional adapter; encoder frozen by default) ────
    trainable = list(head.parameters())
    if channel_adapter is not None:
        trainable += list(channel_adapter.parameters())
    if args.fine_tune:
        trainable += list(encoder.parameters())

    opt       = torch.optim.RAdam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # ── 7. Training loop ─────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        head.train()
        if channel_adapter is not None: channel_adapter.train()
        if args.fine_tune: encoder.train()

        train_losses, train_preds, train_labels = [], [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)   # [B, C, T]
            batch_y = batch_y.to(device)                          # [B] long

            opt.zero_grad()

            # Channel adaptation (CD only)
            if channel_adapter is not None:
                batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

            x_in = batch_x

            # fine_tune=True: gradients must flow through encoder → no no_grad wrapper
            # fine_tune=False: encoder is frozen → wrap with no_grad for efficiency
            if args.fine_tune:
                emb = extract_emb(encoder, x_in, arch_key)
            else:
                with torch.no_grad():
                    emb = extract_emb(encoder, x_in, arch_key)
            logits = head(emb)                                # [B, num_class]
            loss   = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=4.0)
            opt.step()

            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        avg_train_loss = float(np.mean(train_losses))
        train_acc      = accuracy_score(train_labels, train_preds)
        train_f1       = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        # ── Validation ───────────────────────────────────────────────────────
        head.eval()
        if channel_adapter is not None: channel_adapter.eval()
        if args.fine_tune: encoder.eval()

        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device)

                if channel_adapter is not None:
                    batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

                x_in = batch_x

                emb    = extract_emb(encoder, x_in, arch_key)
                logits = head(emb)
                loss   = criterion(logits, batch_y)

                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_acc      = accuracy_score(val_labels, val_preds)
        val_f1       = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # TensorBoard
        writer_train.add_scalar("Loss",     avg_train_loss, epoch)
        writer_train.add_scalar("Accuracy", train_acc,      epoch)
        writer_train.add_scalar("F1_macro", train_f1,       epoch)
        writer_val.add_scalar("Loss",       avg_val_loss,   epoch)
        writer_val.add_scalar("Accuracy",   val_acc,        epoch)
        writer_val.add_scalar("F1_macro",   val_f1,         epoch)

        # Early Stopping & Checkpoint (Val Loss 기준)
        early_stopping(avg_val_loss, encoder, head, channel_adapter, os.path.join(ckpt_root, "best_model.pt"))
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    # ── 8. Test evaluation ───────────────────────────────────────────────────
    best_ckpt = torch.load(os.path.join(ckpt_root, "best_model.pt"), map_location=device)
    head.load_state_dict(best_ckpt["head"])
    if channel_adapter is not None and best_ckpt.get("adapter") is not None:
        channel_adapter.load_state_dict(best_ckpt["adapter"])
    if "encoder" in best_ckpt:
        encoder.load_state_dict(best_ckpt["encoder"])

    head.eval()
    if channel_adapter is not None: channel_adapter.eval()
    encoder.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)

            if channel_adapter is not None:
                batch_x = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)

            x_in = batch_x

            emb    = extract_emb(encoder, x_in, arch_key)
            logits = head(emb)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds,
                        average="macro", zero_division=0)

    print("=" * 50)
    print(f"Classification Test Result ({args.dataset_name})")
    print(f"Accuracy : {test_acc:.6f}")
    print(f"F1 (macro): {test_f1:.6f}")
    print("=" * 50)

    # TensorBoard — only train/val are streamed; test result goes to summary file only
    # (same philosophy as run_linear_probing_lejepa.py)
    # Summary file — same as run_linear_probing_lejepa.py
    summary_path = os.path.join(ckpt_root, "results_summary.txt")
    with open(summary_path, "a") as f:
        f.write(
            f"Dataset: {args.dataset_name} | Pretrain: {args.pretrain_dataset} | "
            f"Arch: {arch_key} | Acc: {test_acc:.6f} | F1: {test_f1:.6f}\n"
        )

    writer_train.close()
    writer_val.close()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing  (mirrors run_linear_probing_lejepa.py style)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LeJEPA Time-Series Classification")

    # Architecture
    p.add_argument("--arch",             default="timesnet_update")
    p.add_argument("--vit_model",        default="vit_small_patch14_dinov2")
    p.add_argument("--proj_dim",         type=int,   default=128)

    # Pretrain
    p.add_argument("--pretrain_dataset", default="tsld")
    p.add_argument("--pretrain_path",    required=True)

    # Dataset  (data_root: directory with TRAIN.ts/TEST.ts, or a CSV file)
    p.add_argument("--data_root",        required=True,
                   help="Path to dataset directory (.ts) or CSV file")
    p.add_argument("--dataset_name",     default="UCR_dataset",
                   help="Human-readable name used in logs and checkpoint paths")

    # Sequence
    p.add_argument("--seq_len",          type=int,   default=0,
                   help="Sequence length (0 for auto-detect from dataset)")

    # Training
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--fine_tune",        type=lambda x: x.lower() in ("true", "1", "yes"), default=False, help="Unfreeze and fine-tune the encoder")
    p.add_argument("--patience",         type=int,   default=10, help="Early stopping patience")

    # Data
    p.add_argument("--val_ratio",        type=float, default=0.1)
    p.add_argument("--num_workers",      type=int,   default=0)

    # Misc
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--pretrain_in_vars", type=int,   default=0,  # 0 = auto-detect from ckpt
                   help="Override pretrain channel count (use only if auto-detect fails)")

    args = p.parse_args()
    train(args)
