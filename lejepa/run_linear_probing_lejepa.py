import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from lejepa.data_ts_lejepa_downstream import get_downstream_loaders
from lejepa.arch_registry import (
    SUPPORTED_ARCHS,
    build_encoder,
    get_embed_dim,
    infer_pretrain_in_vars,
    uses_ci_decoder,
    validate_arch,
)
from lejepa.revin import RevIN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class LeJEPALinearProbingHead(nn.Module):
    def __init__(self, in_vars, embed_dim, pred_len):
        super().__init__()
        # LeJEPA 백본은 C차원을 Projection해서 [B, Embed_Dim] 1개 백터를 내뱉습니다.
        # 타겟은 [B, C, Pred_Len]이므로 Linear 계층 1개로 한 번에 복원합니다.
        self.head = nn.Linear(embed_dim, in_vars * pred_len)
        self.in_vars = in_vars
        self.pred_len = pred_len
        
    def forward(self, x):
        # x: [B, Embed]
        # output: [B, C, Pred_Len]
        out = self.head(x)
        return out.view(-1, self.in_vars, self.pred_len)

class LeJEPALinearProbingHeadCI(nn.Module):
    def __init__(self, in_vars, embed_dim, pred_len):
        super().__init__()
        # 1채널 임베딩을 받아 1채널 길이로 출력
        self.head = nn.Linear(embed_dim, pred_len)
        self.in_vars = in_vars
        self.pred_len = pred_len
        
    def forward(self, x):
        # x: [B * C, Embed]
        out = self.head(x)
        # output: [B, C, Pred_Len]
        return out.view(-1, self.in_vars, self.pred_len)


def extract_emb(encoder, x_in, arch_key):
    if arch_key == "timevlm":
        emb = encoder._process(x_in, target_res=224, training=False, is_downstream=True)
    elif arch_key in ["patchtst", "utica"]:
        actual_enc = encoder.encoder if arch_key == "utica" else encoder
        emb = actual_enc._process(x_in.unsqueeze(1), max_len=512, offsets=None)
    elif arch_key == "timesnet":
        seq = encoder(x_in.transpose(1, 2))
        emb = seq.mean(dim=1)
    else:
        emb = encoder._process(x_in.unsqueeze(1), target_res=224, training=False, is_downstream=True)

    if arch_key == "tivit_indep":
        emb = emb.mean(1)

    return emb

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.arch = validate_arch(args.arch)
    
    # 1. DataLoader
    train_loader, val_loader, test_loader, scaler = get_downstream_loaders(
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_workers=args.num_workers
    )
    
    first_batch_c, _ = next(iter(train_loader))
    in_vars = first_batch_c.shape[1] # Target Data Channels (예: 7)
    
    arch_key = args.arch
    
    # 2. Pretrained Dataset 차원(pretrain_in_vars) 동적 확인
    state_dict = None
    if os.path.exists(args.pretrain_path):
        ckpt = torch.load(args.pretrain_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    pretrain_in_vars = infer_pretrain_in_vars(
        arch_key,
        state_dict,
        in_vars,
        args.pretrain_dataset,
    )
    
    print("="*50)
    print(f"🚀 Architecture: {args.arch.upper()}")
    print(f"📊 Target Dataset: {args.dataset_type} (Channels: {in_vars})")
    print(f"📊 Pretrain Info: {args.pretrain_dataset} (Detected Channels: {pretrain_in_vars})")
    print(f"📊 Prediction Length: {args.pred_len}")
    print("="*50)
    
    # 2.5 Experiment ID & Directory Structure
    # 아키텍처, 타겟 데이터셋, 체크포인트 타입을 조합하여 고유 실험 ID 생성 (pred_len은 제외하여 같은 Run에 기록)
    ckpt_type = os.path.basename(args.pretrain_path).split('_')[2] if 'best' in args.pretrain_path else "total"
    exp_id = f"LeJEPA_{args.pretrain_dataset}_to_{args.dataset_type}_{arch_key}_{ckpt_type}"
    
    if args.log_dir:
        runs_root = os.path.join(args.log_dir, 'runs')
        ckpt_root = args.log_dir
    else:
        runs_root = os.path.join('runs', exp_id)
        ckpt_root = os.path.join('checkpoints', 'linear_probing', exp_id)
    
    runs_dir_train = os.path.join(runs_root, 'train')
    runs_dir_val = os.path.join(runs_root, 'val')
    os.makedirs(runs_dir_train, exist_ok=True)
    os.makedirs(runs_dir_val, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)
    
    writer_train = SummaryWriter(runs_dir_train)
    writer_val = SummaryWriter(runs_dir_val)
    
    # 3. Pretrained Encoder 로드 & Freeze
    encoder = build_encoder(
        arch_key,
        in_vars=pretrain_in_vars,
        vit_model=args.vit_model,
        proj_dim=args.proj_dim,
        use_revin=False,
    ).to(device)
    
    if state_dict is not None:
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        matched_keys = len(state_dict) - len(missing)
        print(f"✅ Loaded pretrained encoder from {args.pretrain_path}")
        print(f"   - {matched_keys}/{len(state_dict)} keys matched.")
        if missing:
            print(f"   ⚠️ Missing keys (not loaded): {missing[:3]}...")
    else:
        print(f"⚠️ Pretrained path not found: {args.pretrain_path}")
        
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # Channel adapter는 완전히 비활성화한다.
    print("ℹ️ Channel Adapter: disabled")

    # 4. Probing Head & RevIN
    embed_dim = get_embed_dim(encoder, arch_key)
        
    if uses_ci_decoder(arch_key):
        decoder = LeJEPALinearProbingHeadCI(in_vars, embed_dim, args.pred_len).to(device)
        print(f"✅ Probing Head [CI]: [B*C, {embed_dim}] -> [B, {in_vars}, {args.pred_len}]")
    else:
        decoder = LeJEPALinearProbingHead(in_vars, embed_dim, args.pred_len).to(device)
        print(f"✅ Probing Head: [B, {embed_dim}] -> [B, {in_vars}, {args.pred_len}]")
    
    revin = None
    if args.use_revin:
        revin = RevIN(num_features=in_vars, affine=False).to(device)
    
    # 5. Optimizer (Decoder만 학습)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # 6. Training Loop
    import time
    for epoch in range(args.epochs):
        decoder.train()
            
        train_loss = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
            
            opt.zero_grad()
            if revin is not None: batch_x = revin(batch_x, mode='norm')
            batch_x_adapted = batch_x
            
            # Encoder forward
            with torch.no_grad():
                if batch_x_adapted.shape[-1] > 512:
                    x_in = batch_x_adapted[:, :, -512:]
                else:
                    x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
                
                emb = extract_emb(encoder, x_in, arch_key)
                
            preds = decoder(emb)
            if revin is not None: preds = revin(preds, mode='denorm')
            
            loss = criterion(preds, batch_y)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            
        avg_train_loss = np.mean(train_loss)
        
        # Validation
        decoder.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
                if revin is not None: batch_x = revin(batch_x, mode='norm')
                batch_x_adapted = batch_x
                
                if batch_x_adapted.shape[-1] > 512:
                    x_in = batch_x_adapted[:, :, -512:]
                else:
                    x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
                
                emb = extract_emb(encoder, x_in, arch_key)
                preds = decoder(emb)
                if revin is not None: preds = revin(preds, mode='denorm')
                val_loss.append(criterion(preds, batch_y).item())
                
        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer_train.add_scalar(f"Loss_pred{args.pred_len}", avg_train_loss, epoch)
        writer_val.add_scalar(f"Loss_pred{args.pred_len}", avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'decoder': decoder.state_dict()},
                       os.path.join(ckpt_root, f"best_model_{args.pred_len}.pt"))
            
    # 7. Test Evaluation
    best_ckpt = torch.load(os.path.join(ckpt_root, f"best_model_{args.pred_len}.pt"))
    decoder.load_state_dict(best_ckpt['decoder'])
    decoder.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
            if revin is not None: batch_x = revin(batch_x, mode='norm')
            batch_x_adapted = batch_x
                
            if batch_x_adapted.shape[-1] > 512:
                x_in = batch_x_adapted[:, :, -512:]
            else:
                x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
            
            emb = extract_emb(encoder, x_in, arch_key)
            preds = decoder(emb)
            if revin is not None: preds = revin(preds, mode='denorm')
            
            preds_np, trues_np = preds.cpu().numpy(), batch_y.cpu().numpy()
            all_preds.append(preds_np)   # [B, C, Pred_len] 누적
            all_trues.append(trues_np)   # [B, C, Pred_len] 누적
    
    # 전체 테스트셋으로 concat 후 최종 MSE/MAE 계산
    all_preds = np.concatenate(all_preds, axis=0)  # [N, C, Pred_len]
    all_trues = np.concatenate(all_trues, axis=0)  # [N, C, Pred_len]
    final_mse = mse(all_trues, all_preds)
    final_mae = mae(all_trues, all_preds)
    
    print("="*50)
    print(f"Zero-Shot Forecasting Test Result ({args.dataset_type}, Pred={args.pred_len})")
    print(f"MSE: {final_mse:.6f} | MAE: {final_mae:.6f}")
    print("="*50)
    
    # 텍스트 파일 저장 (summary) - ckpt_root 내부에 저장
    summary_path = os.path.join(ckpt_root, "results_summary.txt")
    with open(summary_path, "a") as f:
        f.write(f"Target: {args.target_dataset} | Pretrain: {args.pretrain_dataset} | Pred_len: {args.pred_len:4d} | MSE: {final_mse:.6f} | MAE: {final_mae:.6f}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="basic", help=f"Supported: {', '.join(SUPPORTED_ARCHS)}")
    p.add_argument("--pretrain_dataset", default="electricity")
    p.add_argument("--target_dataset", default="ETTm1")
    p.add_argument("--dataset_type", default="ETTm1")
    p.add_argument("--data_path", required=True)
    p.add_argument("--pretrain_path", required=True)
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, default=96)
    
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4) # Linear Probing Learning rate
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--vit_model", default="vit_small_patch14_dinov2")
    p.add_argument("--use_revin", type=lambda x: x.lower() in ("true", "1", "yes", "none"), default=False)
    
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default=None, help="Optional output directory override for checkpoints and TensorBoard logs")
    
    args = p.parse_args()
    
    # dataset_type과 target_dataset을 일치시킴
    args.dataset_type = args.target_dataset
    
    train(args)
