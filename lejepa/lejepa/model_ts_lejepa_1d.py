import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from typing import Optional


class PatchEmbedding1D(nn.Module):
    """PatchTST 방식 1D 패치 임베딩 (Reshape + Linear).

    Conv1d 없이, 시퀀스를 patch_size 단위로 분할한 뒤
    nn.Linear로 d_model 차원으로 투영합니다.
    PatchTST 논문의 W_P = nn.Linear(patch_len, d_model)과 동일한 구조입니다.

    Args:
        patch_size: 한 패치의 시간 프레임 수 (= patch_len).
        d_model: 투영 후 임베딩 차원.
    """

    def __init__(self, patch_size: int = 16, d_model: int = 384):
        super().__init__()
        self.patch_size = patch_size
        # PatchTST W_P에 해당하는 선형 투영: patch_len → d_model
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] — 채널 독립(CI) 처리 후 단일 채널 시계열
        Returns:
            [B, N, d_model] — N = T_padded // patch_size
        """
        T = x.shape[-1]

        # T가 patch_size의 배수가 아닐 경우 우측 zero-padding (정보 유실 방지)
        remainder = T % self.patch_size
        if remainder > 0:
            x = F.pad(x, (0, self.patch_size - remainder))

        # Reshape: [B, T_padded] → [B, N, patch_size]
        N = x.shape[-1] // self.patch_size
        x = x.reshape(x.shape[0], N, self.patch_size)  # [B, N, patch_size]

        # Linear 투영: [B, N, patch_size] → [B, N, d_model]
        return self.proj(x)



class PatchTS1DEncoder(nn.Module):
    """PatchTST 스타일 순수 1D Transformer 기반 DINO 인코더.

    ViT 백본 의존 없이 1D Conv Patch Embedding + 표준 Transformer Encoder를 사용합니다.
    Channel Independent(CI) 방식을 적용하여 채널 C를 배치 차원으로 합칩니다.

    Args:
        in_vars: 입력 시계열 채널(변수) 수.
        d_model: Transformer 임베딩 차원.
        patch_size: 1D 패치 크기 (시간 프레임 단위).
        n_heads: Multi-Head Attention 헤드 수.
        n_layers: Transformer Encoder 레이어 수.
        proj_dim: 최종 Projection Head 출력 차원.
        dropout: Dropout 비율.
    """

    def __init__(
        self,
        in_vars: int,
        d_model: int = 384,
        patch_size: int = 16,
        n_heads: int = 6,
        n_layers: int = 8,
        proj_dim: int = 128,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        self.in_vars = in_vars
        self.d_model = d_model
        self.patch_size = patch_size
        self.use_revin = use_revin

        # 1. 1D 패치 임베딩 (PatchTST 방식: Reshape + Linear)
        self.patch_embed = PatchEmbedding1D(patch_size=patch_size, d_model=d_model)

        # 3. 표준 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # [B, N, D] 형식
            norm_first=True,    # Pre-Norm (안정적 학습)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # 4. Projection Head (SIGReg 손실 연산용)
        # BatchNorm1d로 인한 Val S 손실 폭발 방지를 위해 LayerNorm 적용
        self.proj_head = MLP(d_model, [d_model * 4, d_model * 4, proj_dim], norm_layer=nn.LayerNorm)

    def get_1d_sincos_pos_embed(
        self,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Generate 1D Sin/Cos Positional Embedding for given positional grid.

        Args:
            grid: [N] or [B, N] positional values (fractional patches).
        """
        device = grid.device
        omega = torch.arange(self.d_model // 2, dtype=torch.float32, device=device)
        omega = 1.0 / (10000 ** (omega * 2.0 / self.d_model))  # [D/2]

        if grid.dim() == 1:
            # [N, D/2]
            out = torch.einsum("m,d->md", grid, omega)
            emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, D]
            return emb.unsqueeze(0)  # [1, N, D]
        else:
            # [B, N, D/2]
            out = torch.einsum("bn,d->bnd", grid, omega)
            emb = torch.cat([torch.sin(out), torch.cos(out)], dim=2)  # [B, N, D]
            return emb


    def _process(
        self,
        x: torch.Tensor,
        max_len: int,
        offsets: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a single View type (Global or Local) through the encoder.

        Args:
            x: [B_v, 1, C, T] — B_v = Batch * n_views, T varies by view type.
            max_len: Full sequence length of the Global View (512).
            offsets: [B_v] long tensor — timestep-level crop start positions.
            lengths: [B_v] long tensor — timestep-level crop lengths.
        Returns:
            [B_v * C, d_model]
        """
        B_v, _, C, T = x.shape

        # Step 0. 채널 독립(CI): [B_v, 1, C, T] -> [B_v*C, T]
        x = x.squeeze(1).reshape(B_v * C, T)

        # Step 0.5 Instance Normalization (RevIN) — use_revin=True 일 때만 적용
        if self.use_revin:
            mu = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - mu) / std

        # Step 1. Patch Embedding: [B_v*C, T] -> [B_v*C, N, d_model]
        x = self.patch_embed(x)
        N = x.shape[1]

        # Step 2. Offset-Aware Positional Encoding
        if offsets is None:
            # Global View (Standard): 0번부터 시작하는 정수 그리드
            grid = torch.arange(N, dtype=torch.float32, device=x.device)
            pos_embed = self.get_1d_sincos_pos_embed(grid)  # [1, N, D]
        else:
            # Batch-wise Positional Encoding
            if lengths is not None:
                # Resize 크롭 대응: Fractional Grid 계산
                # start_patch + rel * (crop_len_patch - 1)
                rel = torch.linspace(0, 1, steps=N, device=x.device) # [N]
                start_p = offsets.view(-1, 1) / self.patch_size       # [B_v, 1]
                span_p = lengths.view(-1, 1) / self.patch_size        # [B_v, 1]
                grid = start_p + rel.view(1, -1) * (span_p - 1)      # [B_v, N]
            else:
                # 단순 Offset만 있는 경우 (정수 그리드 시프트)
                grid = offsets.view(-1, 1) / self.patch_size + torch.arange(N, device=x.device).view(1, -1)
            
            pe = self.get_1d_sincos_pos_embed(grid)  # [B_v, N, D]
            # 채널 차원으로 확장: [B_v, N, D] -> [B_v, C, N, D] -> [B_v*C, N, D]
            pos_embed = pe.unsqueeze(1).expand(-1, C, -1, -1).reshape(B_v * C, N, -1)

        x = x + pos_embed


        # Step 3. Transformer 인코더 통과
        x = self.norm(self.transformer(x))  # [B_v*C, N, d_model]

        # Step 4. Global Average Pooling
        return x.mean(dim=1)  # [B_v*C, d_model]

    def forward(self, views: dict) -> tuple:
        """Multi-Resolution Views forward pass.

        Args:
            views: {
                'global': [B, 2, C, T_global=512],
                'local':  [B, 6, C, T_local=256],
                'local_offsets': [B, 6]  — timestep-level crop start positions
            }
        Returns:
            all_emb: [B*C, 8, d_model]
            proj:    [8, B*C, proj_dim]
        """
        B = views["global"].shape[0]
        C = views["global"].shape[2]
        max_len = views["global"].shape[-1]  # 512

        # View 차원을 Batch 차원으로 폄치기
        g_in = views["global"].flatten(0, 1).unsqueeze(1)   # [B*2, 1, C, 512]
        l_in = views["local"].flatten(0, 1).unsqueeze(1)    # [B*6, 1, C, 256]

        # Offsets & Lengths 추출
        g_offsets = views["global_offsets"].flatten(0, 1) if "global_offsets" in views else None
        g_lengths = views["global_lengths"].flatten(0, 1) if "global_lengths" in views else None
        l_offsets = views["local_offsets"].flatten(0, 1) if "local_offsets" in views else None
        l_lengths = views["local_lengths"].flatten(0, 1) if "local_lengths" in views else None

        # Global: offset/scale 반영
        g_emb = self._process(g_in, max_len=max_len, offsets=g_offsets, lengths=g_lengths)          # [B*2*C, d_model]
        # Local: 실제 크롭된 위치 offset/scale 반영
        l_emb = self._process(l_in, max_len=max_len, offsets=l_offsets, lengths=l_lengths)          # [B*6*C, d_model]


        # SIGReg 연산을 위해 (B*C)를 하나의 배치 단위로 묶고 V개 합산
        g_emb = g_emb.reshape(B * 2, C, -1).transpose(0, 1).reshape(B * C, 2, -1)  # [B*C, 2, d_model]
        l_emb = l_emb.reshape(B * 6, C, -1).transpose(0, 1).reshape(B * C, 6, -1)  # [B*C, 6, d_model]

        all_emb = torch.cat([g_emb, l_emb], dim=1)   # [B*C, 8, d_model]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B * C, 8, -1).transpose(0, 1)
        # proj: [8, B*C, proj_dim]

        return all_emb, proj
