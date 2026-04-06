"""
LeJEPA for Time-Series with TimesNet-style Backbone
=====================================================

Architecture:
  Input [B, T, C]
    └─ InputProjection: Linear(C, D)          -> [B, T, D]
    └─ N x TimesBlock                          -> [B, T, D]
    └─ LayerNorm                               -> [B, T, D]  (= z_seq)
  
  SSL branch:
    z_global_seq, z_local_seq via shared encoder
    u_global_seq, u_local_seq via shared MLPProjector

  Downstream branches:
    ClassificationHead : mean-pool z_seq -> linear
    ForecastHead       : z_seq -> lightweight MLP -> [B, pred_len, C]

Static shape invariant: encoder(x).shape == [B, T, D] for any T.
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def fft_top_k_periods(
    x: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k dominant periods from a sequence via FFT.

    Args:
        x    : [B, T, D]  — input sequence
        top_k: number of dominant frequencies to select

    Returns:
        periods : LongTensor  [top_k]          — period lengths (in time steps)
        weights : FloatTensor [B, top_k]       — softmax-normalised amplitude weights
                                                 (one scalar per branch per sample)
    Notes:
        * We average D-channel amplitudes so that 'importance' is channel-agnostic.
        * Periods are clipped to [2, T] to avoid trivial period-1 or period-0.
        * `top_k` is clamped to the number of available unique frequencies.
    """
    B, T, D = x.shape

    # rfft operates on the last dimension; transpose so T is last
    xf = torch.fft.rfft(x, dim=1, norm="ortho")   # [B, T//2+1, D]
    amp = xf.abs()                                  # [B, T//2+1, D]

    # Average across channel dimension -> [B, T//2+1]
    amp_mean = amp.mean(dim=-1)

    # Frequency indices (0 = DC component — skip it)
    n_freq = amp_mean.shape[1]  # T//2 + 1
    top_k = min(top_k, n_freq - 1)  # guard: never exceed available freqs

    # Average amplitude across batch to obtain a single ranking
    amp_avg = amp_mean.mean(dim=0)          # [T//2+1]
    amp_avg[0] = 0.0                        # suppress DC

    # Top-k frequency indices; shape [top_k]
    _, freq_idx = torch.topk(amp_avg, k=top_k)

    # Convert frequency index -> period length
    # freq_idx == 0 is DC (suppressed), smallest useful index is 1
    # period = round(T / freq_idx), clipped to [2, T]
    periods = (T / freq_idx.float()).round().long()
    periods = periods.clamp(min=2, max=T)   # [top_k]

    # Per-sample branch weights: amplitude at selected frequencies -> softmax
    # amp_mean[:, freq_idx] : [B, top_k]
    branch_amps = amp_mean[:, freq_idx]     # [B, top_k]
    weights = F.softmax(branch_amps, dim=-1)    # [B, top_k]

    return periods, weights


def pad_to_multiple(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """
    Pad a sequence [B, T, D] along the time dimension so that T_pad % period == 0.

    Args:
        x      : [B, T, D]
        period : target period

    Returns:
        x_pad  : [B, T_pad, D]  where T_pad is the smallest multiple of `period` >= T
        T_orig : original T (needed for truncation after processing)
    """
    B, T, D = x.shape
    T_orig = T
    remainder = T % period
    if remainder != 0:
        pad_len = period - remainder
        # Pad by repeating the last time step (replication padding on time axis)
        x = F.pad(x, (0, 0, 0, pad_len))   # (D_left, D_right, T_left, T_right)
    return x, T_orig


# ══════════════════════════════════════════════════════════════════════════════
#  InceptionBlock2D
# ══════════════════════════════════════════════════════════════════════════════

class InceptionBlock2D(nn.Module):
    """
    Lightweight Inception-style 2-D convolution block.

    Input  : [B, D, H, W]
    Output : [B, D, H, W]

    Branches:
        1. 1x1  pointwise
        2. 3x3  local context
        3. 5x5  wider context (implemented as two stacked 3x3 for efficiency)
        4. MaxPool 3x3 + 1x1
    All branches use 'same' padding so spatial dims are preserved.
    Branches are concatenated along channel dim, then projected back to D.
    """

    def __init__(self, in_channels: int, norm_type: str = 'batch', debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        mid = max(in_channels // 4, 1)   # channels per branch
        
        def get_norm(channels: int):
            if norm_type == 'instance':
                return nn.InstanceNorm2d(channels, affine=True)
            return nn.BatchNorm2d(channels)

        # Branch 1: 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            get_norm(mid),
            nn.GELU(),
        )
        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            get_norm(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            get_norm(mid),
            nn.GELU(),
        )
        # Branch 3: 1x1 -> 3x3 -> 3x3  (approximates 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            get_norm(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            get_norm(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            get_norm(mid),
            nn.GELU(),
        )
        # Branch 4: MaxPool + 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            get_norm(mid),
            nn.GELU(),
        )

        # Project concatenated branches [4*mid] back to in_channels [D]
        self.project = nn.Sequential(
            nn.Conv2d(mid * 4, in_channels, kernel_size=1, bias=False),
            get_norm(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, H, W]
        assert x.dim() == 4, f"InceptionBlock2D expects 4-D input, got {x.shape}"

        b1 = self.branch1(x)   # [B, mid, H, W]
        b2 = self.branch2(x)   # [B, mid, H, W]
        b3 = self.branch3(x)   # [B, mid, H, W]
        b4 = self.branch4(x)   # [B, mid, H, W]

        out = torch.cat([b1, b2, b3, b4], dim=1)   # [B, 4*mid, H, W]
        out = self.project(out)                      # [B, D,    H, W]

        if self.debug:
            print(f"  [InceptionBlock2D] in: {x.shape} -> out: {out.shape}")

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TimesBlock
# ══════════════════════════════════════════════════════════════════════════════

class TimesBlock(nn.Module):
    """
    TimesNet-style block operating on sequence representations.

    Processing pipeline per call:
        1. FFT -> top-k dominant periods + per-sample branch weights
        2. For each period p:
              pad x to [B, T_pad, D]   where T_pad % p == 0
              reshape -> [B, D, T_pad//p, p]
              InceptionBlock2D         -> [B, D, T_pad//p, p]
              reshape back -> [B, T_pad, D]
              truncate -> [B, T, D]
        3. Weighted aggregation: sum_k (weight_k * branch_k_out)  -> [B, T, D]
        4. Residual: output = input + aggregated               -> [B, T, D]

    Input  : [B, T, D]
    Output : [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        top_k: int = 5,
        norm_type: str = 'batch',
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.debug = debug

        # One InceptionBlock2D shared across all period branches
        self.inception = InceptionBlock2D(in_channels=d_model, norm_type=norm_type, debug=debug)

        # Light layer norm applied before FFT analysis (stabilises FFT inputs)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        assert x.dim() == 3, f"TimesBlock expects [B, T, D], got {x.shape}"
        B, T, D = x.shape
        assert D == self.d_model, (
            f"TimesBlock d_model mismatch: expected {self.d_model}, got {D}"
        )

        residual = x                        # keep for skip connection

        x_norm = self.norm(x)               # [B, T, D] — normalise before FFT

        # ── Step 1: FFT-based period extraction ──────────────────────────────
        periods, weights = fft_top_k_periods(x_norm, top_k=self.top_k)
        # periods : [top_k_actual]   (might be < self.top_k if T is small)
        # weights : [B, top_k_actual]
        actual_k = periods.shape[0]

        if self.debug:
            print(f"  [TimesBlock] input: {x.shape}")
            print(f"  [TimesBlock] top-{actual_k} periods: {periods.tolist()}")

        # ── Step 2: period-wise 2-D convolution ──────────────────────────────
        branch_outputs = []
        for i in range(actual_k):
            p = periods[i].item()           # scalar Python int

            # Pad so T_pad is divisible by p
            x_pad, T_orig = pad_to_multiple(x_norm, p)
            T_pad = x_pad.shape[1]

            rows = T_pad // p               # number of 'rows' in 2D view

            # [B, T_pad, D] -> [B, D, rows, p]
            # Permute to [B, D, T_pad] then view to [B, D, rows, p]
            x2d = x_pad.permute(0, 2, 1).contiguous()          # [B, D, T_pad]
            x2d = x2d.view(B, D, rows, p)                      # [B, D, rows, p]

            if self.debug:
                print(f"  [TimesBlock] period={p}, pad: {T_pad}, 2D shape: {x2d.shape}")

            # Apply InceptionBlock2D
            x2d = self.inception(x2d)                           # [B, D, rows, p]

            # [B, D, rows, p] -> [B, D, T_pad] -> [B, T_pad, D]
            x2d = x2d.view(B, D, T_pad).permute(0, 2, 1)      # [B, T_pad, D]

            # Truncate back to original T
            x2d = x2d[:, :T_orig, :]                            # [B, T, D]

            branch_outputs.append(x2d)

        # ── Step 3: weighted aggregation ─────────────────────────────────────
        # weights: [B, actual_k] -> expand to [B, 1, actual_k] for broadcasting
        # stack branch_outputs -> [B, T, actual_k, D]
        branches_stacked = torch.stack(branch_outputs, dim=2)   # [B, T, actual_k, D]
        w = weights.unsqueeze(1).unsqueeze(-1)                   # [B, 1, actual_k, 1]
        aggregated = (branches_stacked * w).sum(dim=2)           # [B, T, D]

        if self.debug:
            print(f"  [TimesBlock] aggregated: {aggregated.shape}")

        # ── Step 4: residual ─────────────────────────────────────────────────
        out = residual + aggregated                              # [B, T, D]

        assert out.shape == (B, T, D), (
            f"TimesBlock output shape mismatch: expected ({B},{T},{D}), got {out.shape}"
        )
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TimesEncoder
# ══════════════════════════════════════════════════════════════════════════════

class TimesEncoder(nn.Module):
    """
    Full Times-style encoder: InputProjection + stack of TimesBlocks + LayerNorm.

    Input  : [B, T, C]
    Output : [B, T, D]   (sequence representation — NOT pooled)

    Note: no absolute positional embeddings are added by default.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_blocks: int = 3,
        top_k: int = 5,
        norm_type: str = 'batch',
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug

        # Linear projection C -> D
        self.input_proj = nn.Linear(in_channels, d_model)

        # Stack of TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(d_model=d_model, top_k=top_k, norm_type=norm_type, debug=debug)
            for _ in range(num_blocks)
        ])

        # Final normalisation
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        assert x.dim() == 3, f"TimesEncoder expects [B, T, C], got {x.shape}"

        z = self.input_proj(x)          # [B, T, D]

        if self.debug:
            print(f"[TimesEncoder] after InputProjection: {z.shape}")

        for idx, block in enumerate(self.blocks):
            z = block(z)                # [B, T, D]
            if self.debug:
                print(f"[TimesEncoder] after TimesBlock[{idx}]: {z.shape}")

        z = self.norm(z)                # [B, T, D]

        if self.debug:
            print(f"[TimesEncoder] final output (z_seq): {z.shape}")

        return z                        # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
#  MLPProjector  (SSL branch only — shared for global & local)
# ══════════════════════════════════════════════════════════════════════════════

class MLPProjector(nn.Module):
    """
    Lightweight 2-layer MLP projector for the self-supervised branch.

    Operates token-wise: each time step t is projected independently.
    Input  : [B, T, D]
    Output : [B, T, proj_dim]

    This is intentionally kept separate from any downstream head.
    """

    def __init__(
        self,
        d_model: int,
        proj_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or (d_model * 2)

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: [B, T, D]
        assert z_seq.dim() == 3, (
            f"MLPProjector expects [B, T, D], got {z_seq.shape}"
        )
        return self.net(z_seq)          # [B, T, proj_dim]


# ══════════════════════════════════════════════════════════════════════════════
#  ClassificationHead
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    Downstream classification head.

    Pipeline:
        z_seq [B, T, D]
        -> mean pool over T
        -> [B, D]
        -> Linear(D, num_classes)
        -> [B, num_classes]

    Pooling is placed here (NOT inside the encoder) to preserve the
    encoder's sequence-level output invariant.
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: [B, T, D]
        assert z_seq.dim() == 3, (
            f"ClassificationHead expects [B, T, D], got {z_seq.shape}"
        )
        # Mean-pool over time dimension
        pooled = z_seq.mean(dim=1)      # [B, D]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)        # [B, num_classes]
        return logits


# ══════════════════════════════════════════════════════════════════════════════
#  ForecastHead
# ══════════════════════════════════════════════════════════════════════════════

class ForecastHead(nn.Module):
    """
    Lightweight downstream forecasting head.

    Design rationale:
        We use the full sequence z_seq [B, T, D] rather than only the
        last token, because TimesBlock captures periodic patterns globally
        across the sequence.  A compact linear layer maps the flattened
        temporal representation to the forecast horizon.

        Specifically:
            z_seq [B, T, D]
            -> permute + Linear(T, pred_len)          [B, D, pred_len]    (temporal mapping)
            -> permute                                 [B, pred_len, D]
            -> Linear(D, out_channels)                [B, pred_len, C]

        This two-step linear head avoids heavy decoder architectures while
        still using the entire sequence context.

    Input  : [B, T, D]
    Output : [B, pred_len, C_out]
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        out_channels: int,
        input_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.input_len = input_len

        self.dropout = nn.Dropout(dropout)

        # Temporal projection: maps T time steps to pred_len steps, per channel
        self.temporal_proj = nn.Linear(input_len, pred_len)

        # Channel projection: D hidden -> C output channels
        self.channel_proj = nn.Linear(d_model, out_channels)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: [B, T, D]
        assert z_seq.dim() == 3, (
            f"ForecastHead expects [B, T, D], got {z_seq.shape}"
        )
        B, T, D = z_seq.shape

        # Gracefully handle variable-length inputs at inference by interpolating
        # the temporal projection weights.  During training T == self.input_len.
        if T != self.input_len:
            # Interpolate temporal dimension to self.input_len using adaptive pooling
            z_interp = F.adaptive_avg_pool1d(
                z_seq.permute(0, 2, 1),           # [B, D, T]
                output_size=self.input_len,
            ).permute(0, 2, 1)                    # [B, input_len, D]
        else:
            z_interp = z_seq                      # [B, T, D]

        z_interp = self.dropout(z_interp)

        # [B, input_len, D] -> [B, D, input_len] -> temporal_proj -> [B, D, pred_len]
        out = self.temporal_proj(
            z_interp.permute(0, 2, 1)            # [B, D, input_len]
        )                                          # [B, D, pred_len]

        # [B, D, pred_len] -> [B, pred_len, D]
        out = out.permute(0, 2, 1)                # [B, pred_len, D]

        # [B, pred_len, D] -> [B, pred_len, C_out]
        out = self.channel_proj(out)              # [B, pred_len, C_out]

        assert out.shape == (B, self.pred_len, self.out_channels), (
            f"ForecastHead output shape mismatch: expected "
            f"({B}, {self.pred_len}, {self.out_channels}), got {out.shape}"
        )
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  LeJEPATimesModel — top-level module
# ══════════════════════════════════════════════════════════════════════════════

class LeJEPATimesModel(nn.Module):
    """
    LeJEPA for Time-Series with TimesNet-style backbone.

    Components
    ----------
    encoder         : TimesEncoder  — ONE shared instance for all views
    projector       : MLPProjector  — SSL branch only
    cls_head        : ClassificationHead  — downstream classification
    forecast_head   : ForecastHead        — downstream forecasting

    Key invariants
    --------------
    * encoder is shared between global/local views (no separate instances).
    * encoder output is always [B, T, D] (sequence, never pooled).
    * projector is distinct from downstream heads (no merged logic).
    * no absolute positional embeddings in the encoder by default.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        num_blocks: int = 3,
        top_k: int = 5,
        proj_dim: int = 64,
        num_classes: int = 10,
        pred_len: int = 96,
        input_len: int = 96,
        dropout: float = 0.1,
        norm_type: str = 'batch',
        debug: bool = False,
    ) -> None:
        super().__init__()

        # ── Shared encoder (single instance) ──────────────────────────────
        self.encoder = TimesEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_blocks=num_blocks,
            top_k=top_k,
            norm_type=norm_type,
            debug=debug,
        )

        # ── SSL branch ────────────────────────────────────────────────────
        self.projector = MLPProjector(
            d_model=d_model,
            proj_dim=proj_dim,
            hidden_dim=d_model * 2,
        )

        # ── Downstream heads ──────────────────────────────────────────────
        self.cls_head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.forecast_head = ForecastHead(
            d_model=d_model,
            pred_len=pred_len,
            out_channels=in_channels,
            input_len=input_len,
            dropout=dropout,
        )

    # ── SSL forward ───────────────────────────────────────────────────────────

    def forward_ssl(
        self,
        x_global: torch.Tensor,
        x_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Self-supervised forward pass.

        Args:
            x_global : [B, T_g, C]
            x_local  : [B, T_l, C]

        Returns:
            z_global_seq : [B, T_g, D]  — encoder output for global view
            z_local_seq  : [B, T_l, D]  — encoder output for local view
            u_global_seq : [B, T_g, proj_dim]  — projector output for global
            u_local_seq  : [B, T_l, proj_dim]  — projector output for local

        Both views pass through the SAME encoder instance to ensure
        representations live in a shared latent space.
        """
        assert x_global.dim() == 3, (
            f"forward_ssl: x_global must be [B, T_g, C], got {x_global.shape}"
        )
        assert x_local.dim() == 3, (
            f"forward_ssl: x_local must be [B, T_l, C], got {x_local.shape}"
        )
        assert x_global.shape[0] == x_local.shape[0], (
            "forward_ssl: batch size of global/local must match"
        )
        assert x_global.shape[2] == x_local.shape[2], (
            "forward_ssl: channel dimension of global/local must match"
        )

        # Shared encoder — same instance, separate calls
        z_global_seq = self.encoder(x_global)   # [B, T_g, D]
        z_local_seq  = self.encoder(x_local)    # [B, T_l, D]

        # Shared projector — same instance, separate calls
        u_global_seq = self.projector(z_global_seq)  # [B, T_g, proj_dim]
        u_local_seq  = self.projector(z_local_seq)   # [B, T_l, proj_dim]

        return z_global_seq, z_local_seq, u_global_seq, u_local_seq

    # ── Downstream forwards ───────────────────────────────────────────────────

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downstream classification forward.

        Args:
            x : [B, T, C]

        Returns:
            logits : [B, num_classes]
        """
        assert x.dim() == 3, (
            f"forward_classification: input must be [B, T, C], got {x.shape}"
        )
        z_seq  = self.encoder(x)            # [B, T, D]
        logits = self.cls_head(z_seq)       # [B, num_classes]
        return logits

    def forward_forecasting(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downstream forecasting forward.

        Args:
            x : [B, T, C]

        Returns:
            forecast : [B, pred_len, C]
        """
        assert x.dim() == 3, (
            f"forward_forecasting: input must be [B, T, C], got {x.shape}"
        )
        z_seq    = self.encoder(x)              # [B, T, D]
        forecast = self.forecast_head(z_seq)    # [B, pred_len, C]
        return forecast

    def forward(self, views) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        General forward.
        If input is a dict of views, behaves like a LeJEPA encoder returning (all_emb, proj).
        Otherwise returns sequence representation [B, T, D].
        """
        if isinstance(views, dict):
            B = views['global'].shape[0]
            
            # [B, 2, C, T_g] -> [B, 2, T_g, C] -> [B*2, T_g, C]
            g_in = views['global'].permute(0, 1, 3, 2).flatten(0, 1)
            l_in = views['local'].permute(0, 1, 3, 2).flatten(0, 1)
            
            # Forward shared encoder
            z_g_seq = self.encoder(g_in)  # [B*2, T_g, D]
            z_l_seq = self.encoder(l_in)  # [B*6, T_l, D]
            
            # Forward shared projector
            u_g_seq = self.projector(z_g_seq)  # [B*2, T_g, proj_dim]
            u_l_seq = self.projector(z_l_seq)  # [B*6, T_l, proj_dim]
            
            # Pool temporally to produce view-level tokens for identical shape matching
            z_g = z_g_seq.mean(dim=1)  # [B*2, D]
            z_l = z_l_seq.mean(dim=1)  # [B*6, D]
            u_g = u_g_seq.mean(dim=1)  # [B*2, proj_dim]
            u_l = u_l_seq.mean(dim=1)  # [B*6, proj_dim]
            
            # Reshape back to [B, num_views, ...]
            all_emb = torch.cat([z_g.reshape(B, 2, -1), z_l.reshape(B, 6, -1)], dim=1)  # [B, 8, D]
            
            u_g_reshaped = u_g.reshape(B, 2, -1)
            u_l_reshaped = u_l.reshape(B, 6, -1)
            proj = torch.cat([u_g_reshaped, u_l_reshaped], dim=1).transpose(0, 1)  # [8, B, proj_dim]
            
            return all_emb, proj
        else:
            return self.encoder(views)


# ══════════════════════════════════════════════════════════════════════════════
#  Runnable example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("  LeJEPA-TimesNet  — Runnable Sanity Check")
    print("=" * 65)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device : {device}")

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    B          = 2         # batch size
    T_global   = 96        # global view length
    T_local    = 48        # local view length
    C          = 7         # input channels (variables)
    D          = 128       # model hidden dim
    PROJ_DIM   = 64        # SSL projection dim
    NUM_CLS    = 10        # classification classes
    PRED_LEN   = 24        # forecasting horizon
    TOP_K      = 5         # top-k periods in TimesBlock
    NUM_BLOCKS = 3         # number of TimesBlocks
    DEBUG      = True      # set True to see intermediate shapes

    # ── Dummy tensors ─────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x_global   = torch.randn(B, T_global, C, device=device)
    x_local    = torch.randn(B, T_local,  C, device=device)
    x_cls      = torch.randn(B, T_global, C, device=device)
    x_forecast = torch.randn(B, T_global, C, device=device)

    print(f"\n  Input shapes:")
    print(f"    x_global   : {tuple(x_global.shape)}")
    print(f"    x_local    : {tuple(x_local.shape)}")
    print(f"    x_cls      : {tuple(x_cls.shape)}")
    print(f"    x_forecast : {tuple(x_forecast.shape)}")

    # ── Instantiate model ─────────────────────────────────────────────────────
    model = LeJEPATimesModel(
        in_channels=C,
        d_model=D,
        num_blocks=NUM_BLOCKS,
        top_k=TOP_K,
        proj_dim=PROJ_DIM,
        num_classes=NUM_CLS,
        pred_len=PRED_LEN,
        input_len=T_global,
        dropout=0.1,
        debug=DEBUG,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # ── Forward: SSL ──────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  [1] forward_ssl")
    print("-" * 65)
    model.eval()
    with torch.no_grad():
        z_global_seq, z_local_seq, u_global_seq, u_local_seq = model.forward_ssl(
            x_global, x_local
        )

    print(f"\n  SSL output shapes:")
    print(f"    z_global_seq : {tuple(z_global_seq.shape)}  expected [B={B}, T_g={T_global}, D={D}]")
    print(f"    z_local_seq  : {tuple(z_local_seq.shape)}  expected [B={B}, T_l={T_local}, D={D}]")
    print(f"    u_global_seq : {tuple(u_global_seq.shape)}  expected [B={B}, T_g={T_global}, proj={PROJ_DIM}]")
    print(f"    u_local_seq  : {tuple(u_local_seq.shape)}  expected [B={B}, T_l={T_local}, proj={PROJ_DIM}]")

    assert z_global_seq.shape == (B, T_global, D)
    assert z_local_seq.shape  == (B, T_local,  D)
    assert u_global_seq.shape == (B, T_global, PROJ_DIM)
    assert u_local_seq.shape  == (B, T_local,  PROJ_DIM)
    print("  ✓ SSL shapes verified")

    # ── Forward: Classification ───────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  [2] forward_classification")
    print("-" * 65)
    with torch.no_grad():
        logits = model.forward_classification(x_cls)

    print(f"\n  Classification output:")
    print(f"    logits : {tuple(logits.shape)}  expected [B={B}, num_classes={NUM_CLS}]")

    assert logits.shape == (B, NUM_CLS)
    print("  ✓ Classification shape verified")

    # ── Forward: Forecasting ──────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  [3] forward_forecasting")
    print("-" * 65)
    with torch.no_grad():
        forecast = model.forward_forecasting(x_forecast)

    print(f"\n  Forecasting output:")
    print(f"    forecast : {tuple(forecast.shape)}  expected [B={B}, pred_len={PRED_LEN}, C={C}]")

    assert forecast.shape == (B, PRED_LEN, C)
    print("  ✓ Forecasting shape verified")

    # ── Shared encoder check ─────────────────────────────────────────────────
    # Verify that global and local use the exact same encoder object
    print("\n" + "-" * 65)
    print("  [4] Shared encoder identity check")
    print("-" * 65)
    # Both calls in forward_ssl touch model.encoder — no separate encoder exists
    assert len([m for m in model.children() if isinstance(m, TimesEncoder)]) == 1, \
        "There must be exactly one TimesEncoder in the model!"
    print("  ✓ Single shared TimesEncoder confirmed")

    print("\n" + "=" * 65)
    print("  All checks passed.")
    print("=" * 65)
