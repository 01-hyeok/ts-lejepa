import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from timm.layers import resample_abs_pos_embed

class SIGReg(nn.Module):
    def __init__(self, knots: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t); self.register_buffer("phi", window); self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        V, N, D = proj.shape
        A = torch.randn(D, 256, device=proj.device); A = A.div_(A.norm(p=2, dim=0))
        # x_t: [V, N, 256, Knots]
        x_t = (proj @ A).unsqueeze(-1) * self.t
        # 배치(N, dim=1)에 대해 평균을 내어 분포가 가우시안인지 확인
        err = (x_t.cos().mean(1) - self.phi).square() + x_t.sin().mean(1).square()
        return ((err @ self.weights) * V).mean()

class MultiResViTEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        Args:
            in_vars: Number of input channels/variables.
            model_name: timm ViT model name.
            proj_dim: Output projection head dimension.
            embed_dim: ViT embedding dimension.
        """
        super().__init__()
        # C -> 512 단일 투영 레이어 (Global/Local 공통)
        # 시계열 채널 축을 512차원 특징 공간으로 투영하여 512×512 "특징-시간 매니폴드" 생성
        self.proj = nn.Linear(in_vars, 512)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=1)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): self.backbone.patch_embed.strict_img_size = False
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool) -> torch.Tensor:
        """
        2D spatial crop from the 512x512 feature-temporal manifold.
        """
        B_v, C_ch, H, W = x.shape
        out = torch.empty((B_v, C_ch, target_res, target_res), device=x.device, dtype=x.dtype)
        if training:
            for i in range(B_v):
                h_start = random.randint(0, H - target_res)
                w_start = random.randint(0, W - target_res)
                out[i] = x[i, :, h_start:h_start + target_res, w_start:w_start + target_res]
        else:
            h_start = (H - target_res) // 2
            w_start = (W - target_res) // 2
            out = x[:, :, h_start:h_start + target_res, w_start:w_start + target_res]
        return out

    def _process(self, x: torch.Tensor, target_res: int, training: bool = True, is_downstream: bool = False) -> torch.Tensor:
        """
        Process a batch of views through projection, 2D crop, and ViT backbone.

        Args:
            x: Input tensor of shape [B_v, 1, C, T=512].
            target_res: Crop size (224 for global, 98 for local).
            training: Controls random vs center crop behavior.

        Returns:
            Embedding tensor of shape [B_v, embed_dim].
        """
        B_v, _, C, T = x.shape

        # 1. [B_v, 1, C, T] -> [B_v, T, C]
        x = x.squeeze(1).transpose(1, 2)

        # 2. Channel Projection: C -> 512 -> [B_v, T=512, 512]
        x = self.proj(x)  # [B_v, 512, 512]

        # 3. 512×512 매니폴드에서 2D Crop 또는 Resize: [B_v, 1, target_res, target_res]
        x = x.unsqueeze(1)  # [B_v, 1, 512, 512]
        if is_downstream:
            x = F.interpolate(x, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            x = self._2d_crop(x, target_res, training)

        # 4. ViT Patch Embedding + Position Encoding Resampling
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // self.patch_size, target_res // self.patch_size)
        x = self.backbone.patch_embed(x)
        if self.backbone.cls_token is not None:
            x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        x = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x + pos_embed)))

        # 5. Pooling: CLS 토큰 또는 평균
        return x[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else x[:, 0]

    def forward(self, views: dict) -> tuple:
        """
        Forward pass for multi-resolution views.

        Args:
            views: Dict with 'global' [B, 2, C, 512] and 'local' [B, 6, C, 512].

        Returns:
            Tuple of (all_emb [B, 8, embed_dim], proj [8, B, proj_dim]).
        """
        B = views['global'].shape[0]
        training = self.training

        # Global 2개: [B, 2, C, 512] -> flatten -> [B*2, 1, C, 512] -> _process
        g_emb = self._process(views['global'].flatten(0, 1).unsqueeze(1), 224, training)
        # Local 6개: [B, 6, C, 512] -> flatten -> [B*6, 1, C, 512] -> _process
        l_emb = self._process(views['local'].flatten(0, 1).unsqueeze(1), 98, training)

        all_emb = torch.cat([g_emb.reshape(B, 2, -1), l_emb.reshape(B, 6, -1)], dim=1)  # [B, 8, 384]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B, 8, -1).transpose(0, 1)   # [8, B, proj_dim]
        return all_emb, proj
