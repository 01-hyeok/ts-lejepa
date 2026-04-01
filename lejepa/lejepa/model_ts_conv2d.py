import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed
from torchvision.ops import MLP


class TSConv2DCanvas(nn.Module):
    """Learnable Time Series to 2D Image converter using Linear & Conv2D.

    1. Translates channels C to 512 via Linear projection (like 'basic').
    2. Applies learnable Conv2D layers on the resulting 512x512 canvas
       to extract local spatial/temporal patterns.

    Args:
        in_vars: Number of input channels.
        hidden_ch: Base number of intermediate channels for Conv2D.
    """

    def __init__(self, in_vars: int, hidden_ch: int = 32) -> None:
        super().__init__()
        # 1. 채널 방향 투영 (C -> 512)
        # [B, T=512, C] -> [B, 512, 512]가 되도록 투영
        self.proj = nn.Linear(in_vars, 512)
        
        # 2. 512x512 캔버스 상에서 로컬 패턴(채널-시간 관계) 학습
        # Conv2D를 1개만 사용하여 파라미터를 최소화하고 3x3 범위의 가장 인접한 시간/채널만 참조
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, 1, C, T]. T is typically 512.
        Returns:
            Canvas tensor of shape [B, 1, 512, 512].
        """
        B_v, _, C, T = x.shape
        
        # [B, 1, C, T] -> [B, T, C]
        x = x.squeeze(1).transpose(1, 2)
        
        # [B, T, C] -> [B, T, 512]
        canvas = self.proj(x)
        
        # [B, T, 512] -> [B, 1, 512, 512] (Conv2D를 위한 1채널화 및 축 변경)
        # T=512, Feature=512 이므로 height=512, width=512
        canvas = canvas.unsqueeze(1).transpose(2, 3) 
        
        # Conv2D 필터 통과 [B, 1, 512, 512] -> [B, 1, 512, 512]
        canvas = self.conv(canvas)
        
        return canvas


class Conv2DLearnableEncoder(nn.Module):
    """LeJEPA Encoder variant using 2D Conv-based TS-to-Image conversion.

    Converts time series [B, 1, C, T] → learned 2D canvas [B, 1, 512, 512]
    → 2D crop → ViT (in_chans=1) → [B, embed_dim].

    Unlike basic/tiling variants, the channel-to-spatial mapping is done
    entirely via learnable Conv2D layers without explicit channel projection.

    Args:
        in_vars: Number of input channels (stored for API consistency, not
            used in layer definitions — Conv2D is channel-agnostic).
        model_name: timm ViT model name.
        proj_dim: Output projection head dimension.
        embed_dim: ViT embedding dimension.
        hidden_ch: Intermediate channels in Conv2D canvas encoder.
    """

    def __init__(
        self,
        in_vars: int,
        model_name: str = "vit_small_patch14_dinov2",
        proj_dim: int = 128,
        embed_dim: int = 384,
        hidden_ch: int = 32,
    ) -> None:
        super().__init__()
        self.in_vars = in_vars

        # TS → 512×512 Canvas (Linear + Conv2D)
        self.canvas_encoder = TSConv2DCanvas(in_vars=in_vars, hidden_ch=hidden_ch)

        # ViT (in_chans=1: Conv2D 체인의 출력이 항상 1채널)
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=None, in_chans=1
        )
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, "strict_img_size"):
            self.backbone.patch_embed.strict_img_size = False

        self.proj_head = MLP(
            embed_dim,
            [embed_dim * 4, embed_dim * 4, proj_dim],
            norm_layer=nn.BatchNorm1d,
        )

    def _2d_crop(
        self, x: torch.Tensor, target_res: int, training: bool
    ) -> torch.Tensor:
        """2D spatial crop from the 512x512 canvas.

        Args:
            x: Input tensor of shape [B_v, 1, 512, 512].
            target_res: Crop size (224 for global, 98 for local).
            training: If True, random crop; else center crop.
        Returns:
            Cropped tensor of shape [B_v, 1, target_res, target_res].
        """
        _, _, H, W = x.shape
        if training:
            h_start = random.randint(0, H - target_res)
            w_start = random.randint(0, W - target_res)
        else:
            h_start = (H - target_res) // 2
            w_start = (W - target_res) // 2
        return x[:, :, h_start : h_start + target_res, w_start : w_start + target_res]

    def _process(
        self, x: torch.Tensor, target_res: int, training: bool = True, is_downstream: bool = False
    ) -> torch.Tensor:
        """Process a batch through Conv2D canvas encoder and ViT.

        Args:
            x: Input tensor of shape [B_v, 1, C, T].
            target_res: Crop resolution (224 for global, 98 for local).
            training: Controls random vs center crop behavior.
        Returns:
            Embedding tensor of shape [B_v, embed_dim].
        """
        # 1. Conv2D 기반 학습 가능 캔버스 생성: [B_v, 1, C, T] → [B_v, 1, 512, 512]
        canvas = self.canvas_encoder(x)

        # 2. 2D Crop or Resize: [B_v, 1, 512, 512] → [B_v, 1, target_res, target_res]
        if is_downstream:
            canvas = F.interpolate(canvas, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            canvas = self._2d_crop(canvas, target_res, training)

        # 3. ViT Patch Embedding + Position Encoding Resampling
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // self.patch_size, target_res // self.patch_size)

        x_vit = self.backbone.patch_embed(canvas)
        if self.backbone.cls_token is not None:
            x_vit = torch.cat(
                (self.backbone.cls_token.expand(x_vit.shape[0], -1, -1), x_vit), dim=1
            )

        pos_embed = resample_abs_pos_embed(
            orig_pos,
            new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0,
        )
        x_vit = self.backbone.norm(
            self.backbone.blocks(self.backbone.pos_drop(x_vit + pos_embed))
        )

        # 4. Pooling: global_pool='avg'이면 patch 평균, 아니면 CLS 토큰
        return x_vit[:, 1:].mean(1) if self.backbone.global_pool == "avg" else x_vit[:, 0]

    def forward(self, views: dict) -> tuple:
        """Forward pass for multi-resolution JEPA pretraining.

        Args:
            views: Dict with 'global' [B, 2, C, 512] and 'local' [B, 6, C, 512].
        Returns:
            Tuple of (all_emb [B, 8, embed_dim], proj [8, B, proj_dim]).
        """
        B = views["global"].shape[0]
        training = self.training

        # Global 2개: [B, 2, C, 512] → flatten → [B*2, 1, C, 512] → _process
        g_emb = self._process(views["global"].flatten(0, 1).unsqueeze(1), 224, training)
        # Local 6개: [B, 6, C, 512] → flatten → [B*6, 1, C, 512] → _process
        l_emb = self._process(views["local"].flatten(0, 1).unsqueeze(1), 98, training)

        all_emb = torch.cat(
            [g_emb.reshape(B, 2, -1), l_emb.reshape(B, 6, -1)], dim=1
        )  # [B, 8, embed_dim]
        proj = (
            self.proj_head(all_emb.flatten(0, 1))
            .reshape(B, 8, -1)
            .transpose(0, 1)
        )  # [8, B, proj_dim]
        return all_emb, proj
