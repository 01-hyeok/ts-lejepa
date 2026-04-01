import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from timm.layers import resample_abs_pos_embed

class MultiResViTCIEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        LeJEPA CI (1D Sequence & Patch Level Variant)
        """
        super().__init__()
        self.in_vars = in_vars
        self.embed_dim = embed_dim
        
        # ViT 입력 채널은 1채널 (Channel Independent)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=1)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): 
            self.backbone.patch_embed.strict_img_size = False
            
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def get_1d_sincos_pos_embed(self, embed_dim: int, length: int, cls_token: bool = False) -> torch.Tensor:
        """1차원 Sin/Cos Positional Embedding을 생성합니다."""
        grid = torch.arange(length, dtype=torch.float32)
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega * 2.0 / (embed_dim // 2))) # [D/2]
        
        # [length, embed_dim//2]
        out = torch.einsum('m,d->md', grid, omega)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1) # [length, embed_dim]
        
        if cls_token:
            emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0) # [length+1, embed_dim]
        return emb.unsqueeze(0) # [1, N, embed_dim]

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B_v, 1, C, T] -> 변동하는 길이 T의 시퀀스 (Global: 512, Local: 128 등)
        """
        # 0. Flatten Channels into Batch Dimension: [B_v, 1, C, T] -> [B_v * C, 1, 1, T]
        if x.dim() == 4:
            B_v, _, C, T = x.shape
            x = x.transpose(1, 2).reshape(B_v * C, 1, 1, T)
        else:
            raise ValueError("Expected 4D input")
            
        # 1. 패치 분할 시 떨어지는 잔여 길이(Padding) 보정
        # T (예: 512, 128) 길이가 patch_size (예: 14)로 나누어 떨어지지 않는 경우 우측 패딩
        # 이 처리가 없으면 Conv2D에서 Valid Padding으로 소실됨
        remainder = T % self.patch_size
        if remainder > 0:
            pad_len = self.patch_size - remainder
            # [B_v*C, 1, 1, T] 우측 패딩 추가 -> [B_v*C, 1, 1, T + pad_len]
            x = F.pad(x, (0, pad_len))
            T = T + pad_len
        
        # 2. 기존 2D 패치 임베딩 모듈(Conv2d)을 활용하기 위해 세로 크기를 patch_size 만큼 Tiling
        # [B_v*C, 1, 1, T] -> [B_v*C, 1, patch_size, T]
        x_img = x.expand(-1, -1, self.patch_size, -1)

        # 3. 1D Patch Embedding
        # patch_embed를 통과하면 세로(height) 차원은 1이 되고 가로(width) 차원은 T // patch_size 가 됨.
        x_patches = self.backbone.patch_embed(x_img) # [B_v*C, N=1*(T//P), D]
        
        if self.backbone.cls_token is not None:
            x_patches = torch.cat((self.backbone.cls_token.expand(x_patches.shape[0], -1, -1), x_patches), dim=1)
            
        # 4. 순수 1D Positional Embedding 생성 및 주입
        N = T // self.patch_size
        pos_embed = self.get_1d_sincos_pos_embed(
            embed_dim=self.embed_dim, 
            length=N, 
            cls_token=(self.backbone.cls_token is not None)
        ).to(x.device)
        
        # 5. Transformer 블록 통과
        feat = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x_patches + pos_embed)))
        
        # 6. Global Average Pooling or CLS Token 추출
        emb = feat[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else feat[:, 0] # [B_v*C, 384]
        
        return emb

    def forward(self, views: dict) -> tuple:
        B = views['global'].shape[0]
        C = views['global'].shape[2]

        # [B, V, C, T] -> [B * V, 1, C, T]
        g_in = views['global'].flatten(0, 1).unsqueeze(1) # [B*2, 1, C, 512]
        l_in = views['local'].flatten(0, 1).unsqueeze(1)  # [B*6, 1, C, 128]

        g_emb = self._process(g_in) # [B*2*C, 384]
        l_emb = self._process(l_in) # [B*6*C, 384]

        # SIGReg 연산을 위해 (B*C)를 하나의 샘플(Batch)로 취급하고 V개를 합침
        g_emb = g_emb.reshape(B * 2, C, -1).transpose(0, 1).reshape(B * C, 2, -1)
        l_emb = l_emb.reshape(B * 6, C, -1).transpose(0, 1).reshape(B * C, 6, -1)

        all_emb = torch.cat([g_emb, l_emb], dim=1) # [B*C, 8, 384]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B * C, 8, -1).transpose(0, 1) # [8, B*C, proj_dim]
        
        return all_emb, proj
