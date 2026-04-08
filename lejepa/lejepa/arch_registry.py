from __future__ import annotations

import torch.nn as nn

from .model_ts_conv import MultiResViTConvEncoder
from .model_ts_conv2d import Conv2DLearnableEncoder
from .model_ts_lejepa_1d import PatchTS1DEncoder
from .model_ts_lejepa_basic import MultiResViTEncoder
from .model_ts_lejepa_ci import MultiResViTCIEncoder
from .model_ts_lejepa_tiling import MultiResViTTilingEncoder
from .model_ts_timesblock import LeJEPATimesModel
from .model_ts_timevlm import TimeVLMEncoder
from .model_ts_tivit import TiViTDependentEncoder, TiViTIndependentEncoder
from .model_ts_utica import UTICAEncoder

SUPPORTED_ARCHS = (
    "basic",
    "tiling",
    "tiling_repeat",
    "tivit_indep",
    "tivit_dep",
    "timevlm",
    "conv2d",
    "tiling_ci",
    "conv",
    "patchtst",
    "timesnet",
    "utica",
)

ADAPTER_FREE_ARCHS = frozenset({
    "timevlm",
    "tiling_ci",
    "patchtst",
    "utica",
    "conv",
})

CHANNEL_AGGREGATING_ARCHS = frozenset({
    "tiling_ci",
    "patchtst",
    "utica",
    "conv",
})

CHECKPOINT_CHANNEL_KEYS = {
    "basic": "proj.weight",
    "tiling": "backbone.patch_embed.proj.weight",
    "tiling_repeat": "backbone.patch_embed.proj.weight",
    "tivit_dep": "backbone.patch_embed.proj.weight",
    "conv2d": "canvas_encoder.proj.weight",
    "timesnet": "encoder.input_proj.weight",
}


def normalize_arch(arch: str) -> str:
    return arch.lower().replace("-", "_")


def validate_arch(arch: str) -> str:
    arch_key = normalize_arch(arch)
    if arch_key not in SUPPORTED_ARCHS:
        supported = ", ".join(SUPPORTED_ARCHS)
        raise ValueError(f"Unknown architecture: {arch}. Supported: {supported}")
    return arch_key


def infer_pretrain_in_vars(
    arch: str,
    state_dict: dict | None,
    in_vars: int,
    pretrain_dataset: str,
) -> int:
    arch_key = validate_arch(arch)

    if arch_key in ADAPTER_FREE_ARCHS:
        return in_vars if arch_key == "timevlm" else 1

    if state_dict is not None:
        key = CHECKPOINT_CHANNEL_KEYS.get(arch_key)
        if key is not None and key in state_dict:
            return state_dict[key].shape[1]

    return 321 if pretrain_dataset == "electricity" else in_vars


def needs_channel_adapter(arch: str, pretrain_in_vars: int, in_vars: int) -> bool:
    arch_key = validate_arch(arch)
    return arch_key not in ADAPTER_FREE_ARCHS and pretrain_in_vars != in_vars


def uses_ci_decoder(arch: str) -> bool:
    return validate_arch(arch) in CHANNEL_AGGREGATING_ARCHS


def build_encoder(
    arch: str,
    in_vars: int,
    vit_model: str = "vit_small_patch14_dinov2",
    proj_dim: int = 128,
    patch_size: int = 16,
    use_revin: bool = False,
) -> nn.Module:
    arch_key = validate_arch(arch)

    if arch_key == "basic":
        return MultiResViTEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key in {"tiling", "tiling_repeat"}:
        return MultiResViTTilingEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "tivit_indep":
        return TiViTIndependentEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "tivit_dep":
        return TiViTDependentEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "timevlm":
        return TimeVLMEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "conv2d":
        return Conv2DLearnableEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "tiling_ci":
        return MultiResViTCIEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "conv":
        return MultiResViTConvEncoder(in_vars=in_vars, model_name=vit_model, proj_dim=proj_dim)
    if arch_key == "patchtst":
        return PatchTS1DEncoder(
            in_vars=in_vars,
            proj_dim=proj_dim,
            patch_size=patch_size,
            use_revin=use_revin,
        )
    if arch_key == "timesnet":
        return LeJEPATimesModel(
            in_channels=in_vars,
            proj_dim=proj_dim,
        )
    if arch_key == "utica":
        return UTICAEncoder(
            in_vars=in_vars,
            proj_dim=proj_dim,
            patch_size=patch_size,
            use_revin=use_revin,
        )

    raise ValueError(f"Unknown architecture: {arch}")


def get_embed_dim(encoder: nn.Module, arch: str) -> int:
    arch_key = validate_arch(arch)
    if arch_key == "utica":
        return encoder.encoder.d_model
    if arch_key == "patchtst":
        return encoder.d_model
    if arch_key == "timesnet":
        return encoder.encoder.input_proj.out_features
    return encoder.backbone.num_features
