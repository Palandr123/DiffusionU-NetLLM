import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.models.attention_processor import Attention
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers import StableDiffusionXLPipeline


class CustomAttentionProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, resolution: int = 64, save_aux: bool = True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.resolution = resolution
        self.save_aux = save_aux

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        ### extract cross-attention maps
        query_ = attn.head_to_batch_dim(query)
        key_ = attn.head_to_batch_dim(key)
        scores_ = attn.get_attention_scores(query_, key_, attention_mask)
        scores_ = scores_.reshape(len(query), -1, *scores_.shape[1:]).mean(1)
        h = w = math.isqrt(scores_.shape[1])
        scores_ = scores_.reshape(len(scores_), h, w, -1)
        if self.resolution != scores_.shape[2]:
            scores_ = TF.resize(
                scores_.permute(0, 3, 1, 2), self.resolution, antialias=True
            ).permute(0, 2, 3, 1)
        try:
            if not self.save_aux:
                len_ = len(attn._aux["attn"])
                del attn._aux["attn"]
                attn._aux["attn"] = [None] * len_ + [scores_.cpu()]
            else:
                attn._aux["attn"][-1] = attn._aux["attn"][-1].cpu()
                attn._aux["attn"].append(scores_.cpu())
        except:
            try:
                del attn._aux["attn"]
            except:
                pass
            attn._aux = {"attn": [scores_]}

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_attention(
    model: StableDiffusionXLPipeline, target_attention_list: list[str] | None = None
):
    for name, block in model.unet.named_modules():
        if isinstance(
            block, (CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn)
        ):
            for attn_name, attn in block.named_modules():
                full_name = name + "." + attn_name
                if isinstance(attn, Attention) and full_name in target_attention_list:
                    attn.processor = CustomAttentionProcessor()
