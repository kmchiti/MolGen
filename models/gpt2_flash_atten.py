from abc import ABC
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None


class GPT2MolGen_flash_atten(GPTLMHeadModel, ABC):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: Optional[int],
            vocab_size: Optional[int],
            bos_token_id: Optional[str],
            eos_token_id: Optional[str],
            use_flash_attn: bool = True,
            fused_bias_fc: bool = True,
            fused_mlp: bool = True,
            fused_dropout_add_ln: bool = True,
            residual_in_fp32: bool = True,
            pad_vocab_size_multiple: int = 8,
            use_rms_norm: bool = False,
            prenorm: bool = None,
            rotary_emb_fraction: float = 0.0,  # should be rotary_dim / headdim,
            rotary_emb_base: float = 10000.0,
            rotary_emb_scale_base: float = None,
            rotary_emb_interleaved: bool = True,

    ):
        config = GPT2Config(
            vocab_size=vocab_size,
            n_ctx=max_seq_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_flash_attn=use_flash_attn,
            fused_bias_fc=fused_bias_fc,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            use_rms_norm=use_rms_norm,
            prenorm=prenorm,
            rotary_emb_fraction=rotary_emb_fraction,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base=rotary_emb_scale_base,
            rotary_emb_interleaved=rotary_emb_interleaved,
        )

        GPT2LMHeadModel.__init__(self, config)

