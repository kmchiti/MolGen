from abc import ABC
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig
from typing import Optional

class GPT2MolGen(GPT2LMHeadModel, ABC):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: Optional[int],
            vocab_size: Optional[int],
            bos_token_id: Optional[str],
            eos_token_id: Optional[str],
            use_flash_attention_2: Optional[bool] = False,
    ):
        config = AutoConfig.from_pretrained(
                            model_name_or_path,
                            vocab_size=vocab_size,
                            n_ctx=max_seq_length,
                            bos_token_id=bos_token_id,
                            eos_token_id=eos_token_id,
                            use_flash_attention_2=use_flash_attention_2,
                    )
        # TODO: add flash_atten
        GPT2LMHeadModel.__init__(self, config)
