from abc import ABC
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig
from typing import Optional

class GPT2MolGen(GPT2LMHeadModel, ABC):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: Optional[int] = 64,
            vocab_size: Optional[int] = 30002,
            bos_token_id: Optional[str] = "<bos>",
            eos_token_id: Optional[str] = "<eos>",
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
        GPT2LMHeadModel.__init__(self, config)
