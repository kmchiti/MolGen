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
            pad_token_id: Optional[str] = "[PAD]",
            activation_function: Optional[float] = "gelu_new",
    ):
        config = AutoConfig.from_pretrained(
                            model_name_or_path,
                            vocab_size=vocab_size,
                            n_ctx=max_seq_length,
                            bos_token_id=bos_token_id,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                            activation_function=activation_function,
                    )
        GPT2LMHeadModel.__init__(self, config)

    def save_HF_model(self, config, tokenizer, output_dir: str, dataset_name: str = None, token=None):

        self.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        upload_name = f'MolGen/GPT2-{dataset_name}'
        self.push_to_hub(upload_name, token=token)
        tokenizer.push_to_hub(upload_name, token=token)
