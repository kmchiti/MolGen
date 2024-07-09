import os.path
from abc import ABC
import copy
from transformers import GPT2Config
from typing import Optional
from collections import OrderedDict, namedtuple
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaConfig
from flash_attn.models.llama import llama_config_to_gpt2_config, inv_remap_state_dict_hf_llama
from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.utils.distributed import all_gather_raw
except ImportError:
    all_gather_raw = None

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class Llama_small_flash_atten(GPTLMHeadModel, ABC):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: Optional[int],
            vocab_size: Optional[int],
            hidden_size: int,
            num_hidden_layers: int,
            num_attention_heads: int,
            intermediate_size: int,
            bos_token_id: Optional[str],
            eos_token_id: Optional[str],
            use_flash_attn: bool = True,
            fused_bias_fc: bool = True,
            fused_mlp: bool = False,
            fused_dropout_add_ln: bool = True,
            residual_in_fp32: bool = True,
    ):
        config = LlamaConfig(n_ctx=max_seq_length,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads,
                             intermediate_size=intermediate_size,
                             vocab_size=vocab_size,
                             bos_token_id=bos_token_id,
                             eos_token_id=eos_token_id)
        config = llama_config_to_gpt2_config(config)
        config.use_flash_attn = use_flash_attn
        config.fused_bias_fc = fused_bias_fc
        config.fused_mlp = fused_mlp
        config.fused_dropout_add_ln = fused_dropout_add_ln
        config.residual_in_fp32 = residual_in_fp32
        # config.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.config_copy = copy.deepcopy(config)
        self.config_copy.max_seq_length = max_seq_length
        self.config_copy.model_name_or_path = model_name_or_path
        self.config_copy.intermediate_size = intermediate_size

        GPTLMHeadModel.__init__(self, config)

    # TODO: here we ignore attention_mask to make it compatible with HF trainer. The MHA in flash-attention should
    #  be reimplement and integrate attention_mask like here:
    #  https://github.com/huggingface/transformers/blob/0864dd3beb238b7bec3528a3d1d6c17a28f51a51/src/transformers/models/llama/modeling_llama.py#L536
    def forward(self, input_ids, attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None,
                position_ids=None, inference_params=None, num_last_tokens=0):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        assert (
                input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, "(n b) ... d -> b ... (n d)", b=b)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=lm_logits)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
        )

    def save_pretrained(self, save_directory):
        """Save the model and tokenizer"""
        self.config_copy.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """Load the model and tokenizer"""
        config = GPT2Config.from_pretrained(load_directory)
        model = cls(model_name_or_path=config.model_name_or_path,
                    max_seq_length=config.max_seq_length,
                    vocab_size=config.vocab_size,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    bos_token_id=config.bos_token_id,
                    eos_token_id=config.eos_token_id,
                    use_flash_attn=config.use_flash_attn,
                    fused_bias_fc=config.fused_bias_fc,
                    fused_mlp=config.fused_mlp,
                    fused_dropout_add_ln=config.fused_dropout_add_ln,
                    residual_in_fp32=config.residual_in_fp32)

        if os.path.exists(load_directory):
            state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"))
        else:
            state_dict = state_dict_from_pretrained(load_directory)
        model.load_state_dict(state_dict)
        return model

    def save_HF_model(self, tokenizer, output_dir: str, dataset_name: str = None,
                      token=None, ckpt=None):
        from huggingface_hub import HfApi

        save_directory = os.path.join(output_dir, 'HF')
        self.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        # Push to Hugging Face Hub
        api = HfApi()
        if ckpt is None:
            upload_name = f'MolGen/Llama-small-{dataset_name}'
        else:
            upload_name = f'MolGen/Llama-small-{dataset_name}_{ckpt}'
        if not api.repo_exists('MolGen/llama_small_FA_ZINC_270M'):
            api.create_repo(repo_id=upload_name, token=token)
        api.upload_folder(
            folder_path=save_directory,
            path_in_repo=".",
            repo_id=upload_name,
            token=token
        )

