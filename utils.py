from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict, namedtuple
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig
import copy
import json
import hashlib
import math
import re
try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    GPTLMHeadModel = None

def unroll_configs(cfg: Dict[str, Any], parent_key='', sep='_') -> Dict[str, Any]:
    """
    Recursively unroll a nested dictionary of configurations and remove keys with None values.

    Args:
        cfg (Dict[str, Any]): The input dictionary containing configuration options.
        parent_key (str): The parent key for the current level of recursion.
        sep (str): The separator used to separate parent and child keys.

    Returns:
        Dict[str, Any]: The output unrolled dictionary.
    """
    items = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(unroll_configs(value, new_key, sep=sep))
        elif value is not None:  # Exclude keys with None values
            items[new_key] = value
    return items


def creat_unique_experiment_name(config: DictConfig) -> str:
    """
    Generate a unique experiment name based on the provided configurations.

    Args:
        config (Dict[str, Any]): The input dictionary containing experiment configurations.

    Returns:
        str: A unique experiment name.
    """
    _config = OmegaConf.to_container(copy.deepcopy(config))
    model_arch = _config['model']['model_name_or_path']
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    exp_name = f"{model_arch}_{hash_name}"
    return exp_name

# code from: https://github.com/huggingface/transformers/blob/bd50402b56980ff17e957342ef69bd9b0dd45a7b/src/transformers/trainer.py#L2758
def is_world_process_zero(train_args) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on several
    machines, this is only going to be `True` for one process).
    """
    # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
    # process index.
    from transformers.utils.import_utils import is_sagemaker_mp_enabled
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        return smp.rank() == 0
    else:
        return train_args.process_index == 0

# inverse mode of 'remap_state_dict_hf_gpt2' function
# from: https://github.com/Dao-AILab/flash-attention/blob/92dd5703ecdb99aa4a4aee9817f28557907403a2/flash_attn/models/gpt.py#L930
def inv_remap_state_dict_gpt2_FA_hf(state_dict, config=None):
    # Embedding
    def key_mapping_pos_emb(key):
        return re.sub("transformer.embeddings.position_embeddings.", "transformer.wpe.", key)

    def key_mapping_word_emb(key):
        return re.sub("transformer.embeddings.word_embeddings.", "transformer.wte.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    state_dict = OrderedDict((key_mapping_word_emb(k), v) for k, v in state_dict.items())

    # TODO: don't know what to do with padding??
    # word_embeddings = state_dict.pop("model.embed_tokens.weight")
    # pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    # vocab_size = (
    #     math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    # )
    # state_dict["model.embed_tokens.weight"] = F.pad(
    #     word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    # )

    # LM head
    # TODO: same issue with padding embedding layers

    # LayerNorm
    def key_mapping_ln(key):
        pattern = re.compile(r"transformer\.layers\.(\d+)\.norm([12])\.(weight|bias)")
        return pattern.sub(lambda m: f"transformer.h.{m.group(1)}.ln_{m.group(2)}.{m.group(3)}", key)

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key, value):
        key = re.sub(r"transformer.layers.(\d+).mlp.fc1.bias", r"transformer.h.\1.mlp.c_fc.bias", key)
        key = re.sub(r"transformer.layers.(\d+).mlp.fc2.bias", r"transformer.h.\1.mlp.c_proj.bias", key)

        if re.match(r'transformer\.layers\.(\d+)\.mlp\.fc1\.weight', key):
            value = value.t()
            key = re.sub(r"transformer.layers.(\d+).mlp.fc1.weight", r"transformer.h.\1.mlp.c_fc.weight", key)

        if re.match(r'transformer\.layers\.(\d+)\.mlp\.fc2\.weight', key):
            value = value.t()
            key = re.sub(r"transformer.layers.(\d+).mlp.fc2.weight", r"transformer.h.\1.mlp.c_proj.weight", key)

        return key, value

    state_dict = OrderedDict((key_mapping_mlp(k, v)) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key, value):
        key = re.sub(r"transformer.layers.(\d+).mixer.Wqkv.(weight|bias)", r"transformer.h.\1.attn.c_attn.\2", key)
        key = re.sub(r"transformer.layers.(\d+).mixer.out_proj.(weight|bias)", r"transformer.h.\1.attn.c_proj.\2", key)

        if re.match(r'transformer\.h\.(\d+)\.attn\.c_attn\.weight', key) or re.match(
                r'transformer\.h\.(\d+)\.attn\.c_proj\.weight', key):
            value = value.t()
        return key, value

    state_dict = OrderedDict((key_mapping_attn(k, v)) for k, v in state_dict.items())

    return state_dict


def save_HF_model(model, config: GPT2Config, output_dir: str):
    assert isinstance(model, GPTLMHeadModel)

    # update the config
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    config = AutoConfig.from_pretrained(
        'gpt2',
        vocab_size=(math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple),
        n_ctx=config.n_ctx,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
    )

    updated_state_dict = inv_remap_state_dict_gpt2_FA_hf(model.state_dict())
    new_model = GPT2LMHeadModel(config)
    network_kvpair = new_model.state_dict()
    for key in updated_state_dict.keys():
        network_kvpair[key] = updated_state_dict[key]
    new_model.load_state_dict(network_kvpair)
    new_model.save_pretrained(output_dir)

