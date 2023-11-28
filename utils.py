from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
import copy
import json
import hashlib


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
    model_arch = _config.model.model_name_or_path
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    exp_name = f"{model_arch}_{hash_name}"
    return exp_name
