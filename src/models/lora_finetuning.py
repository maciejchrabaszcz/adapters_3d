from typing import Optional

import torch

from peft import LoraConfig, get_peft_model


def get_conv_and_linear_layers(model: torch.nn.Module):
    target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    return target_modules


def get_layers_from_keys(modules: list, keys: list):
    target_modules = []
    for name in modules:
        if any(
            [key.replace(".bias", "").replace(".weight", "") == name for key in keys]
        ):
            target_modules.append(name)
    return target_modules


def get_model_with_lora(
    model: torch.nn.Module,
    r: int = 8,
    lora_alpha: int = 8,
    keys_not_loaded: Optional[list] = None,
):
    target_modules = get_conv_and_linear_layers(model)

    if keys_not_loaded is not None:
        modules_to_save = get_layers_from_keys(target_modules, keys_not_loaded)
        target_modules = list(set(target_modules) - set(modules_to_save))
    else:
        modules_to_save = None
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(
        model,
        config,
    )
    return peft_model
