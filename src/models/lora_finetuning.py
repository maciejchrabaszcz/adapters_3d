import torch

from peft import LoraConfig, get_peft_model


def get_conv_and_linear_layers(model: torch.nn.Module):
    target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    return target_modules


def get_model_with_lora(model: torch.nn.Module, r: int = 8, lora_alpha: int = 8):
    target_modules = get_conv_and_linear_layers(model)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(
        model,
        config,
    )
    return peft_model
