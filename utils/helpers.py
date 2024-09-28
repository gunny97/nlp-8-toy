import pickle
import torch.nn as nn
import bitsandbytes as bnb


def find_linear_names(model, train_mode = 'lora'):
    """
    This function identifies all linear layer names within a model that use 4-bit quantization.
    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
    Returns:
        list: A list containing the names of all identified linear layers with 4-bit quantization.
    """
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear

    # Set to store identified layer names
    lora_module_names = set()

    # Iterate through named modules in the model
    for name, module in model.named_modules():
        # Check if the current module is an instance of the 4-bit linear layer class
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # Special case: remove 'lm_head' if present
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
    return list(lora_module_names)


def load_pickle(dir):
    with open(dir, "rb") as f:
        embeds = pickle.load(f)
    return embeds