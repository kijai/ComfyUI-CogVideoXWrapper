import importlib.metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.30.3'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")
    
def remove_specific_blocks(model, block_indices_to_remove):
    import torch.nn as nn
    transformer_blocks = model.transformer_blocks
    new_blocks = [block for i, block in enumerate(transformer_blocks) if i not in block_indices_to_remove]
    model.transformer_blocks = nn.ModuleList(new_blocks)
    
    return model