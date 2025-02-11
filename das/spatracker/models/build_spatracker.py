# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .core.spatracker.spatracker import SpaTracker


def build_spatracker(
    checkpoint: str,
    seq_length: int = 8,
):
    model_name = checkpoint.split("/")[-1].split(".")[0]
    return build_spatracker_from_cfg(checkpoint=checkpoint, seq_length=seq_length)



# model used to produce the results in the paper
def build_spatracker_from_cfg(checkpoint=None, seq_length=8):
    return _build_spatracker(
        stride=4,
        sequence_len=seq_length,
        checkpoint=checkpoint,
    )


def _build_spatracker(
    stride,
    sequence_len,
    checkpoint=None,
):
    spatracker = SpaTracker(
        stride=stride,
        S=sequence_len,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            if "safetensors" in checkpoint:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint)
            else:
                state_dict = torch.load(f, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                model_paras = spatracker.state_dict()
                paras_dict = {k: v for k,v in state_dict["model"].items() if k in spatracker.state_dict()}
                model_paras.update(paras_dict)
                state_dict = model_paras
        spatracker.load_state_dict(state_dict)
    return spatracker
