# https://github.com/MinusZoneAI/ComfyUI-CogVideoX-MZ/blob/9616415220fd09388622f40f6609e4ed81f048a5/mz_gguf_loader.py

import torch
import torch.nn as nn
import gc


class quantize_lazy_load():
    def __init__(self):
        self.device = None

    def __enter__(self):
        self.device = torch.device("meta")
        self.device.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.device.__exit__(exc_type, exc_value, traceback)


def quantize_load_state_dict(model, state_dict, device="cpu"):
    quant_keys = []
    for key in state_dict.keys():
        if key.endswith(".Q4_0_qweight"):
            quant_keys.append(key.replace(".Q4_0_qweight", ""))
            qtype = "Q4_0"
        elif key.endswith(".Q8_0_qweight"):
            quant_keys.append(key.replace(".Q8_0_qweight", ""))
            qtype = "Q8_0"

    for name, module in model.named_modules():
        if name in quant_keys:
            q_linear = WQLinear_GGUF.from_linear(
                linear=module,
                device=device,
                qtype=qtype,
            )
            set_op_by_name(model, name, q_linear)

    model.to_empty(device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


import torch.nn.functional as F


class WQLinear_GGUF(nn.Module):
    def __init__(
        self, in_features, out_features, bias, dev, qtype="Q4_0"
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.qtype = qtype

        qweight_shape = quant_shape_to_byte_shape(
            (out_features, in_features), qtype
        )
        self.register_buffer(
            f"{qtype}_qweight",
            torch.zeros(
                qweight_shape,
                dtype=torch.uint8,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear,
        device="cpu",
        qtype="Q4_0",
    ):
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device,
            qtype=qtype,
        )
        return q_linear

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )

    @torch.no_grad()
    def forward(self, x):
        if self.qtype == "Q4_0":
            dequant = dequantize_blocks_Q4_0(self.Q4_0_qweight, x.dtype)
        elif self.qtype == "Q8_0":
            dequant = dequantize_blocks_Q8_0(self.Q8_0_qweight, x.dtype)
        else:
            raise ValueError(f"Unknown qtype: {self.qtype}")
        
        return F.linear(x, dequant, bias=self.bias.to(x.dtype) if self.bias is not None else None)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def quant_shape_to_byte_shape(shape, qtype) -> tuple[int, ...]:
    # shape = shape[::-1]
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of Q4_0 block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape, qtype) -> tuple[int, ...]:
    # shape = shape[::-1]
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of Q4_0 type size ({type_size})")
    return (*shape[:-1], shape[-1] // type_size * block_size)


GGML_QUANT_SIZES = {
    "Q4_0": (32, 2 + 16),
    "Q8_0": (32, 2 + 32),
}


def dequantize_blocks_Q4_0(data, dtype=torch.float16):
    block_size, type_size = GGML_QUANT_SIZES["Q4_0"]

    data = data.to(torch.uint8)
    shape = data.shape

    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = data.reshape((n_blocks, type_size))

    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8

    out = (d * qs)

    out = out.reshape(quant_shape_from_byte_shape(
        shape,
        qtype="Q4_0",
    )).to(dtype)
    return out

def dequantize_blocks_Q8_0(data, dtype=torch.float16):
    block_size, type_size = GGML_QUANT_SIZES["Q8_0"]

    data = data.to(torch.uint8)
    shape = data.shape

    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = data.reshape((n_blocks, type_size))

    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(torch.float32)

    qs = qs.view(torch.int8).to(torch.float32)

    out = (d * qs)

    out = out.reshape(quant_shape_from_byte_shape(
        shape,
        qtype="Q8_0",
    )).to(dtype)
    return out

