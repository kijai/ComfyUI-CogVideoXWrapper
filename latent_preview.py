import io

import torch
from PIL import Image
import struct
import numpy as np
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import comfy.model_management
import folder_paths
import comfy.utils
import logging

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            ).to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("GIF", preview_image, MAX_PREVIEW_RESOLUTION)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self):
        latent_rgb_factors = [[0.11945946736445662, 0.09919175788574555, -0.004832707433877734], [-0.0011977028264356232, 0.05496505130267682, 0.021321622433638193], [-0.014088548986590666, -0.008701477861945644, -0.020991313281459367], [0.03063921972519621, 0.12186477097625073, 0.0139593690235148], [0.0927403067854673, 0.030293187650929136, 0.05083134241694003], [0.0379112441305742, 0.04935199882777209, 0.058562766246777774], [0.017749911959153715, 0.008839453404921545, 0.036005638019226294], [0.10610119248526109, 0.02339855688237826, 0.057154257614084596], [0.1273639464837117, -0.010959856130713416, 0.043268631260428896], [-0.01873510946881321, 0.08220930648486932, 0.10613256772247093], [0.008429116376722327, 0.07623856561000408, 0.09295712117576727], [0.12938137079617007, 0.12360403483892413, 0.04478930933220116], [0.04565908794779364, 0.041064156741596365, -0.017695041535528512], [0.00019003240570281826, -0.013965147883381978, 0.05329669529635849], [0.08082391586738358, 0.11548306825496074, -0.021464170006615893], [-0.01517932393230994, -0.0057985555313003236, 0.07216646476618871]]

        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        # if latent_rgb_factors_bias is not None:
        #     self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        latent_image = torch.nn.functional.linear(x0[0].permute(1, 2, 0), self.latent_rgb_factors,
                                                    bias=self.latent_rgb_factors_bias)
        return preview_to_image(latent_image)


def get_previewer():
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer method

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if previewer is None:
            previewer = Latent2RGBPreviewer()
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer()

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

