# WORK IN PROGRESS

Spreadsheet (WIP) of supported models and their supported features: https://docs.google.com/spreadsheets/d/16eA6mSL8XkTcu9fSWkPSHfRIqyAKJbR1O99xnuGdCKY/edit?usp=sharing

## Update 9
Added preliminary support for [Go-with-the-Flow](https://github.com/VGenAI-Netflix-Eyeline-Research/Go-with-the-Flow)

This uses LoRA weights available here: https://huggingface.co/Eyeline-Research/Go-with-the-Flow/tree/main

To create the input videos for the NoiseWarp process, I've added a node to KJNodes that works alongside my SplineEditor, and either [comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes) or just cv2 inpainting to create the cut and drag input videos.

The workflows are in the example_workflows -folder.

Quick video to showcase: First mask the subject, then use the cut and drag -workflow to create a video as seen here, then that video is used as input to the NoiseWarp node in the main workflow.

https://github.com/user-attachments/assets/112706b0-a38b-4c3c-b779-deba0827af4f

## BREAKING Update8

This is big one, and unfortunately to do the necessary cleanup and refactoring this will break every old workflow as they are.
I apologize for the inconvenience, if I don't do this now I'll keep making it worse until maintaining becomes too much of a chore, so from my pov there was no choice.

*Please either use the new workflows or fix the nodes in your old ones before posting issue reports!*

Old version will be kept in a legacy branch, but not maintained

- Support CogVideoX 1.5 models
- Major code cleanup (it was bad, still isn't great, wip)
- Merge Fun -model functionality into main pipeline:
    - All Fun specific nodes, besides image encode node for Fun -InP models are gone
    - Main CogVideo Sampler works with Fun models
    - DimensionX LoRAs now work with Fun models as well

- Remove width/height from the sampler widgets and detect from input instead, this meanst text2vid now requires using empty latents
- Separate VAE from the model, allow using fp32 VAE
- Add ability to load some of the non-GGUF models as single files (only few available for now: https://huggingface.co/Kijai/CogVideoX-comfy)
- Add some torchao quantizations as options
- Add interpolation as option for the main encode node, old interpolation specific node is gone
- torch.compile optimizations
- Remove PAB in favor of FasterCache and cleaner code
- other smaller things I forgot about at this point

For Fun -model based workflows it's more drastic change, for others migrating generally means re-setting many of the nodes.

## Update7

- Refactored the Fun version's sampler to accept any resolution, this should make it lot simpler to use with Tora. **BREAKS OLD WORKFLOWS**, old FunSampler nodes need to be remade.
- The old bucket resizing is now on it's own node (CogVideoXFunResizeToClosestBucket) to keep the functionality, I honestly don't know if it matters at all, but just in case.
- Fun version's vid2vid is now also in the same node, the old vid2vid node is deprecated.
- Added support for FasterCache, this trades more VRAM use for speed with slight quality hit, similar to PAB: https://github.com/Vchitect/FasterCache
- Improved torch.compile support, it actually works now

## Update6

Initial support for Tora (https://github.com/alibaba/Tora)

Converted model (included in the autodownload node):

https://huggingface.co/Kijai/CogVideoX-5b-Tora/tree/main


https://github.com/user-attachments/assets/d5334237-03dc-48f5-8bec-3ae5998660c6


## Update5
This week there's been some bigger updates that will most likely affect some old workflows, sampler node especially probably need to be refreshed (re-created) if it errors out!

New features:
- Initial context windowing with FreeNoise noise shuffling mainly for vid2vid and pose2vid pipelines for longer generations, haven't figured it out for img2vid yet
- GGUF models and tiled encoding for I2V and pose pipelines (thanks to MinusZoneAI)
- [sageattention](https://github.com/thu-ml/SageAttention) support (Linux only) for a speed boost, I experienced ~20-30% increase with it, stacks with fp8 fast mode, doesn't need compiling
- Support CogVideoX-Fun 1.1 and it's pose models with additional control strength and application step settings, this model's input does NOT have to be just dwpose skeletons, just about anything can work
- Support LoRAs

https://github.com/user-attachments/assets/ddeb8f38-a647-42b3-a4b1-c6936f961deb

https://github.com/user-attachments/assets/c78b2832-9571-4941-8c97-fbcc1a4cc23d

https://github.com/user-attachments/assets/d9ed98b1-f917-432b-a16e-e01e87efb1f9



## Update4
Initial support for the official I2V version of CogVideoX: https://huggingface.co/THUDM/CogVideoX-5b-I2V

**Also needs diffusers 0.30.3**

https://github.com/user-attachments/assets/c672d0af-a676-495d-a42c-7e3dd802b4b0



## Update3

Added initial support for CogVideoX-Fun: https://github.com/aigc-apps/CogVideoX-Fun

Note that while this one can do image2vid, this is NOT the official I2V model yet, though it should also be released very soon.

https://github.com/user-attachments/assets/68f9ed16-ee53-4955-b931-1799461ac561


## Updade2

Added **experimental** support for onediff, this reduced sampling time by ~40% for me, reaching 4.23 s/it on 4090 with 49 frames. 
This requires using Linux, torch 2.4.0, onediff and nexfort installation:

`pip install --pre onediff onediffx`

`pip install nexfort`

First run will take around 5 mins for the compilation.

## Update
5b model is now also supported for basic text2vid: https://huggingface.co/THUDM/CogVideoX-5b

It is also autodownloaded to `ComfyUI/models/CogVideo/CogVideoX-5b`, text encoder is not needed as we use the ComfyUI T5.

https://github.com/user-attachments/assets/991205cc-826e-4f93-831a-c10441f0f2ce

Requires diffusers 0.30.1 (this is specified in requirements.txt)

Uses same T5 model than SD3 and Flux, fp8 works fine too. Memory requirements depend mostly on the video length. 
VAE decoding seems to be the only big that takes a lot of VRAM when everything is offloaded, peaks at around 13-14GB momentarily at that stage.
Sampling itself takes only maybe 5-6GB.


Hacked in img2img to attempt vid2vid workflow, works interestingly with some inputs, highly experimental.

https://github.com/user-attachments/assets/e6951ef4-ea7a-4752-94f6-cf24f2503d83

https://github.com/user-attachments/assets/9e41f37b-2bb3-411c-81fa-e91b80da2559

Also added temporal tiling as means of generating endless videos:

https://github.com/kijai/ComfyUI-CogVideoXWrapper

https://github.com/user-attachments/assets/ecdac8b8-d434-48b6-abd6-90755b6b552d



Original repo:
https://github.com/THUDM/CogVideo

CogVideoX-Fun:
https://github.com/aigc-apps/CogVideoX-Fun

Controlnet:
https://github.com/TheDenk/cogvideox-controlnet
