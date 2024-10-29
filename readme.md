# WORK IN PROGRESS
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
