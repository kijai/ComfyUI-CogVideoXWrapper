# WORK IN PROGRESS

## Updade2

Added **experimental** support for onediff, this reduced sampling time by ~30% for me, reaching 4.23 it/s on 4090 with 49 frames. 
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
