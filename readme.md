# WORK IN PROGRESS

Requires diffusers 0.30.0 (this is specified in requirements.txt)

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
