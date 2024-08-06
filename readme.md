# WORK IN PROGRESS

Currently requires diffusers with PR: https://github.com/huggingface/diffusers/pull/9082

This is specified in requirements.txt

Uses same T5 model than SD3 and Flux, fp8 works fine too. Memory requirements depend mostly on the video length. 
VAE decoding seems to be the only big that takes a lot of VRAM when everything is offloaded, peaks at around 13-14GB momentarily at that stage.
Sampling itself takes only maybe 5-6GB.

Hacked in img2img to attempt vid2vid workflow, works interestingly with some inputs, highly experimental.

https://github.com/user-attachments/assets/e6951ef4-ea7a-4752-94f6-cf24f2503d83

https://github.com/user-attachments/assets/9e41f37b-2bb3-411c-81fa-e91b80da2559



Original repo:
https://github.com/THUDM/CogVideo
