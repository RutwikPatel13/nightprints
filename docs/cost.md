# Cost

Target: $0. Every piece below has a free tier that covers this project.

| Piece | Tool | Cost | Limit to know about |
|---|---|---|---|
| Base model | SDXL base 1.0 from Hugging Face | free | about 7 GB download, no account needed |
| Dataset generation | Kaggle notebook, T4 GPU | free | 30 GPU hours per week, 12 hours per session |
| LoRA training | Kaggle, same | free | same |
| Post-processing | rembg, spandrel, Pillow | free | runs on CPU, fine on a Mac |
| Upscaler weights | Real-ESRGAN x4plus from GitHub | free | |
| App | Gradio with a share link from Kaggle, or local on the Mac | free | share links expire after about 72 hours |
| Code hosting | GitHub | free | |

## Rough time budget on a free T4

These are estimates and will vary.

- Dataset: about 300 images at roughly 30 seconds each, so 2 to 3 hours.
- Training: 1500 steps, roughly 2 to 4 hours.
- Eval: 10 prompts x 4 seeds x 2 (with and without LoRA), under an hour.

That fits in one week's free Kaggle quota with room for a retrain.

## What could cost money later

- A permanent hosted app. Free options shut down or sleep, so for your friend the practical free setup is running the app on demand in Kaggle or on your Mac.
- If you ever want Flux quality, Flux dev's license blocks commercial use. Stick with SDXL for prints that get sold.
