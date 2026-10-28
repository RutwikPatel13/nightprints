#!/usr/bin/env bash
# Phase 2b: train an SDXL LoRA with the official diffusers example script.
# Sized for free 16 GB T4s: fp16, gradient checkpointing, 8-bit Adam, batch 1 per GPU.
# Uses every visible GPU. Kaggle's "T4 x2" gives two; each holds a full model copy
# and they average gradients (data parallel). Gradient accumulation is divided by
# the GPU count so the effective batch, and so the training recipe, stays the same.
#   NUM_GPUS=1 bash scripts/train.sh   # force one GPU if multi-GPU hangs
#   RES=768 bash scripts/train.sh      # if you hit out-of-memory
set -euo pipefail

STYLE="${STYLE:-configs/styles/dino.yaml}"
RES="${RES:-1024}"
DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
NUM_GPUS="${NUM_GPUS:-$DETECTED}"
[ "$NUM_GPUS" -ge 1 ] || NUM_GPUS=1

read_cfg() { python -c "import yaml,sys; c=yaml.safe_load(open('$STYLE')); print(eval('c'+sys.argv[1]))" "$1"; }

BASE=$(read_cfg "['model']['base']")
VAE=$(read_cfg "['model']['vae']")
TRAIN_DIR=$(read_cfg "['paths']['train']")
OUT_DIR=$(read_cfg "['paths']['lora']")
TRIGGER=$(read_cfg "['trigger']")
STEPS=$(read_cfg "['train']['steps']")
LR=$(read_cfg "['train']['lr']")
RANK=$(read_cfg "['train']['rank']")
ACCUM=$(read_cfg "['train']['grad_accum']")
SEED=$(read_cfg "['train']['seed']")

ACCUM=$(( ACCUM / NUM_GPUS ))
[ "$ACCUM" -ge 1 ] || ACCUM=1
MULTI=""
[ "$NUM_GPUS" -gt 1 ] && MULTI="--multi_gpu"
echo "training on $NUM_GPUS GPU(s), grad accumulation $ACCUM"

# Use the training script that matches the installed diffusers version.
DV=$(python -c "import diffusers; print(diffusers.__version__)")
SCRIPT="scripts/.train_text_to_image_lora_sdxl.py"
if [ ! -f "$SCRIPT" ]; then
  TAG="v${DV%%.dev*}"
  curl -fsSL "https://raw.githubusercontent.com/huggingface/diffusers/${TAG}/examples/text_to_image/train_text_to_image_lora_sdxl.py" -o "$SCRIPT" \
    || curl -fsSL "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py" -o "$SCRIPT"
fi

mkdir -p "$OUT_DIR"
accelerate launch $MULTI --mixed_precision=fp16 --num_processes="$NUM_GPUS" --num_machines=1 --dynamo_backend=no "$SCRIPT" \
  --pretrained_model_name_or_path="$BASE" \
  --pretrained_vae_model_name_or_path="$VAE" \
  --train_data_dir="$TRAIN_DIR" \
  --caption_column="text" \
  --resolution="$RES" \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps="$ACCUM" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --mixed_precision=fp16 \
  --learning_rate="$LR" \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --rank="$RANK" \
  --max_train_steps="$STEPS" \
  --checkpointing_steps=500 \
  --seed="$SEED" \
  --validation_prompt="$TRIGGER, character, baby t-rex sleeping under a blanket" \
  --validation_epochs=50 \
  --output_dir="$OUT_DIR" \
  --report_to=tensorboard

echo "LoRA saved to $OUT_DIR/pytorch_lora_weights.safetensors"
