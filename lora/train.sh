#!/usr/bin/env bash
# P3 步驟 3／4：用 kohya sd-scripts（~/creative/sd-scripts，commit 11d8ced）訓練 LUNA 的 Illustrious LoRA。
# 用法：bash lora/train.sh v1     （資料集先用 tools/p3_prepare_dataset.py 產生到 lora/<version>/dataset/）
set -euo pipefail
VER=${1:-v1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SD=~/creative/sd-scripts
OUT=$ROOT/lora/$VER
cat > "$OUT/dataset.toml" <<TOML
[general]
enable_bucket = true
bucket_no_upscale = true
min_bucket_reso = 640
max_bucket_reso = 1536
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = 2

  [[datasets.subsets]]
  image_dir = "$(ls -d "$OUT"/dataset/*_lunacsf)"
  num_repeats = $(basename "$(ls -d "$OUT"/dataset/*_lunacsf)" | cut -d_ -f1)
TOML
cd "$SD"
.venv/bin/accelerate launch --num_cpu_threads_per_process 4 --mixed_precision bf16 sdxl_train_network.py \
  --pretrained_model_name_or_path ~/creative/ComfyUI/models/checkpoints/Illustrious-XL-v1.0.safetensors \
  --dataset_config "$OUT/dataset.toml" \
  --output_dir "$OUT/output" --output_name "luna_illustrious_$VER" --save_model_as safetensors \
  --network_module networks.lora --network_dim 32 --network_alpha 16 --network_train_unet_only \
  --optimizer_type AdamW8bit --learning_rate 1e-4 --lr_scheduler cosine --lr_warmup_steps 100 \
  --max_train_epochs 10 --save_every_n_epochs 2 \
  --mixed_precision bf16 --save_precision fp16 --gradient_checkpointing --sdpa \
  --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs \
  --seed 20261024 --logging_dir "$OUT/logs" \
  --sample_every_n_epochs 2 --sample_prompts "$ROOT/lora/sample_prompts.txt" --sample_sampler euler_a
