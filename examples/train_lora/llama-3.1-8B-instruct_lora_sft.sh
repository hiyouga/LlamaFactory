#!/bin/bash

set -x

OUTPUT="saves/llama-3.1-8B-instruct_lora_sft"
mkdir -p "$OUTPUT"

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

llamafactory-cli train \
    --model_name_or_path "$MODEL_PATH" \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target all \
    --dataset identity,alpaca_en_demo \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir "$OUTPUT" \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 18000 > "$OUTPUT/train.log" 2>&1
