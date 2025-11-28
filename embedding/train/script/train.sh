#!/bin/bash
set -e

eval "$(~/miniconda/bin/conda shell.bash hook)"
conda activate train_embedding

MASTER_PORT=30000
DEVICES="0,1,2,3,4,5,6,7"

# Train
num_epochs=5
learning_rate="1e-4"  # 5e-5, 1e-5
weight_decay=0.1
num_warmup_steps=100
lr_scheduler_type=linear


max_length=4096
train_batch_size=32768
per_device_train_batch_size=128
num_devices="${DEVICES//,/}"
num_devices="${#num_devices}"
echo "num_devices: $num_devices"
gradient_accumulation_steps=$((train_batch_size / per_device_train_batch_size / num_devices))

echo "GRADIENT_ACCUMULATION_STEPS: $gradient_accumulation_steps"

dataset_name="msmarco_qwen3_embedding_8b_hn3"
dataset_config=config/dataset_config/${dataset_name}.yaml
model_name=Qwen3-Embedding-0.6B
model_name_or_path=./checkpoints/$model_name

instruction_type="general"
precision=fp16
matryoshka_dims=""  # e.g. "4096,2048,1024,512" to enable Matryoshka loss
pooling_strategy="last"

output_model_name=${model_name}-${dataset_name}-maxlen${max_length}-${precision}-lr${learning_rate}-warmup${num_warmup_steps}-decay${weight_decay}-bs${train_batch_size}-perdevice${per_device_train_batch_size}-${instruction_type}
output_dir=./checkpoints/$output_model_name


# if logs directory does not exist, create it
if [ ! -d "logs" ]; then
    mkdir logs
fi

# date in the format of YYYYMMDD
date=$(date "+%Y%m%d")
log=logs/${date}_train_$output_model_name.log
save_steps=5
save_total_limit=100

matryoshka_args=()
if [ -n "$matryoshka_dims" ]; then
    matryoshka_args=(--matryoshka_dims "$matryoshka_dims")
fi

WANDB_DISABLED="true" CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch \
    --main_process_port $MASTER_PORT\
    --config_file config/deepspeed_config/deepspeed_zero3.yaml \
    src/train.py \
    --dataset_config $dataset_config \
    --max_length $max_length \
    --model_name_or_path $model_name_or_path \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --precision $precision \
    --instruction_type $instruction_type \
    --num_train_epochs $num_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --num_warmup_steps $num_warmup_steps \
    --save_steps $save_steps \
    --output_dir $output_dir \
    --pooling_strategy $pooling_strategy \
    --group_by_length \
    --logging_steps 5 \
    "${matryoshka_args[@]}" > $log 2>&1
    # --continual_learning \ # when continuing learning from a checkpoint
    # --resume_from_checkpoint \


# Copy config files to output checkpoint directory
max=0
dir=""
# Find the latest checkpoint directory
for file in $output_dir/checkpoint-*; do
    num=${file##*-}  # Extract number from directory name
    echo "file: $file"
    echo "num: $num"
    if ((num > max)); then  # Update if current number is greater than max
        max=$num
        dir=$file
    fi
done

if [ -d "${dir}" ]; then
    echo "Processing ${dir}..."
    # File list to copy
    file_list=("configuration_RW.py" "modelling_RW.py" "tokenizer_config.json" "tokenizer.json" "special_tokens_map.json" "tokenizer.model" "config.json")

    # Copy existing files from source model
    for file in "${file_list[@]}"; do
        if [ -f "${model_name_or_path}/${file}" ]; then
            cp "${model_name_or_path}/${file}" "${dir}"
        fi
    done

    # Get absolute path
    abs_path=$(realpath "${dir}")

fi
