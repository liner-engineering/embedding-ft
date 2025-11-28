#!/bin/bash
set -e
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate deepspeed_train

MASTER_PORT=30001
DEVICES="0,1,2,3,4,5,6,7"

dataset_config=config/dataset_config/test1.yaml
src_dir="<your_model_path>"
output_dir=sanity_test_continue
log=logs/test2.log
save_steps=1
save_total_limit=10


max_length=512
train_batch_size=4096
per_device_train_batch_size=16
num_devices="${DEVICES//,/}"
num_devices="${#num_devices}"
echo "num_devices: $num_devices"
gradient_accumulation_steps=$((train_batch_size / per_device_train_batch_size / num_devices))



echo "GRADIENT_ACCUMULATION_STEPS: $gradient_accumulation_steps"

# Train
num_epochs=4
learning_rate="1e-5"
weight_decay=0.1
num_warmup_steps=5
lr_scheduler_type=linear

WANDB_DISABLED="true" CUDA_VISIBLE_DEVICES=$DEVICES nohup accelerate launch \
    --main_process_port $MASTER_PORT\
    --config_file config/deepspeed_config/deepspeed_zero3.yaml \
    src/train.py \
    --dataset_config $dataset_config \
    --sanity_test \
    --continual_learning \
    --max_length $max_length \
    --model_name_or_path $src_dir \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --max_steps 0 \
    --num_train_epochs $num_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --num_warmup_steps $num_warmup_steps \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    --output_dir $output_dir \
    --use_peft > $log 2>&1 


# 현재 위치에서 output/falcon-7b 디렉터리로 이동하지 않고 작업을 수행
max=0
dir=""
# 'output/falcon-7b/arcc-base2' 디렉터리의 'checkpoint-*' 형태의 모든 디렉터리를 순회
for file in $output_dir/checkpoint-*; do
    num=${file##*-}  # 디렉터리 이름에서 숫자 부분만 추출
    echo "file: $file"
    echo "num: $num"
    if ((num > max)); then  # 현재 숫자가 최대값보다 크면 업데이트
        max=$num
        dir=$file
    fi
done

if [ -d "${dir}" ]; then
    echo "Processing ${dir}..."
    # 파일 목록
    file_list=("configuration_RW.py" "modelling_RW.py" "tokenizer_config.json" "tokenizer.json" "special_tokens_map.json" "tokenizer.model" "config.json")

    # 파일 목록을 반복하며 존재하는 파일을 복사
    for file in "${file_list[@]}"; do
        if [ -f "${src_dir}/${file}" ]; then
            cp "${src_dir}/${file}" "${dir}"
        fi
    done
    
    # 절대 경로를 얻어서 변수에 저장합니다.
    abs_path=$(realpath "${dir}")

fi
