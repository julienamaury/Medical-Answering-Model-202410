#!/bin/bash
# Copyright 2024 Research Center of Body Data Science from South China University of Technology. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File: run_single_gpu_train_model_llama3-8b.sh
# Description: 
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2023/10/15

# Usage:
# cd ~/scutcyr/LLaMA-Factory/train_models/jiyichat
# chmod +x run_single_gpu_train_model_llama3-8b.sh
# CUDA_VISIBLE_DEVICES=6 ./run_single_gpu_train_model_llama3-8b.sh 8 0.0001

# 检查是否有足够的参数传入
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <lora_rank> <learning_rate>"
    exit 1
fi

# 读取参数
LORA_RANK=$1
LEARNING_RATE=$2

# 与批量运行无关的参数
ROOT_BASE_DIR=/home/phd-chen.yirong/scutcyr/LLaMA-Factory
MODEL_NAME_OR_PATH=/home/sharefiles/modelscope.cn/Meta-Llama-3-8B-Instruct

# 固定数据集
DATASET="RAG_MedQA_Mainland_train"
RESULT_BASE_SAVE_DIR=${ROOT_BASE_DIR}/saves/jiyichat_on_${DATASET}/llama3-8b/lora
NUM_TRAIN_EPOCHS=2
# --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,
# constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr}
LR_SCHEDULER_TYPE="cosine"


# 切换到项目根路径
cd $ROOT_BASE_DIR


echo "【通知】llamafactory-cli开始执行训练任务，超参数配置：epochs=${NUM_TRAIN_EPOCHS}；lora_rank=$LORA_RANK；learning_rate=$LEARNING_RATE；lr_scheduler=${LR_SCHEDULER_TYPE}"
llamafactory-cli train \
    --run_name="sft_with_epochs_${NUM_TRAIN_EPOCHS}_lora_rank_${LORA_RANK}_lr_${LEARNING_RATE}_lr_scheduler_${LR_SCHEDULER_TYPE}" \
    --seed=42 \
    --data_seed=42 \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --stage="sft" \
    --do_train \
    --finetuning_type="lora" \
    --lora_target="all" \
    --lora_rank=$LORA_RANK \
    --dataset=$DATASET \
    --template="llama3" \
    --cutoff_len=32000 \
    --max_samples=10000000 \
    --val_size=0.001 \
    --overwrite_cache \
    --preprocessing_num_workers=16 \
    --output_dir="${RESULT_BASE_SAVE_DIR}/sft_with_epochs_${NUM_TRAIN_EPOCHS}_lora_rank_${LORA_RANK}_lr_${LEARNING_RATE}_lr_scheduler_${LR_SCHEDULER_TYPE}" \
    --logging_steps=10 \
    --save_steps=500 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size=6 \
    --auto_find_batch_size \
    --gradient_accumulation_steps=4 \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --lr_scheduler_type=$LR_SCHEDULER_TYPE \
    --warmup_ratio=0.1 \
    --fp16 \
    --per_device_eval_batch_size=1 \
    --evaluation_strategy="epoch"

if [ $? -eq 0 ]; then
    echo "【通知】成功执行推理任务，超参数配置：epochs=${NUM_TRAIN_EPOCHS}；lora_rank=$LORA_RANK；learning_rate=$LEARNING_RATE；lr_scheduler=${LR_SCHEDULER_TYPE}"
else
    echo "【错误】执行推理任务失败，超参数配置：epochs=${NUM_TRAIN_EPOCHS}；lora_rank=$LORA_RANK；learning_rate=$LEARNING_RATE；lr_scheduler=${LR_SCHEDULER_TYPE}"
fi
