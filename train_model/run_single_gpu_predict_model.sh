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

# File: run_single_gpu_predict_model.sh
# Description: 
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2023/10/15

# Usage:
# chmod +x run_single_gpu_predict_model.sh
# CUDA_VISIBLE_DEVICES=6 ./run_single_gpu_predict_model.sh 0.3 0.7 2000

# 检查是否有足够的参数传入
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <temperature> <top_p> <checkpoint_steps>"
    exit 1
fi

# 读取参数
TEMPERATURE=$1
TOP_P=$2
checkpoint_steps=$3

# 与批量运行无关的参数
ROOT_BASE_DIR=/home/phd-chen.yirong/scutcyr/LLaMA-Factory
EVAL_BATCH_SIZE=2


MODEL_NAME=jiyichat_on_RAG_MedQA_Mainland_train
#-checkpoint-${checkpoint_steps}
# 微调的模型路径
ADAPTER_NAME_OR_PATH=${ROOT_BASE_DIR}/saves/jiyichat_on_RAG_MedQA_Mainland_train/qwen1.5-14b/lora/sft_with_epochs_3_lora_rank_8_lr_0.0001_lr_scheduler_cosine/checkpoint-${checkpoint_steps}

# Meta-Llama-3-8B-Instruct
# Qwen1.5-14B-Chat

MODEL_NAME_OR_PATH=/home/sharefiles/huggingface.co/Qwen1.5-14B-Chat
# /home/sharefiles/huggingface.co/Qwen1.5-14B-Chat
# /home/sharefiles/modelscope.cn/Meta-Llama-3-8B-Instruct
# /home/sharefiles/huggingface.co/Llama3-8B-Chinese-Chat

# 数据集
# 两个不同的测试集
# 无检索知识的：MedQA_Mainland_test
# 有检索知识的：RAG_MedQA_Mainland_test
# 小规模，无检索知识的：MedQA_Mainland_test_300
# 小规模，有检索知识的：RAG_MedQA_Mainland_test_300
DATASET="RAG_MedQA_Mainland_test_300"

# 结果保存路径
RESULT_BASE_SAVE_DIR=/home/phd-chen.yirong/scutcyr/LLaMA-Factory/train_models/jiyichat/results

# 超参数
TOP_K=20
MAX_NEW_TOKENS=4096

# 切换到项目根路径
cd $ROOT_BASE_DIR


echo "【通知】llamafactory-cli开始执行推理任务，超参数配置：temperature=$TEMPERATURE；top_p=$TOP_P；模型路径：$ADAPTER_NAME_OR_PATH"
#     --overwrite_output_dir \
llamafactory-cli train \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --adapter_name_or_path=$ADAPTER_NAME_OR_PATH \
    --stage="sft" \
    --do_predict \
    --finetuning_type="lora" \
    --dataset=$DATASET \
    --template="qwen" \
    --cutoff_len=32000 \
    --max_samples=10000 \
    --overwrite_cache \
    --preprocessing_num_workers=16 \
    --output_dir="${RESULT_BASE_SAVE_DIR}/predict_${DATASET}_use_${MODEL_NAME}_checkpoint_${checkpoint_steps}_with_temperature_${TEMPERATURE}_top_p_${TOP_P}_top_k_${TOP_K}" \
    --per_device_eval_batch_size=$EVAL_BATCH_SIZE \
    --predict_with_generate \
    --temperature=$TEMPERATURE \
    --top_p=$TOP_P \
    --top_k=$TOP_K \
    --max_new_tokens=$MAX_NEW_TOKENS

if [ $? -eq 0 ]; then
    echo "【通知】成功执行推理任务，超参数配置：temperature=$TEMPERATURE；top_p=$TOP_P；模型路径：$ADAPTER_NAME_OR_PATH"
else
    echo "【错误】执行推理任务失败，超参数配置：temperature=$TEMPERATURE；top_p=$TOP_P；模型路径：$ADAPTER_NAME_OR_PATH"
fi
