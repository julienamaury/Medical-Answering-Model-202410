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

# File: run_multiple_gpu_predict_model.sh
# Description: 
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2024/05/27

# Usage:
# chmod +x run_single_gpu_predict_model.sh
# chmod +x run_multiple_gpu_predict_model.sh
# ./run_multiple_gpu_predict_model.sh


# 项目根路径
# 其存在run_single_gpu_predict_model.sh
ROOT_BASE_DIR=/home/phd-chen.yirong/scutcyr/LLaMA-Factory/train_models/jiyichat

# 切换到项目根路径
cd $ROOT_BASE_DIR

CUDA_VISIBLE_DEVICES_LIST=("0" "1" "2" "3" "4" "5" "6" "7")

# 温度系数数组
temperatures=("0.95" "0.9" "0.85" "0.8" "0.75" "0.7" "0.65" "0.6" "0.5" "0.4" "0.3")
# top_p数组
top_ps=("0.95" "0.9" "0.85" "0.8" "0.75" "0.7" "0.65")

# checkpoint
checkpoints=("2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" "6000" "6500" "7000" "7500" "8000" "9000" "10000")


temperatures_and_top_ps=()

# 初始LORA_RANK_LEARNING_RATES队列
for temperature in "${temperatures[@]}"; do
    for top_p in "${top_ps[@]}"; do
        temperatures_and_top_ps+=("$temperature $top_p")
    done
done

# 任务队列，存储所有任务组合
task_queue=()

# 初始化任务队列
# <temperature> <top_p> <checkpoint_steps>
for temperature_and_top_p in "${temperatures_and_top_ps[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
        task_queue+=("$temperature_and_top_p $checkpoint")
    done
done

echo "【bash提示】推理任务总数为: ${#task_queue[@]}"


# 超参数
MODEL_NAME=jiyichat_on_RAG_MedQA_Mainland_train
DATASET="RAG_MedQA_Mainland_test_300"
TOP_K=20
MAX_NEW_TOKENS=4096



# 函数：运行模型推理任务
run_inference() {
    local temp=$1
    local top_p=$2
    local checkpoint=$3
    local gpu_index=$4

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_index

    # 这里替换成实际的推理命令
    echo "【bash提示】Running inference on GPU $gpu_index with temperature $temp and top_p $top_p on model: sft_with_epochs_3_lora_rank_8_lr_0.0001_lr_scheduler_cosine/checkpoint-${checkpoint}"
    nohup sh run_single_gpu_predict_model.sh "$temp" "$top_p" "$checkpoint" > "${ROOT_BASE_DIR}/nohup_subtasks_log/predict_${DATASET}_use_${MODEL_NAME}_checkpoint_${checkpoint}_with_temperature_${temp}_top_p_${top_p}_top_k_${TOP_K}_nohup.log" 2>&1 &
    echo "【bash提示】输出运行日志到：${ROOT_BASE_DIR}/nohup_subtasks_log/predict_${DATASET}_use_${MODEL_NAME}_checkpoint_${checkpoint}_with_temperature_${temp}_top_p_${top_p}_top_k_${TOP_K}_nohup.log"
    sleep 5
    # 模拟推理任务执行时间
    #sleep 10
}

# 主循环，持续检查GPU状态并分配任务
echo "【bash提示】进入主循环"
while true; do
    for gpu_index in "${CUDA_VISIBLE_DEVICES_LIST[@]}"; do
        # 检查GPU是否空闲
        nvidia-smi --id="$gpu_index" | grep -q 'No running processes found'
        
        if [ $? -eq 0 ]; then
            # GPU空闲，分配新任务
            echo "【bash提示】GPU: ${gpu_index}处于空闲状态，启动任务"
            if [ ${#task_queue[@]} -gt 0 ]; then
                task=${task_queue[0]}
                IFS=' ' read -ra numbers <<< ${task}
                task_queue=("${task_queue[@]:1}") # 移除队列中的第一个任务
                run_inference "${numbers[0]}" "${numbers[1]}" "${numbers[2]}" "$gpu_index"
                sleep 10
            fi
        fi
    done

    # 检查是否所有任务都已完成
    if [ ${#task_queue[@]} -eq 0 ]; then
        echo "【bash提示】All inference tasks have been completed."
        break
    fi

    # 短暂休眠，避免CPU过载
    sleep 120 # 每隔120秒查询1次
done


# 等待所有后台任务完成
wait

echo "All inference tasks have been completed."


