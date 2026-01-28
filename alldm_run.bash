#!/bin/bash

# ================= 配置区域 =================
# 在这里填入你可用的 GPU ID 列表
# 因为一共有 3(环境) * 2(icvf/no-icvf) = 6 个实验
# 请确保这里至少有 6 个 ID，或者根据显存情况复用 ID (例如: (0 1 2 0 1 2))
GPU_IDS=(0 1 2 3 4 5) 

# 基础参数
SEED=1
EPOCH=30
MAX_STEPS=2000000
UPDATE_STEP=4000
DOWNSTREAM="td3"
WANDB_BASE="dmone"

# 任务列表 (walk, run, stand)
TASKS=("walk" "run" "stand")

# ICVF 开关列表 (1代表开启, 0代表关闭)
ICVF_SETTINGS=(1 0)

# 计数器，用于分配 GPU
count=0

# ================= 循环执行 =================
for task in "${TASKS[@]}"; do
    # 构造完整的环境名称
    ENV_NAME="dm_control/humanoid-${task}-v0"

    for use_icvf in "${ICVF_SETTINGS[@]}"; do
        
        # 1. 获取当前任务分配的 GPU
        gpu_id=${GPU_IDS[$count]}
        
        # 2. 根据是否使用 ICVF 设置参数和命名
        if [ "$use_icvf" -eq 1 ]; then
            ICVF_FLAG="--using_icvf"
            WANDB_NAME="${WANDB_BASE}_${task}_with_icvf"
        else
            ICVF_FLAG=""
            WANDB_NAME="${WANDB_BASE}_${task}_no_icvf"
        fi

        echo "---------------------------------------"
        echo "正在启动任务 [$((count+1))/6]:"
        echo "Env: $ENV_NAME | ICVF: $use_icvf | GPU: cuda:$gpu_id"
        
        # 3. 执行命令 (使用 nohup 也可以，这里使用 & 后台运行)
        # 注意：这里保留了你提供的 --minus, --expert_episode multiple 等参数
        python main_dm.py \
            --env_name "$ENV_NAME" \
            --f_epoch $EPOCH \
            --max_training_timesteps $MAX_STEPS \
            --update_timestep $UPDATE_STEP \
            --downstream "$DOWNSTREAM" \
            --expert_episode one \
            --minus \
            --seed $SEED \
            --cuda "cuda:$gpu_id" \
            --wandb_name "$WANDB_NAME" \
            $ICVF_FLAG > "log_${task}_icvf${use_icvf}.txt" 2>&1 &

        # 获取后台进程 PID
        pid=$!
        echo "PID: $pid | Log saved to: log_${task}_icvf${use_icvf}.txt"
        
        # 计数器加1
        count=$((count + 1))
        
        # 可选：稍微暂停几秒，防止瞬间并发导致文件读取冲突
        sleep 3
    done
done

echo "---------------------------------------"
echo "所有实验已在后台启动。"