#!/bin/bash

# 环境列表
envs=(
  "ant-expert-v2"
  "halfcheetah-expert-v2"
  "hopper-expert-v2"
  "walker2d-expert-v2"
)

# seed 列表
seeds=(0 1)

# 对应 GPU 分配（按顺序分配给 env × seed）
gpus=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")

# 通用参数（不变的）
max_training_timesteps=1000000
expert_episode="one"
downstream="td3"

mkdir -p logs

# ================================
# 🧩 多组实验参数（每一组代表一轮）
# ================================
rounds=(
  # 格式: "update_timestep alpha wandb_name"
  "4000 10 noise0.5 30 1e-3 3e-4 1e-3"
)

# ================================
# 🚀 定义运行一轮的函数
# ================================
run_round() {
  local update_timestep=$1
  local alpha=$2
  local wandb_name=$3
  local f_epoch=$4
  local lr_f=$5
  local lr_actor=$6
  local lr_critic=$7
  local gpu_idx=0

  echo "=========================================="
  echo "🚀 Starting round: ${wandb_name}"
  echo "   update_timestep=${update_timestep}, alpha=${alpha}"
  echo "=========================================="

  for env in "${envs[@]}"; do
    for seed in "${seeds[@]}"; do
      cuda=${gpus[$gpu_idx]}
      echo "===== Running ${env} seed ${seed} on ${cuda} ====="
      python main.py \
        --env_name ${env} \
        --f_epoch ${f_epoch} \
        --max_training_timesteps ${max_training_timesteps} \
        --update_timestep ${update_timestep} \
        --downstream ${downstream} \
        --expert_episode ${expert_episode} \
        --minus \
        --lr_f ${lr_f} \
        --lr_actor ${lr_actor} \
        --lr_critic ${lr_critic} \
        --seed ${seed} \
        --using_icvf \
        --alpha ${alpha} \
        --cuda ${cuda} \
        --wandb_name "${wandb_name}" \
        > logs/${env}_s${seed}_${wandb_name}.log 2>&1 &
      ((gpu_idx++))
    done
  done

  # 等所有后台任务结束
  wait
  echo "✅ Round ${wandb_name} finished."
}

# ================================
# 🧠 主循环：自动跑所有 round
# ================================
for round in "${rounds[@]}"; do
  run_round $round
done

echo "🎉 All rounds finished successfully!"
