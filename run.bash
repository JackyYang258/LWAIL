python main.py \
--env_name ant-expert-v2 \
--f_epoch 30 \
--max_training_timesteps 1000000 \
--update_timestep 4000 \
--downstream 'td3' \
--expert_episode one \
--minus \
--seed 0 \
--wandb_name reward