python main_dm.py \
--env_name dm_control/ball_in_cup-catch-v0 \
--f_epoch 30 \
--max_training_timesteps 1000000 \
--update_timestep 4000 \
--downstream 'td3' \
--expert_episode multiple \
--minus \
--seed 1 \
--cuda cuda:6 \
--wandb_name dmicvf
# --using_icvf \