python main.py \
--env_name ant-expert-v2 \
--f_epoch 40 \
--max_training_timesteps 1500000 \
--update_timestep 4000 \
--downstream 'td3' \
--one_episode \
--seed 0 \
--using_icvf \
--wandb_name antone_icvf_td3_log_ss
# --wandb_name hop_icvf_ppopre15_sigmoid_ss_full