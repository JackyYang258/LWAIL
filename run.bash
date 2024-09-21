python main.py \
--env_name walker2d-expert-v2 \
--f_epoch 20 \
--max_training_timesteps 2000000 \
--update_timestep 4000 \
--downstream 'td3' \
--one_episode \
--using_icvf \
--wandb_name walone_icvf_td3_log_ss
# --wandb_name hop_icvf_ppopre15_sigmoid_ss_full