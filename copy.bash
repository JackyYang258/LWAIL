#!/bin/bash

# 拷贝并重命名模型文件
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_10k/ant-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/ant_10k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_10k/hopper-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/hopper_10k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_10k/halfcheetah-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/halfcheetah_10k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_10k/walker2d-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/walker2d_10k.pt

cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_50k/ant-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/ant_50k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_50k/hopper-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/hopper_50k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_50k/halfcheetah-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/halfcheetah_50k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_50k/walker2d-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/walker2d_50k.pt

cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_100k/ant-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/ant_100k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_100k/hopper-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/hopper_100k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_100k/halfcheetah-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/halfcheetah_100k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_100k/walker2d-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/walker2d_100k.pt

cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_500k/ant-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/ant_500k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_500k/hopper-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/hopper_500k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_500k/halfcheetah-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/halfcheetah_500k.pt
cp /home/kaiyan3/siqi/ICVF_PyTorch/experiment_output_500k/walker2d-random-v2/phi_400000.pt /home/kaiyan3/siqi/IntentDICE/model/walker2d_500k.pt
echo "所有模型文件已成功拷贝并重命名！"