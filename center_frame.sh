#!/bin/bash

#SBATCH --job-name frame
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y3
#SBATCH -o /data/psh68380/repos/LLaVA/%A-%x.out
#SBATCH -e /data/psh68380/repos/LLaVA/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12345

python -u /data/psh68380/repos/LLaVA/center_frame.py \
--dataset "ssv2" \
--frame_num 3 \
--frame_save_folder "/data/datasets/ssv2_center_frame/train/frame_num_3"
    
echo "Job finish"
exit 0