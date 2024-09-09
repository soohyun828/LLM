#!/bin/bash

#SBATCH --job-name llava
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y7
#SBATCH -o /data/psh68380/repos/LLaVA/%A-%x.out
#SBATCH -e /data/psh68380/repos/LLaVA/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12345
workspaceFolder = "/data/psh68380/repos/LLaVA"
export PYTHONPATH="${workspaceFolder}:${workspaceFolder}/llava/eval"

# Set the path to save checkpoints
# OUTPUT_DIR='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/eval_lr_5e-4_epoch_100'
# path to UCF101 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='/local_datasets/ai_hub_sketch_mw/01/val'
# path to pretrain model
# MODEL_PATH='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
# --data_root "/local_datasets/ai_hub_sketch_mw/01/train"
python -u /data/psh68380/repos/LLaVA/llava/eval/run_llava_for_SCAT.py \
--model_path "liuhaotian/llava-v1.5-7b" \
--image_path "/local_datasets/ai_hub_mw_cropped" \
--sep "," \
--temperature 0 \
--num_beams 1 \
--max_new_tokens 512 \
    
echo "Job finish"
exit 0