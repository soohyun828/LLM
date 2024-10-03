#!/bin/bash

#SBATCH --job-name gpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y2
#SBATCH -o /data/psh68380/repos/LLaVA/%A-%x.out
#SBATCH -e /data/psh68380/repos/LLaVA/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12345


python -u /data/psh68380/repos/LLaVA/run_gpt.py \
--descriptor_type "spatio" \
--label_file "/data/psh68380/repos/Video-CBM_/data/kinetics400_classes.txt" 

    
echo "Job finish"
exit 0