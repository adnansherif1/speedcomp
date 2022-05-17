#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=zanino

config=$1
seed=$2
lr=$3

echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
export PATH=/data/adnans/miniconda3/bin:$PATH
source ~/.bashrc
conda activate my-rdkit-env

echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

echo "python main.py --configs $config --num_workers 0 --lr $lr --seed $seed --devices $CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port 12424 main.py --configs $config --num_workers 0 --lr $lr --seed $seed --devices $CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 0 --lr $lr --seed $seed --devices $CUDA_VISIBLE_DEVICES