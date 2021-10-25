#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=como

config=$1
seed=$2
lr=$3

echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
export PATH=/data/ethanbmehta/miniconda3/bin:$PATH
source ~/.bashrc
conda activate graph-aug

echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

echo "python main.py --configs $config --num_workers 8 --lr $lr --seed $seed --devices $CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --lr $lr --seed $seed --devices $CUDA_VISIBLE_DEVICES
