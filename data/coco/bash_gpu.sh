#!/bin/bash
#$ -N pumpkin
#$ -l h_rt=00:05:00
#$ -l h_vmem=40G
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
#module load anaconda
conda activate /exports/eddie/scratch/s2523033/anaconda/envs/clip_prefix_caption/
#conda create -n mypython python torch
#source activate mypython
#conda list
#pip install torch
export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033
python ./clip_MLP_bloom.py
conda deactivate









