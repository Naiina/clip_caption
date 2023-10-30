#!/bin/bash
#$ -N zclippref
#$ -l h_rt=05:00:00
#$ -l h_vmem=60G
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
conda activate /exports/eddie/scratch/s2523033/anaconda/envs/clip_prefix_caption/
#export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033
#python parse_coco.py
python predict.py --mapping_type transformer --prefix_length 40 --prefix_length_clip 40
#python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer --prefix_length 40 --prefix_length_clip 40
conda deactivate









