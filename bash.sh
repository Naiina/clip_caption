#!/bin/bash
#$ -N xpred_on_coco
#$ -l h_rt=10:30:00
#$ -l h_vmem=80G
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
conda activate /exports/eddie/scratch/s2523033/anaconda/envs/clip_prefix_caption/
#export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033
#python parse_coco.py

#python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train_bloom/ --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --epochs 3
#python compute_eval_loss.py 
python predict.py --only_prefix --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --train_or_val "val" --coco_train_path "coco_train_bloom/" --checkpoint "002" --model_name "bloom" --dataset "coco"
#python predict.py --only_prefix --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --train_or_val "val" --coco_train_path "coco_train_bloom/" --checkpoint "003" --model_name "bloom"
conda deactivate









