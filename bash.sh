#!/bin/bash
#$ -N xsmall
#$ -l h_rt=10:30:00
#$ -l h_vmem=80G
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
conda activate /exports/eddie/scratch/s2523033/anaconda/envs/clip_prefix_caption/
#export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033

data_folder="wit_small" 
train_folder="wit_small_train"
checkpoint="002"
metric="rouge"

python load_wit_dataset --outpath $data_folder --lang "en" --train_size 100 --val_size 50
python parse_coco.py --dataset $data_folder --train_or_val "val"
python parse_coco.py --dataset $data_folder --train_or_val "train"
python train.py --data "./data/${data_folder}/oscar_split.pkl" --out_dir $train_folder --epochs 2 --prefix_length 10 --prefix_length_clip 10 
python predict.py --bs 40 --train_or_val "val" --train_path $train_folder --checkpoint "002" --model_name "bloom" --dataset $data_folder
#python predict.py --only_prefix --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --train_or_val "val" --coco_train_path "coco_train_bloom/" --checkpoint "003" --model_name "bloom"
python compute_score.py --path $train_folder --checkpoint $checkpoint --dataset_name $datafolder --metric $metric
conda deactivate









