#!/bin/bash
#$ -N xbig
#$ -l h_rt=20:30:00
#$ -l h_vmem=80G
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
conda activate /exports/eddie/scratch/s2523033/anaconda/envs/clip_prefix_caption/
#export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033

data_folder="wit_big" 
train_folder="wit_big_train"
metric="rouge"
model=$"bloom"
epoch=4
checkpoint="003"
train_size=566852
val_size=202700
lang="en"

python load_wit_dataset.py --outpath $data_folder --lang $lang --train_size $train_size --val_size $val_size
python parse_coco.py --dataset $data_folder --train_or_val "val"
python parse_coco.py --dataset $data_folder --train_or_val "train"
accelerate launch train.py --data "./data/${data_folder}/oscar_split_train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model --prefix_length 10 --prefix_length_clip 10 
#python train.py --data "./data/${data_folder}/oscar_split_train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model--prefix_length 10 --prefix_length_clip 10 
python predict.py --train_or_val "val" --train_path $train_folder --checkpoint $checkpoint --model_name $model --dataset $data_folder
#python compute_score.py --path $train_folder --checkpoint $checkpoint --dataset_name $datafolder --metric $metric
conda deactivate









