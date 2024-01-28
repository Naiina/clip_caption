#!/bin/bash
#$ -N multi30k_fr 
#$ -l h_rt=20:30:00
#$ -l h_vmem=80G
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
source ~/.bashrc
module load anaconda/2022.05
<<<<<<< Updated upstream
conda activate clipcap
=======
conda activate testclip

>>>>>>> Stashed changes
#export DEEPLAKE_DOWNLOAD_PATH=/exports/eddie/scratch/s2523033

data_folder="multi30k" 
train_folder="multi30k_train"
metric="rouge"
model=$"bloom"
epoch=5
checkpoint="003"
train_size=566852
val_size=202700
lang="en"

<<<<<<< Updated upstream
python load_wit_dataset.py --outpath $data_folder --lang $lang --train_size $train_size --val_size $val_size
python parse_coco.py --dataset $data_folder --train_or_val "val"
python parse_coco.py --dataset $data_folder --train_or_val "train"
accelerate launch train.py --data "./data/${data_folder}/oscar_split_train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model --prefix_length 10 --prefix_length_clip 10 
python predict.py --train_or_val "val" --train_path $train_folder --checkpoint $checkpoint --model_name $model --dataset $data_folder
python compute_score.py --path $train_folder --checkpoint $checkpoint --dataset_name $data_folder --metric $metric
=======

#python load_wit_dataset.py --outpath $data_folder --lang $lang --train_size $train_size --val_size $val_size
#python parse_multi30k.py --dataset $data_folder --train_or_val "val"
#python parse_multi30k.py --dataset $data_folder --train_or_val "train"
echo "im here"
accelerate launch train.py --data "./data/${data_folder}/train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model --prefix_length 10 --prefix_length_clip 10 
#python train.py --data "./data/${data_folder}/oscar_split_train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model--prefix_length 10 --prefix_length_clip 10 
#python predict.py --train_or_val "val" --train_path $train_folder --checkpoint $checkpoint --model_name $model --dataset $data_folder
#python compute_score.py --path $train_folder --checkpoint $checkpoint --dataset_name $data_folder --metric $metric
>>>>>>> Stashed changes
conda deactivate









