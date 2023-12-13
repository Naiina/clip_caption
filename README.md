# CLIP multilanguage prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  





## Inspired from the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)




## Description  
We replaced GPT2 by Bloom to get a multilanguage captioning network
we used the WIT dataset 





## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## WIT training

The bash.sh file gives an example of a training over a toy dataset.

To load the wit dataset, use:

```
python load_wit_dataset.py --outpath $data_folder --lang $lang --train_size $train_size --val_size $val_size
```
To parse the val and train set use

```
python parse_coco.py --dataset $data_folder --train_or_val "val"
python parse_coco.py --dataset $data_folder --train_or_val "train"
```
to train and predict use
```
python train.py --data "./data/${data_folder}/oscar_split_train.pkl" --out_dir $train_folder --epochs $epoch --model_name $model--prefix_length 10 --prefix_length_clip 10 
python predict.py --train_or_val "val" --train_path $train_folder --checkpoint $checkpoint --model_name $model --dataset $data_folder
```

TO FIX: I am running compute_score.py locally because I can't install evaluate on the cluster






## Citation
If you use this code for your research, please cite:
```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```




## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

## Contact
For any inquiry please contact us at our email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


