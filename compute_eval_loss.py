
import clip
import os
import argparse
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from train import ClipCaptionModel, ClipCaptionPrefix, MappingType , ClipCocoDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import skimage.io as io
import PIL.Image
import json
from os import listdir
from PIL import Image 
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CPU = torch.device("cpu")
checkpoint_dir = "coco_train_nina_modif_val"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    model = ClipCaptionPrefix(prefix_length =10, clip_length=10,
                                    num_layers=8, mapping_type="transformer")
    data = './coco_train_40pref_lenght/oscar_split_ViT-B_32_val.pkl'

    dataset = ClipCocoDataset(data, 10, normalize_prefix='normalize_prefix')
    val_dataloader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)


    for epoch in range(10):

        weights_path = checkpoint_dir + "/coco_prefix-00"+str(epoch)+".pt"
        model.load_state_dict(torch.load(weights_path, map_location=CPU))
        model = model.eval()
        model = model.to(device)
        l_loss = []

        for idx, (tokens, mask, prefix) in tqdm(enumerate(val_dataloader)):
                # tokens: one int per word in the caption + zero padd
                # mask: ones(prefix_lenght)+ ones(caption len) + zero padd
                # clip prefix of size batch_size * 512 (clip output)
            
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] #we want the predictions of what is happening after the image
            #torch_size: batch_size*max_len_caption*50257
            
            
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            l_loss.append(loss)
            print(loss)
            
        with open(f"/exports/eddie/scratch/s2523033/CLIP_prefix_caption/"+checkpoint_dir+"/loss_"+str(epoch)+"_val.pkl", 'wb') as f:
            pickle.dump(l_loss, f)