
import clip
import os
import argparse
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from clip_models import ClipCaptionModel, ClipCaptionPrefix, MappingType , ClipCocoDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import skimage.io as io
import PIL.Image
import json
from os import listdir
from PIL import Image 
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


import io
from statistics import mean 
import numpy as np

#CPU = torch.device("cpu")


def compute_loss(device,CPU,epoch,model,path,data_folder,val_set_size):
    #CPU = torch.device("cpu")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--path')
    #parser.add_argument('--nb_epochs')
    #parser.add_argument('--dataset')

    #args = parser.parse_args()
    #nb_epochs = int(args.nb_epochs)
    #print(nb_epochs)
    checkpoint_dir = path
    #dataset_name = args.dataset
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model = ClipCaptionPrefix(prefix_length =10, clip_length=10,
                                        num_layers=8, mapping_type="transformer", model_name='bigscience/bloom-560m')
        
        data = './data/'+data_folder+'/oscar_split_val.pkl'
        


        dataset = ClipCocoDataset(data, 10, normalize_prefix='normalize_prefix', model_name='bigscience/bloom-560m')
        val_dataloader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)

        #ll_loss = []
        #l_mean = []

        #for epoch in range(nb_epochs):

        weights_path = checkpoint_dir + "/coco_prefix-00"+str(epoch)+".pt"
        #model.load_state_dict(torch.load(weights_path, map_location=CPU))
        #model = model.eval()
        model = model.to(device)
        l_loss_eval = []
            

        for idx, (tokens, mask, prefix) in tqdm(enumerate(val_dataloader)):
                # tokens: one int per word in the caption + zero padd
                # mask: ones(prefix_lenght)+ ones(caption len) + zero padd
                # clip prefix of size batch_size * 512 (clip output)
            if idx>int(val_set_size):
                break
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] #we want the predictions of what is happening after the image
            #torch_size: batch_size*max_len_caption*50257
            
            
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            l_loss_eval.append( loss.cpu().detach().numpy())
            #print(loss)
                
            #with open(f"/exports/eddie/scratch/s2523033/CLIP_prefix_caption/"+checkpoint_dir+"/loss_"+str(epoch)+"_val.pkl", 'wb') as f:
            #    pickle.dump(l_loss_eval, f)
        mean_loss_eval= np.mean(l_loss_eval)
            #ll_loss.append(l_loss_eval)
        #del model
        #torch.cuda.empty_cache()
        #mean_loss_eval = l_lo
            


    

    """

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)


    ll_loss = []
    l_mean = []
    for i in range(nb_epochs):
        with open("./"+path+"/loss_"+str(i)+"_val.pkl", 'rb') as f:
            tensor_loss = CPU_Unpickler(f).load()

        loss = [elem.numpy() for elem in tensor_loss]
        ll_loss.append(loss)
        l_mean.append(np.mean(loss))
        print(l_mean)
    """

    return(mean_loss_eval)  
    #plt.plot(l_mean)
    #plt.title("mean loss on val set" )
    #plt.xlabel("epoch")
    #plt.savefig(args.path+"/loss_over_epoch_val.png")
