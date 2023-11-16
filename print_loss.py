import matplotlib.pyplot as plt
import pickle
import torch
import io
from statistics import mean 
import numpy as np
import argparse

CPU = torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





parser = argparse.ArgumentParser()
parser.add_argument('--train_or_val', default="val", choices=('train', 'val'))
parser.add_argument('--path', default="coco_train_nina_modif_val")
args = parser.parse_args()
train_or_val = args.train_or_val

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
if train_or_val == "val":
    l_mean = []
    for i in range(10):
        with open("./"+args.path+"/loss_"+str(i)+"_val.pkl", 'rb') as f:
            tensor_loss = CPU_Unpickler(f).load()

        loss = [elem.numpy() for elem in tensor_loss]
        l_mean.append(np.mean(loss))
        print(l_mean)

else :
    l_mean = []
    for i in range(10):
        with open("./"+path+"/loss_"+str(i)+".pkl", 'rb') as f:
            tensor_loss = CPU_Unpickler(f).load()

        loss = [elem.detach().numpy() for elem in tensor_loss]
        l_mean.append(np.mean(loss))
        print(l_mean)
    
plt.plot(l_mean)
plt.title("mean loss on %s set" %train_or_val )
plt.xlabel("epoch")
plt.savefig(args.path+"/loss_over_epoch"+train_or_val+".png")

"""
loss=2.58]
coco_prefix: 100%|██████████| 14168/14168 [31:15<00:00,  7.55it/s, loss=2.47]
coco_prefix: 100%|██████████| 14168/14168 [31:14<00:00,  7.56it/s, loss=2.29]
coco_prefix: 100%|██████████| 14168/14168 [32:42<00:00,  7.22it/s, loss=2.32]
coco_prefix: 100%|██████████| 14168/14168 [32:53<00:00,  7.18it/s, loss=2.3] 
coco_prefix: 100%|██████████| 14168/14168 [32:10<00:00,  7.34it/s, loss=2.14]
coco_prefix: 100%|██████████| 14168/14168 [37:22<00:00,  6.32it/s, loss=2.25]
coco_prefix: 100%|██████████| 14168/14168 [31:09<00:00,  7.58it/s, loss=1.97]
coco_prefix:  94%|█████████▍| 13374/14168 [29:24<01:44,  7.60it/s, loss=1.96

"""