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
parser.add_argument('--path', default="coco_train_wit")
parser.add_argument('--nb_epochs')
args = parser.parse_args()
nb_epochs = int(args.nb_epochs)
print(nb_epochs)
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
    for i in range(nb_epochs):
        with open("./"+args.path+"/loss_"+str(i)+"_val.pkl", 'rb') as f:
            tensor_loss = CPU_Unpickler(f).load()

        loss = [elem.numpy() for elem in tensor_loss]
        l_mean.append(np.mean(loss))
        print(l_mean)

else :
    l_mean = []
    for i in range(nb_epochs):
        with open("./"+path+"/loss_"+str(i)+".pkl", 'rb') as f:
            tensor_loss = CPU_Unpickler(f).load()

        loss = [elem.detach().numpy() for elem in tensor_loss]
        l_mean.append(np.mean(loss))
        print(l_mean)
    
plt.plot(l_mean)
plt.title("mean loss on %s set" %train_or_val )
plt.xlabel("epoch")
plt.savefig(args.path+"/loss_over_epoch"+train_or_val+".png")
