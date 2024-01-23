from compute_score import compute_score_rouge
from compute_eval_loss import compute_loss
import clip_models
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
#from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel , BloomForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import os
import pickle
import sys
import argparse 
import json
from typing import Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
import neptune
from predict import main_pred

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"


l_lr = [5e-6, 2e-5, 5e-5]
#l_batch_size = [30, 40, 50]

torch_seed = 2
torch.manual_seed(torch_seed)







def train(run,train_path,train_size,nb_epochs, dataset: clip_models.ClipCocoDataset, model: clip_models.ClipCaptionModel, args,
          lr: float = 5e-5, warmup_steps: int = 5000,  output_prefix: str = "" ):
    # Create a Neptune run object

    

    #device = torch.device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CPU = torch.device("cpu")
    accelerator = Accelerator()
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(train_path):
        os.makedirs(train_path)
   
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    model, optimizer, train_dataloader, scheduler = accelerator.prepare( model, optimizer, train_dataloader, scheduler)
    # save_config(args)
    #for epoch in range(1):
    #l_mean_loss = []
    #l_norm_param = []
    
    patience = 0
    mem_loss_eval = float('inf')
    for epoch in range(nb_epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc="train_epoch"+str(epoch))
        l_loss_train = []
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            if idx > int(train_size):
                break
            # tokens: one int per word in the caption + zero padd (from bloom tokeniser)
            # mask: ones(prefix_lenght)+ ones(caption len) + zero padd
            # clip prefix of size batch_size * 512 (clip output)
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1] #we want the predictions of what is happening after the image
            #torch_size: batch_size*max_len_caption*50257
            
            
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=-100)
            
            #print(loss)
            accelerator.backward(loss)
            #loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            int_loss = loss.cpu().detach().numpy()
            l_loss_train.append(int_loss)
            #print(idx)
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(train_path, f"{output_prefix}_latest.pt"),
                )
        #grads = [ param.grad.detach().flatten() for param in model.parameters() if param.grad is not None and param.requires_grad]
        #norm = torch.cat(grads).norm()
        #l_norm_param.append(norm)
        
        progress.close()
        #l_mean_loss.append(np.mean(l_loss))
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(train_path, f"{output_prefix}-{epoch:03d}.pt"),
            )

        #for elem in l_train_mean_loss:
        run["train_mean_loss_over_epochs"].append(float(np.mean(l_loss_train)))
        #for elem in l_norm_param:
        #run["norm_param"].append(norm)

        model.eval()
        predicted_capt = train_path+"/predicted_captions_val_"+str(epoch)+".json"
        main_pred(device,model,args.data_folder,args.model_name,args.prefix_length,args.val_size,epoch,
                "val",train_path,use_beam_search = False)
        #results = compute_score_rouge(train_path,predicted_capt,args.metric,args.data_folder,epoch)
        #print(results)
        #run["rouge_score_epoch "+str(epoch+1)]=results
        #print("predict_end")
        mean_loss_eval = compute_loss(device,CPU,epoch,model,train_path,args.data_folder,args.val_size)
        run["eval_loss"].append( mean_loss_eval )
        print (patience)

        if mem_loss_eval < mean_loss_eval and epoch>0:
            patience = patience + 1
        else:
            mem_loss_eval = mean_loss_eval

        if patience == 2:
            break



            #with open(f"loss_{epoch:03d}.pkl", 'wb') as f:
            #    pickle.dump(l_loss, f)
    #plt.plot(l_mean_loss)
    #plt.title("mean loss on train set" )
    #plt.xlabel("epoch")
    #plt.savefig(train_path+"/loss_over_epoch_train.png")

    #return l_mean_loss, l_norm_param


def main_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed_data', default='./data/oscar_split_train.pkl')
    parser.add_argument("--dataset_name")
    #parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=6)
    #parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=32)
    #parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--only_prefix', default='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    #parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--model_name', type=str, default = "bloom")
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument("--train_path")
    parser.add_argument('--metric')
    parser.add_argument("--train_size", default = 400000)
    parser.add_argument("--val_size", default = 40000)
    parser.add_argument("--data_folder")
    parser.add_argument("--flickr", default = False)
    parser.add_argument("--train_from", default = None)


    args = parser.parse_args()
    is_flickr = args.flickr
    model_name = {"bloom" : 'bigscience/bloom-560m', "gpt": "gpt2"}[args.model_name]
    prefix_length = args.prefix_length
    nb_epochs = args.epochs
    #dataset_name = args.dataset_name
    dataset = clip_models.ClipCocoDataset(args.parsed_data, prefix_length, model_name, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    mapp_type = {'mlp': clip_models.MappingType.MLP, 'transformer': clip_models.MappingType.Transformer}[args.mapping_type]
    #checkpoint = args.checkpoint
    train_path = args.train_path
    
    lr = args.lr
    lr=float(lr)
    if is_flickr:
        project="naiina/clip-coco-wit-flickr"
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzllNjM4MS0zYjBhLTQwNGUtOGM3Mi1hYjE3ZTVjOWVjMTgifQ=="
        tags=["clip-bloom","coco-en-wit-de-flickr-de"]
    else:
        project="naiina/clip-prefix-caption-multilingual"
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzllNjM4MS0zYjBhLTQwNGUtOGM3Mi1hYjE3ZTVjOWVjMTgifQ=="
        tags=["clip-bloom", "wit-en"]
    run = neptune.init_run(
        project=project, 
        api_token=api_token, tags=tags,  # optional
    )
    
    run["torchseed"] = torch_seed
    run["lr"] = lr
    
    run["batch_size"] = args.bs
    run["train_size"] = args.train_size
    run["val_size"] = args.val_size
    run["train_path"] = args.train_path 

    
    
    model = clip_models.ClipCaptionPrefix(prefix_length, model_name, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
    
    if is_flickr=="True":
        weights_path = args.train_from + "/coco_prefix-002.pt"
        CPU = torch.device("cpu")
        model.load_state_dict(torch.load(weights_path, map_location=CPU))
    

    # Log single value
    # Specify a field name ("seed") inside the run and assign a value to it
    warmup_steps = 5000

    run["warmup_steps"] = warmup_steps

    
    

    train(run,train_path,args.train_size,nb_epochs, dataset, model, args, lr, warmup_steps, output_prefix=args.prefix)
    
    #del model
    #torch.cuda.empty_cache()

    

    #l_checkpoint = [1,3,5]
    
    #metric = args.metric

    """
    print("start to predict")
    for checkpoint in l_checkpoint:
        print(checkpoint)
        predicted_capt = train_path+"/predicted_captions_val_"+str(checkpoint)+".json"
        main_pred(args.data_folder,args.model_name,prefix_length,args.val_size,prefix_dim,args.mapping_type,checkpoint,
                "val",train_path,args.is_rn,args.num_layers,args.prefix_length_clip,use_beam_search = False)
        results = compute_score_rouge(train_path,predicted_capt,metric,args.data_folder,checkpoint)
        print(results)
        run["rouge_score_epoch "+str(checkpoint+1)]=results
    print("predict_end")
    l_mean,l_loss = compute_loss(train_path, nb_epochs,args.data_folder,args.val_size)
    """
    
    
    

    #print("l_mean",l_mean)
    #print("l_loss_type:",type(l_loss))
    #for elem in l_mean:
        #print("type_in l_mean",type(elem),"l_meantype: ",type(l_mean))
    #    run["val_mean_loss_over_epochs"].append(float(elem))
    #for i,l in enumerate(l_loss):
        #print(type(l))
    #    for elem in l:
    #        
    #        run["loss_epoch "+str(i)].append(elem)
    
    run.stop()



#if __name__ == '__main__':
main_all()
    

