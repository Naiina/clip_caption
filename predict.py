import os
import argparse
import clip
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from clip_models import ClipCaptionModel, ClipCaptionPrefix, MappingType
#from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import skimage.io as io
import PIL.Image
import json
from os import listdir
from PIL import Image
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel , BloomForCausalLM

torch.manual_seed(0)
np.random.seed(0)

#D = torch.device

CPU = torch.device("cpu")




def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.embedding_func(tokens)
        for i in range(entry_length):
            outputs = model.modell(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.embedding_func(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.embedding_func(tokens)

            for i in range(entry_length):

                outputs = model.modell(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.embedding_func(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main_pred(device,model,dataset_path,model_name,prefix_length,pred_data_size,checkpoint,train_or_val,train_path,use_beam_search = False):


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False
    )
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--train_or_val', type=str)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    #parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--only_prefix', default='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    #parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument("--train_path")
    parser.add_argument("--checkpoint", default = "latest")
    parser.add_argument("--model_name", default = "bloom")
    parser.add_argument("--dataset")
    parser.add_argument("--data_size", default = 40000)
    args = parser.parse_args()
    model_name = {"bloom" : 'bigscience/bloom-560m', "gpt": "gpt2"}[args.model_name]
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    prefix_length = args.prefix_length
    dataset_name = args.dataset
    data_size = int(args.data_size)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    #if args.only_prefix:
    #weights_path = "coco_train/coco_prefix_latest.pt"
    """
    model_name = {"bloom" : 'bigscience/bloom-560m', "gpt": "gpt2"}[model_name]
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    #prefix_length = prefix_length
    
    pred_data_size = int(pred_data_size)
    #prefix_dim = 640 if is_rn else 512
    #mapp_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[mapping_type]

    #model = ClipCaptionPrefix(prefix_length, model_name = model_name, clip_length=prefix_length_clip, prefix_size=prefix_dim,
    #                              num_layers=num_layers, mapping_type=mapping_type)
        
    
    #if str(checkpoint) == "latest":
    #    weights_path = (train_path+"/coco_prefix_"+str(checkpoint)+".pt")
    #else :
    #    weights_path = (train_path+"/coco_prefix-00"+str(checkpoint)+".pt")
    #model.load_state_dict(torch.load(weights_path, map_location=CPU))
    #model = model.eval()
    #model = model.to(device)

    #create_annotation_val_file()
    
    
    predicted_captions = []

    i = 0
    data_path = "data/"+dataset_path+"/"+train_or_val+"/"
    
    #print(data_path)
    #exit()
    for f in tqdm(listdir(data_path)):
        #d = data[elem]
        i = i+1
        if "wit" in dataset_path:
            img_id = int(f[:-4])
        if dataset_path == "coco":
            img_id = int(f[19:-4])
        else:
            img_id = int(f[:-4])

        #filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(data_path+f)
        pil_image = PIL.Image.fromarray(image)
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(
                device, dtype=torch.float32
            )
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            out = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            out = generate2(model, tokenizer, embed=prefix_embed)
        d = {"image_id": img_id, "caption": out}
        #print(d)
        predicted_captions.append(d)


        if i%10000 == 0:
            #with open('filename', 'w', encoding='utf8') as json_file:
            #    json.dump("ברי צקלה", json_file, ensure_ascii=False)
            #json_object = json.dumps(predicted_captions)

            with open(train_path+"predicted_captions_"+train_or_val+"_"+str(checkpoint)+".json", "w", encoding = 'utf8') as outfile:
                json.dump(predicted_captions, outfile, ensure_ascii=False)

        if i>pred_data_size:
            with open(train_path+"/predicted_captions_"+train_or_val+"_"+str(checkpoint)+".json", "w",encoding="utf8") as outfile:
                json.dump(predicted_captions,outfile,ensure_ascii=False)
            break
    #json_object = json.dumps(predicted_captions)
    with open(train_path+"/predicted_captions_"+train_or_val+"_"+str(checkpoint)+".json", "w",encoding="utf8") as outfile:
        json.dump(predicted_captions,outfile,ensure_ascii=False)
    
    #del model
    #torch.cuda.empty_cache()
    
    


#if __name__ == '__main__':
    
#    main()
    
<<<<<<< Updated upstream
=======
    main()
    
>>>>>>> Stashed changes
