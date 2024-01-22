import json
import pickle
import json
import os
import sys
from tqdm import tqdm
import argparse



def main(clip_model_type, dataset, train_or_val = "train"):
    out_path = f"./data/"+dataset+"/annotations/"+train_or_val+".json"
        
    with open('./data/'+dataset+'/annotations/'+train_or_val+'.fr', 'r') as f:
        caps = f.readlines()
    with open('./data/'+dataset+'/'+train_or_val+'.txt', 'r') as f:
        img = f.readlines()
    data = dict(zip(img, caps))

    all_captions = []
    
    for i, (img_id, cap) in tqdm(enumerate(data.items())):
        img_id = img_id[:-5]
        d = {"image_id": img_id, "id": img_id, "caption": cap}
        all_captions.append(d)
    with open(out_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_captions, json_file)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset')
    parser.add_argument('--train_or_val', choices=('train', 'val'))
    args = parser.parse_args()
    dataset = args.dataset
    main(args.clip_model_type, args.dataset, args.train_or_val)
