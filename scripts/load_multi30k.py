import json
import pickle
import json
import os
import sys
from tqdm import tqdm
import argparse



def main(dataset, train_or_val, lang):
    out_path = f"./data/{dataset}/annotations/{train_or_val}_captions_{lang}.json"
        
    with open('./data/'+dataset+'/raw/'+train_or_val+'.'+lang, 'r') as f:
        caps = f.readlines()
    with open('./data/'+dataset+'/image_splits/'+train_or_val+'.txt', 'r') as f:
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
    parser.add_argument('--train_or_val', choices=('train', 'val'))
    parser.add_argument('--lang')
    args = parser.parse_args()
    main("multi30k", args.train_or_val, args.lang)
