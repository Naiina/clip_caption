import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
import sys
from tqdm import tqdm
import argparse



def main(clip_model_type: str,dataset, train_or_val = "train"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
   
    out_path = f"./data/"+dataset+"/"+train_or_val+".pkl"
        
    with open('./data/'+dataset+'/annotations/'+train_or_val+'.fr', 'r') as f:
        caps = f.readlines()
    with open('./data/'+dataset+'/'+train_or_val+'.txt', 'r') as f:
        img = f.readlines()
    data = dict(zip(img, caps))

    with open('./data/'+dataset+'/annotations/'+'val.fr', 'r') as f:
        val_caps = f.readlines()
    with open('./data/'+dataset+'/'+'val.txt', 'r') as f:
        val_img = f.readlines()
    val = dict(zip(val_img, val_caps))
    
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    all_embeddings = []
    all_captions = []
    
    for i, (img_id, cap) in tqdm(enumerate(data.items())):
        img_id = img_id[:-5]
        d = {"image_id": img_id, "id": img_id, "caption": cap.strip()}
        filename = "./data/"+dataset+"/images/"+img_id+".jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device) # tensors size [1, 3, 224, 224]
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i+1) % 10000 == 0:
            print(f"{i}", file=sys.stderr)
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset')
    parser.add_argument('--train_or_val', choices=('train', 'val'))
    args = parser.parse_args()
    dataset = args.dataset
    main(args.clip_model_type, args.dataset, args.train_or_val)
