import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse



def main(clip_model_type: str,dataset, train_or_val = "train"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
   
    out_path = f"./data/"+dataset+"/oscar_split_"+train_or_val+".pkl"
        
    with open('./data/'+dataset+'/annotations/'+train_or_val+'_caption.json', 'r') as f:
        data = json.load(f)
    

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
    #for i in tqdm(range(100)):
        d = data[i]
        img_id = d["image_id"]
        if dataset == "coco":
            filename = f"./data/coco/"+train_or_val+"/COCO_"+train_or_val+"2014_{int(img_id):012d}.jpg"
        else:
            filename = "./data/"+dataset+"/"+train_or_val+"/"+img_id+".jpg"
        
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device) # tensors size [1, 3, 224, 224]
        #print(image.size())
        
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        
        if (i + 1) % 10000 == 0:
            print(f"{i}")
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
    parser.add_argument('--train_or_val', choices=('train', 'val', 'test'))
    args = parser.parse_args()
    dataset = args.dataset
    train_or_val = args.train_or_val
    exit(main(args.clip_model_type, dataset, train_or_val))