

from datasets import load_dataset
from torchvision import transforms
from PIL import Image  
import PIL  
import json
from torch.utils.data import DataLoader
import imghdr
from tqdm import tqdm
import argparse 

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

def generate_image_folder_and_annot_file(target_lang, train_set_size, val_set_size, im_folder_train, im_folder_val,caption_file_train, caption_file_val):

    data = load_dataset("wikimedia/wit_base", split="train", streaming=True)
    i = 0
    l_caption = []
    im_folder = im_folder_train
    print("hey")
    b = True
    real_train_set_size = train_set_size
    for idx, k in tqdm(enumerate(data)):
        
        #capt = k['caption_attribution_description'] #ref, filter langu
        wit_info = k["wit_features"]
        capt = wit_info["caption_reference_description"]
        lang = wit_info["language"]
        
        #print(lang)
        for c,l in zip(capt,lang):
            #print("bef", l)
            if l == target_lang and c != None:
                #print("inside",l)
                if c[-1] != ".":
                    c = c+"."
                #print(c)
                image = k['image']
                if 'Jpeg' in str(type(image)):
                    i+=1
                    d_capt = {"image_id": str(idx), "id": str(idx), "caption": c}
                    l_caption.append(d_capt)
                    image.save(im_folder+str(idx)+".jpg")

        if b == True and i > train_set_size -1:
            real_train_set_size = i
            print("train", i)
            with open(caption_file_train, 'w', encoding='utf8') as outfile:
                json.dump(l_caption, outfile, ensure_ascii=False)
            print(len(l_caption))
            b = False
            l_caption = []
            im_folder = im_folder_val
        if i > real_train_set_size + val_set_size -1:
            print("val", i)
            with open(caption_file_val, 'w', encoding='utf8') as outfile:
                json.dump(l_caption, outfile, ensure_ascii=False)
            exit()
    #print(i)
    #with open(caption_file, 'w') as outfile:
    #    json.dump(l_caption, outfile)
    #dict_keys(['image', 'image_url', 'embedding', 'metadata_url', 'original_height', 'original_width', 'mime_type', 'caption_attribution_description', 'wit_features'])


#train_or_val = "val"
dataset = "wit_en"
#dataset = "wit_small"

target_lang ='en'

parser = argparse.ArgumentParser()
parser.add_argument('--out_path')
parser.add_argument('--train_size', default = 83000)
parser.add_argument('--val_size', default = 20000)
parser.add_argument('--dataset_name')
parser.add_argument('--lang')

args = parser.parse_args()

dataset = args.dataset_name
train_set_size = args.train_size
val_set_size = args.val_size
target_lang = args.lang

caption_file_train = "data/"+dataset+"/annotations/train_caption.json"
caption_file_val = "data/"+dataset+"/annotations/val_caption.json"
im_folder_train = "data/"+dataset+"/train/"
im_folder_val = "data/"+dataset+"/val/"

generate_image_folder_and_annot_file(target_lang, train_set_size, val_set_size,im_folder_train, im_folder_val, caption_file_train, caption_file_val)