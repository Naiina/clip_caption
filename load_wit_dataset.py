


from datasets import load_dataset
from torchvision import transforms
from PIL import Image  
import PIL  
import json
from torch.utils.data import DataLoader
import imghdr
from tqdm import tqdm
import argparse 
import os
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 10485760  # this is the current value

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

def generate_image_folder_and_annot_file(target_lang, train_set_size, val_set_size, im_folder_train, im_folder_val,caption_file_train, caption_file_val):

    data = load_dataset("wikimedia/wit_base", split="train", streaming=True)
    i = 0
    j = 0
    l_caption = []
    im_folder = im_folder_train
    print("hey")
    b = True
    real_train_set_size = train_set_size
    save_folder = caption_file_train
    


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
                    j+=1
                    d_capt = {"image_id": str(idx), "id": str(idx), "caption": c}
                    l_caption.append(d_capt)
                    image.save(im_folder+str(idx)+".jpg")
        if j>10000:
            with open(save_folder, 'w', encoding='utf8') as outfile:
                json.dump(l_caption, outfile, ensure_ascii=False)
            j = 0

        if b == True and i > train_set_size -1:
            real_train_set_size = i
            print("train", i)
            with open(save_folder, 'w', encoding='utf8') as outfile:
                json.dump(l_caption, outfile, ensure_ascii=False)
            print(len(l_caption))
            b = False
            l_caption = []
            im_folder = im_folder_val
            save_folder = caption_file_val
        if i > real_train_set_size + val_set_size -1:
            print("val", i)
            with open(save_folder, 'w', encoding='utf8') as outfile:
                json.dump(l_caption, outfile, ensure_ascii=False)
            exit()
    #print(i)
    #with open(caption_file, 'w') as outfile:
    #    json.dump(l_caption, outfile)
    #dict_keys(['image', 'image_url', 'embedding', 'metadata_url', 'original_height', 'original_width', 'mime_type', 'caption_attribution_description', 'wit_features'])




parser = argparse.ArgumentParser()
parser.add_argument('--train_size', default = 566852)
parser.add_argument('--val_size', default = 202700)
parser.add_argument('--outpath')
parser.add_argument('--lang')

args = parser.parse_args()

out_path = args.outpath
train_set_size = int(args.train_size)
val_set_size = int(args.val_size)
target_lang = args.lang

caption_file_train = "data/"+out_path+"/annotations/train_caption.json"
caption_file_val = "data/"+out_path+"/annotations/val_caption.json"
im_folder_train = "data/"+out_path+"/train/"
im_folder_val = "data/"+out_path+"/val/"
if not os.path.exists("data/"+out_path):
    os.mkdir("data/"+out_path)
    os.mkdir("data/"+out_path+"/annotations")
    os.mkdir(im_folder_train)
    os.mkdir(im_folder_val)
generate_image_folder_and_annot_file(target_lang, train_set_size, val_set_size,im_folder_train, im_folder_val, caption_file_train, caption_file_val)

