#from datasets import load_metric
import json
from tqdm import tqdm
import evaluate

from datasets import load_metric
import argparse 
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pdb




def convert_to_dict(l_pred, l_annot):
    d_pred = {}
    d_annot = {}
    for elem in tqdm(l_pred):
        item = elem["image_id"]
        d_pred[int(item)] = elem["caption"]
    for elem in tqdm(l_annot):
        item = elem["image_id"]
        if item in d_annot.keys():
            d_annot[int(item)].append(elem["caption"])
        else:
            d_annot[int(item)] = [elem["caption"]]

    return d_pred, d_annot 

def keys_mismatch (d1, d2):
    k1 = d1.keys()
    k2 = d2.keys()
    l_1 = [ k in d2 for k in k1]
    l_2 = [k in d1 for k in k2]
    print(l_1.count(False), l_2.count(False))
    

def print_some_capt(d1,d2):
    k = d1.keys()
    for elem in k:
        print("")
        print(d1[elem])
        print(d2[elem])

def convert_to_list_for_rouge (path,d_annot, d_pred,checkpoint):
    l_annot_r = []
    l_pred_r = []
    annot_keys = d_annot.keys()
    for k in tqdm(d_pred.keys()):
        if k in annot_keys:
            l_annot_r.append(d_annot[k])
            l_pred_r.append(d_pred[k])
    
    
    with open (path+"/l_pred_r_val_"+checkpoint+".json", "w",encoding='utf8') as f:
        json.dump(l_pred_r,f, ensure_ascii=False)

    with open(path+"/l_annot_r_val_"+checkpoint+".json", 'w',encoding='utf8') as outfile:
        json.dump(l_annot_r, outfile, ensure_ascii=False)
    return l_annot_r, l_pred_r

def convert_annot_to_spice_format(annot_file,out_path):
    with open(annot_file, "r") as f:
        annot = json.load(f)

    annotation_file_spice = '../pycocoevalcap/example/captions_val2014.json'
    with open(annotation_file_spice, "r") as f:
        annot_spice = json.load(f)

    annot_spice["annotations"] = annot
    with open(out_path+"annot_val_spice_format.json", 'w') as outfile:
        json.dump(annot_spice, outfile)
    return out_path+"annot_val_spice_format.json"




def compute_score(path,predict_file,metric,dataset,checkpoint):
    
    predict_path = path+predict_file
    annot_path = "data/"+dataset+"/annotations/"
    annot_file = annot_path+"val_caption.json"
    
    with open(predict_path, "r") as f:
        l_predict = json.load(f)
    with open(annot_file, "r") as f:
        l_annot = json.load(f)

    if metric == "spice":
        annot_spice_file = convert_annot_to_spice_format(annot_file,annot_path)

        coco = COCO(annot_spice_file)
        coco_result = coco.loadRes(predict_path)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
    if metric == "rouge": 
        rouge = evaluate.load('rouge')
        d_pred, d_annot = convert_to_dict(l_predict, l_annot)
        #print(d_pred)
        l_annot_r, l_pred_r = convert_to_list_for_rouge (path,d_annot, d_pred,checkpoint)
        results = rouge.compute(predictions=l_pred_r, references=l_annot_r)
        print(results)

parser = argparse.ArgumentParser()
parser.add_argument('--path')
parser.add_argument('--checkpoint')
parser.add_argument('--dataset_name')
parser.add_argument('--metric')

args = parser.parse_args()

checkpoint = args.checkpoint
path = args.path
predicted_capt = path+"/predicted_captions_val_"+checkpoint+".json"
metric = args.metric
dataset_name = args.dataset_name

compute_score(path,predicted_capt,metric,dataset_name,checkpoint)



