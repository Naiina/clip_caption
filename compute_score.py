#from datasets import load_metric
import json
from tqdm import tqdm
#import evaluate

from datasets import load_metric

metric = load_metric("./rouge")

CLIP_prefix_caption/s2


path = "../coco_train_nina_modif_los/"
checkpoint = "002"


def convert(l_pred, l_annot):
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
    #for i,elem in enumerate(k1):
    #    if i<10:
    #        print(type(elem))
    #for i,elem in enumerate(k2):
    #    if i<10:
    #        print(type(elem))
        
    l_1 = [ k in d2 for k in k1]
    l_2 = [k in d1 for k in k2]
    print(l_1.count(False), l_2.count(False))
    

def print_some_capt(d1,d2):
    k = d1.keys()
    for elem in k:
        print("")
        print(d1[elem])
        print(d2[elem])

def convert_to_list_for_rouge (path,d_annot, d_pred):
    l_annot_r = []
    l_pred_r = []
    annot_keys = d_annot.keys()
    for k in tqdm(d_pred.keys()):
        if k in annot_keys:
            l_annot_r.append(d_annot[k])
            l_pred_r.append(d_pred[k])
    
    
    with open (path+"l_pred_r_"+val_or_train+"_"+checkpoint+".json", "w") as f:
        json.dump(l_pred_r,f)

    with open(path+"l_annot_r_"+val_or_train+"_"+checkpoint+".json", 'w') as outfile:
        json.dump(l_annot_r, outfile)
    

if val_or_train == "val":
    predict_file = path + 'predicted_captions_val_'+checkpoint+'.json'
    annotation_file = "../data/coco/annotations/annotation_val_propre.json"
else:
    predict_file = path + 'predicted_captions_train_'+checkpoint+'.json'
    annotation_file = "../data/coco/annotations/train_caption.json"
with open(predict_file, "r") as f:
    l_predicted = json.load(f)

with open(annotation_file, "r") as f:
    l_annot = json.load(f)

d_pred, d_annot  = convert(l_predicted, l_annot)
keys_mismatch(d_pred,d_annot)
#print_some_capt(d_annot,d_pred)

convert_to_list_for_rouge(path,d_pred, d_annot)











#rouge = evaluate.load('rouge')


annotation_file = path+'l_pred_r_'+val_or_train+'_'+checkpoint+'.json'
predict_file = path+"l_annot_r_"+val_or_train+"_"+checkpoint+".json"

with open(predict_file, "r") as f:
    predictions = json.load(f)
with open(annotation_file, "r") as f:
    references = json.load(f)
#pred = [elem[0] for elem in predictions]

#references = [["The quick brown fox jumps over the lazy dog","The quick brown fox jumps over the lazy dog"],["hey"]]
#predictions = ["The quick brown fox jumps over the lazy dog.","hey"]

result = metric.compute(predictions=predictions, references=references, language="en")

result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}
print(f"Language: English, result: {result}")

exit()


results = rouge.compute(predictions=predictions, references=references)
print(results)

exit()
