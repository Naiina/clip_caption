import json
from tqdm import tqdm

val_or_train = "val"
path = "coco_train_prefix10/"
checkpoint = "009"


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
    
    if val_or_train == "val":
        with open (path+"l_pred_r_val_"+checkpoint+".json", "w") as f:
            json.dump(l_pred_r,f)

        with open(path+"l_annot_r_val_"+checkpoint+".json", 'w') as outfile:
            json.dump(l_annot_r, outfile)
    
if __name__ == '__main__':
    if val_or_train == "val":
        predict_file = path + 'predicted_captions_val_'+checkpoint+'.json'
        annotation_file = "data/coco/annotations/annotation_val_propre.json"
    with open(predict_file, "r") as f:
        l_predicted = json.load(f)

    with open(annotation_file, "r") as f:
        l_annot = json.load(f)

    d_pred, d_annot  = convert(l_predicted, l_annot)
    keys_mismatch(d_pred,d_annot)
    #print_some_capt(d_annot,d_pred)

    convert_to_list_for_rouge(path,d_pred, d_annot)





