#from datasets import load_metric
import json
from tqdm import tqdm
#import evaluate

from datasets import load_metric

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pdb


path = "../coco_train_nina_modif_loss/"
checkpoint = "002"
metric = "spice"




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




def compute_score(predict_folder,predict_file,metric,train_or_val):
    if metric == "rouge":
        #keys_mismatch(d_pred,d_annot)
        #print_some_capt(d_annot,d_pred)

        #convert_to_list_for_rouge(path,d_pred, d_annot)
        met = load_metric("./rouge")
        annotation_file = path+'l_pred_r_'+val_or_train+'_'+checkpoint+'.json'
        #predict_file = path+"l_annot_r_"+val_or_train+"_"+checkpoint+".json"

        with open(predict_folder+predict_file, "r") as f:
            predictions = json.load(f)
        with open(annotation_file, "r") as f:
            references = json.load(f)
        #pred = [elem[0] for elem in predictions]

        #references = [["The quick brown fox jumps over the lazy dog","The quick brown fox jumps over the lazy dog"],["hey"]]
        #predictions = ["The quick brown fox jumps over the lazy dog.","hey"]

        result = met.compute(predictions=predictions, references=references, language="en")

        result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}
        print(f"Language: English, result: {result}")

        results = rouge.compute(predictions=predictions, references=references)
        print(results)
    if metric == "spice":
        
        #results_file = '../pycocoevalcap/example/captions_val2014_fakecap_results.json'
        predicted_capt = predict_folder + predict_file

        annot_path = "data/coco/annotations/"
        annot_file = annot_path+"annotation_val_propre.json"
        annot_spice_file = convert_annot_to_spice_format(annot_file,annot_path)

        #with open(annot_spice_file, "r") as f:
        #    annot = json.load(f)
        #pdb.set_trace()

        

        # create coco object and coco_result object
        #coco = COCO(annotation_file)
        coco = COCO(annot_spice_file)
        #print(coco)
        #
        coco_result = coco.loadRes(predicted_capt)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)

        # evaluate on a subset of images by setting
        # coco_eval.params['image_id'] = coco_result.getImgIds()
        # please remove this line when evaluating the full validation set
        coco_eval.params['image_id'] = coco_result.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')


compute_score("coco_train_nina_modif_loss2/","predicted_captions_val_002.json","spice","val")

exit()


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





