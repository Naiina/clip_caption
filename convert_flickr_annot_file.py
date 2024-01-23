import json
l = []
#i = 0
with open('./data/flickr/annotations/valid-de_gmt.jsonl', 'r') as f:        
        for line in f:
            #i=i+1
            d=json.loads(line)
            caption = d["sentences"][0]
            id = d["id"]
            new_d = {"image_id": id, "id": id, "caption": caption}
            l.append(new_d)

with open("data/flickr/annotations/val_caption.json", 'w', encoding='utf8') as outfile:
    json.dump(l, outfile, ensure_ascii=False)


#{"image_id": "1", "id": "1", "caption": "Mausolé Moulay Ali Cherif ancêtre de la Dynastie Alaouite."}