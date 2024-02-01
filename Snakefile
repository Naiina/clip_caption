rule multi30k_raw:
    output:
        "data/multi30k/raw/{split}.{lang}",
    shell:
        """
        wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{wildcards.split}.{wildcards.lang}.gz &&
        gunzip {wildcards.split}.{wildcards.lang}.gz &&
        mv {wildcards.split}.{wildcards.lang} {output}
        """


rule multi30k_splits:
    output:
        "data/multi30k/image_splits/{split}.txt",
    shell:
        "curl https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/{wildcards.split}.txt -o {output}"


rule multi30k_annotations:
    input:
        "data/multi30k/raw/{split}.{lang}",
        "data/multi30k/image_splits/{split}.txt",
    output:
        "data/multi30k/annotations/{split}_captions_{lang}.json",
    shell:
        "python scripts/load_multi30k.py --train_or_val {wildcards.split} --lang {wildcards.lang}"


rule multi30k:
    input:
        expand(
            "data/multi30k/annotations/{split}_captions_{lang}.json",
            split=["train", "val"],
            lang=["fr"],
        ),


rule multi30k_embedding:
    input:
        "data/multi30k/annotations/{split}_captions_{lang}.json",
        "data/multi30k/image_splits/{split}.txt",
    output:
        "data/multi30k/embedding/{split}_{lang}_{vision}.pkl",  
    shell:
        "python scripts/multi30k_embedding.py --train_or_val {wildcards.split} --lang {wildcards.lang} --dataset multi30k"

rule embedding:
    input:
        expand(
            "data/multi30k/embedding/{split}_{lang}_{vision}.pkl",
            split=["train", "val"],
            lang=["fr"],
            vision=["ViT-B.32"],
        )

#rule coco_annotations:
#    output:
#        "data/coco/annotations/"

rule coco_images:
    output:
        "data/coco/images/"
