data/coco/annotations/train_caption.json:
	mkdir -p data/coco/annotations/
	gdown --fuzzy https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing -O data/coco/annotations/train_caption.json

data/coco/train2014/:
	mkdir -p data/coco/
	cd data/coco/ && wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip

data/coco/val2014/:
	mkdir -p data/coco/
	cd data/coco/ && wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip

data/coco/: data/coco/train2014/ data/coco/val2014/ data/coco/annotations/train_caption.json
data/coco/clip/: src/preprocess/coco.py data/coco/annotations/train_caption.json
results/: src/train.py data/coco/clip/
	python src/train.py
