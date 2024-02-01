import json

import pandas as pd
from pandas.core.common import flatten
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm.rich import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

caption_model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
# use gpt 2 for text model
text_model = AutoModelForCausalLM.from_pretrained("gpt2")
text_model_tuned = caption_model.decoder
feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

class COCODataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["train", "val"]
        self.transform = transform
        with open(Path(self.data_dir) / "annotations" / "train_caption.json", "r") as f:
            captions = json.load(f)
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            path = (
                self.data_dir
                / f"{self.split}2014"
                / f"COCO_{self.split}2014_{int(cap['image_id']):012d}.jpg"
            )
            if path.is_file():
                data.append(cap)
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"])
        path = (
            self.data_dir
            / f"{self.split}2014"
            / f"COCO_{self.split}2014_{img_id:012d}.jpg"
        )
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, caption['caption'], img_id


def batch_detok(batch):
    weird = "Ġ"
    detok = []
    batch_words = []
    for ids in batch:
        sent_detok = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        words = [] 
        word = ""
        count = 0
        for tok in sent_detok:
            if tok.startswith(weird):
                words += count * [word]
                word = tok[1:]
                count = 1
            else:
                word += tok
                count += 1
        words += count * [word]
        batch_words.append(words)
        detok.append(sent_detok)
    return detok, batch_words


def predict_step(images, captions, image_paths):

    pixel_values = images.squeeze().to(device)

    tokenized = tokenizer(
        captions, padding="max_length", truncation=True, return_tensors="pt"
    )

    # the conditional probability of the caption given the image
    print(pixel_values.shape)
    loss_caption = caption_model(
        pixel_values=pixel_values,
        labels=tokenized.input_ids,
        decoder_attention_mask=tokenized.attention_mask,
    )
    loss_text = text_model(
        **tokenized,
        labels=tokenized.input_ids,
    )
    loss_text_tuned = text_model_tuned(
        **tokenized,
        labels=tokenized.input_ids,
    )
    batch_size = loss_caption.logits.shape[0]
    text_loss = []
    caption_loss = []
    prior_by_tok = []
    prior_tuned_by_tok = []
    cond_by_tok = []
    text_tuned_loss = []
    for sentence in range(batch_size):
        mask = tokenized.attention_mask[sentence] == 1
        cond_H = torch.nn.functional.cross_entropy(
            loss_caption.logits[sentence][mask], tokenized.input_ids[sentence][mask], reduction="none"
        )
        prior_H = torch.nn.functional.cross_entropy(
            loss_text.logits[sentence][mask], tokenized.input_ids[sentence][mask], reduction="none"
        )
        prior_tuned_H = torch.nn.functional.cross_entropy(
            loss_text_tuned.logits[sentence][mask], tokenized.input_ids[sentence][mask], reduction="none"
        )
        
        prior_by_tok.extend(prior_H.detach().cpu().tolist())
        prior_tuned_by_tok.extend(prior_tuned_H.detach().cpu().tolist())
        cond_by_tok.extend(cond_H.detach().cpu().tolist())
        text_loss.append(prior_H.mean().item())
        text_tuned_loss.append(prior_tuned_H.mean().item())
        caption_loss.append(cond_H.mean().item())
    text_loss = np.array(text_loss)
    caption_loss = np.array(caption_loss)
    # the prior probability of the caption
    toks, words = batch_detok(tokenized.input_ids)
    pixel_values = pixel_values.to(device)

    index = [len(sent) * [i] for i, sent in enumerate(toks)]

    return (
        {
            "caption": captions,
            "mutual_information": text_loss - caption_loss,
            "image": image_paths,
            "prior": text_loss,
            "conditional": caption_loss,
            "tuned": text_tuned_loss
        },
        {
            "token": list(flatten(toks)),
            "prior": prior_by_tok,
            "conditional": cond_by_tok,
            "word": list(flatten(words)),
            "sentence": list(flatten(index)),
            "mutual_information": np.array(prior_by_tok) - np.array(cond_by_tok),
            "tuned": prior_tuned_by_tok
        }
    )


captions = [
    "A test sentence.",
    "A test sentence with indubitably obscure verbage.",
    "Two dogs and a cat.",
    "Two cats and a dog."
]
image_paths = [
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
]
def make_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert(mode="RGB")
    return img

# images = [make_image(path) for path in image_paths]

# results_sent, results_tok = [pd.DataFrame(x) for x in predict_step(images, captions, image_paths)]

# with open('test_sent.csv', 'w') as f:
#     results_sent.to_csv(f)
# with open('test_tok.csv', 'w') as f:
#     results_tok.to_csv(f)
if __name__ == "__main__":
    val_dataloader = DataLoader(
            COCODataset("data/coco", "val", transform=lambda x: feature_extractor(x, return_tensors="pt")),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    full_results_sent = []
    full_results_tok = [] 
    for batch in tqdm(val_dataloader):
        images, captions, image_paths = batch
        results_sent, results_tok = [pd.DataFrame(x) for x in predict_step(images['pixel_values'], captions, image_paths)]
        full_results_sent.append(results_sent)
        full_results_tok.append(results_tok)

    results_sent2 = pd.concat(full_results_sent, ignore_index=True)
    results_tok2 = pd.concat(full_results_tok, ignore_index=True)
    with open('results_sent.csv', 'w') as f:
        results_sent2.to_csv(f)
    with open('results_tok.csv', 'w') as f:
        results_tok2.to_csv(f)


# print(tokenizer.convert_ids_to_tokens(tokenized["input_ids"][1], skip_special_tokens=True))
