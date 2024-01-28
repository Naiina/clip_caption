import pandas as pd
from pandas.core.common import flatten
import numpy as np
import torch
from PIL import Image
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

feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

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


def predict_step(image_paths, captions):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

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
    batch_size = loss_caption.logits.shape[0]
    text_loss = []
    caption_loss = []
    prior_by_tok = []
    cond_by_tok = []
    for sentence in range(batch_size):
        mask = tokenized.attention_mask[sentence] == 1
        cond_H = torch.nn.functional.cross_entropy(
            loss_caption.logits[sentence][mask], tokenized.input_ids[sentence][mask], reduction="none"
        )
        prior_H = torch.nn.functional.cross_entropy(
            loss_text.logits[sentence][mask], tokenized.input_ids[sentence][mask], reduction="none"
        )
        
        prior_by_tok.extend(prior_H.detach().cpu().tolist())
        cond_by_tok.extend(cond_H.detach().cpu().tolist())
        text_loss.append(prior_H.mean().item())
        caption_loss.append(cond_H.mean().item())
    text_loss = np.array(text_loss)
    caption_loss = np.array(caption_loss)
    # the prior probability of the caption
    toks, words = batch_detok(tokenized.input_ids)

    index = [len(sent) * [i] for i, sent in enumerate(toks)]

    return (
        {
            "caption": captions,
            "mutual_information": text_loss - caption_loss,
            "image": image_paths,
            "prior": text_loss,
            "conditional": caption_loss,
        },
        {
            "token": list(flatten(toks)),
            "prior": prior_by_tok,
            "conditional": cond_by_tok,
            "word": list(flatten(words)),
            "sentence": list(flatten(index)),
            "mutual_information": np.array(prior_by_tok) - np.array(cond_by_tok),
        }
    )


captions = [
    "A test sentence.",
    "A test sentence with indubitably obscure verbage.",
    "Two dogs and a cat.",
    "Two cats and a dog."
]
images = [
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
    "istockphoto-1251352680-612x612.jpg",
]

results_sent, results_tok = [pd.DataFrame(x) for x in predict_step(images, captions)]
with open('results_sent.csv', 'w') as f:
    results_sent.to_csv(f)
with open('results_tok.csv', 'w') as f:
    results_tok.to_csv(f)
# print(tokenizer.convert_ids_to_tokens(tokenized["input_ids"][1], skip_special_tokens=True))
