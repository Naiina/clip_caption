import lightning as L
import clip
import evaluate
import torch
import wandb
from torch import nn
from transfomers import AutoModel, AutoTokenizer

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    return module

class ClipCap(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vision_model, self.preprocess_vis = clip.load(cfg.vision_model.name)
        freeze(self.vision_model)
        self.language_model = freeze(AutoModel.from_pretrained(cfg.language_model.name))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.language_model.name)
        self.projector = MLP((cfg.vision_model.dim, (cfg.language_model.dim * cfg.prefix_length) // 2, cfg.language_model.dim * cfg.prefix_length))
        self.prefix_length = cfg.prefix_length
        self.rouge = evaluate.load('rouge')

    def forward(self, img, tokens):
        batch_size = img.shape[0]
        out_clip = self.vision_model.encode_image(img)
        out_mlp = self.mlp(out_clip).view(self.prefix_length, batch_size, -1)
        caption_emb = self.language_model.get_imput_embeddings(tokens.input_ids)
        total_emb = torch.cat((out_mlp, caption_emb), dim=0)
        labels = torch.cat(
            (torch.full((batch_size, self.prefix_length), -100), tokens), dim=1
        )
        return self.language_model(inputs_embeds=total_emb, labels=labels)

    def training_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")
            wandb.define_metric("val_loss", summary="min")
            wandb.define_metric("train_loss", summary="min")
            # wandb.define_metric("test_acc", summary="max")
            # wandb.define_metric("test_loss", summary="min")
        img, caption = batch
        img = self.preprocess_vis(img)
        tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        result = self(img, tokens)
        self.log("train_loss", result.loss)
        return result.loss

    def validation_step(self, batch, batch_idx):
        img, caption = batch
        img = self.preprocess_vis(img)
        tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        result = self(img, tokens)
        self.log("val_loss", result.loss)
        preds = torch.argmax(result.logits, dim=2)
        preds = self.tokenizer.batch_decode(preds)
        refs = self.tokenizer.batch_decode(tokens.input_ids)
        rouge = self.rouge.compute(preds, refs)
        self.log("val_accuracy", rouge["rouge2"], on_epoch=True)
        return result.loss

    def test_step(self, batch, batch_idx):
        img, caption = batch
        img = self.preprocess_vis(img)
        tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        result = self(img, tokens)
        self.log("val_loss", result.loss)
        preds = torch.argmax(result.logits, dim=2)
        preds = self.tokenizer.batch_decode(preds)
        refs = self.tokenizer.batch_decode(tokens.input_ids)
        rouge = self.rouge.compute(preds, refs)
        self.log("val_accuracy", rouge["rouge2"], on_epoch=True)
        self.log("val_loss", result.loss)
        return result.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=2e-5)
        return optimizer
