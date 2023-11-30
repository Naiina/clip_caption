import logging
import os
from pathlib import Path

import wandb
from huggingface_hub import create_repo, hf_hub_download, login
from lightning.pytorch import Callback


def get_or_create_repo(cfg) -> str:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    logging.info(
        f"Logged in to HuggingFace Hub, creating repo {cfg.repo_name} if needed"
    )
    repo_path = cfg.hf_user + "/" + cfg.repo_name
    repo_url = create_repo(repo_path, repo_type="model", exist_ok=True)
    print(repo_url)
    return repo_url


class ImageCaptionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_caps = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_caps = self.val_caps[:num_samples]

    def on_train_start(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_toks = pl_module.tokenizer(
            self.val_caps, return_tensors="pt", padding=True, truncation=True
        )
        val_toks = val_toks.to(device=pl_module.device)
        outputs = pl_module(val_imgs, val_toks)
        preds = outputs.logits.argmax(-1)
        preds = pl_module.tokenizer.batch_decode(preds)
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(img, caption=f"Pred: {pred}, Target: {target}")
                    for img, pred, target in zip(val_imgs, preds, self.val_caps)
                ],
                "global_step": trainer.global_step,
            }
        )


def download_checkpoint_hf(repo_url: str, cfg):
    Path("checkpoints", cfg.experiment_name).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints", cfg.experiment_name)
    if cfg.from_checkpoint is not None:
        ckpt_path = hf_hub_download(
            repo_id=cfg.repo_name,
            cache_dir=ckpt_path,
            resume_download=True,
            filename=cfg.from_checkpoint,
            repo_type="model",
        )
    else:
        try:
            ckpt_path = hf_hub_download(
                repo_id=cfg.repo_name,
                cache_dir=ckpt_path,
                filename="latest.ckpt",
                resume_download=True,
                repo_type="model",
            )
        except:
            return None
    return ckpt_path


# TODO: wandb api key in docker
# TODO:
def download_checkpoint(project_name: str, experiment_name: str) -> str | None:
    wandb_api = wandb.Api()
    try:
        artifact = wandb_api.artifact(
            f"{project_name}/model-{experiment_name}:latest", type="model"
        )
        artifact_dir = artifact.download()
        return str(Path(artifact_dir) / "model.ckpt")
    except wandb.errors.CommError:
        return None
