import logging
import os
from pathlib import Path

import wandb
from huggingface_hub import create_repo, hf_hub_download, login


def get_or_create_repo(cfg) -> str:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    logging.info(
        f"Logged in to HuggingFace Hub, creating repo {cfg.repo_name} if needed"
    )
    repo_path = cfg.hf_user + "/" + cfg.repo_name
    repo_url = create_repo(repo_path, repo_type="model", exist_ok=True)
    print(repo_url)
    return repo_url


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
