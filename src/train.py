from typing import Tuple

import clip
import hydra
import torch
import torch.utils.data as data
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ClipCap
from dataset import COCODataset
from utils import download_checkpoint


def get_splits(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    _, preprocess_vis = clip.load(cfg.clipcap.vision_model.name)
    train_set = COCODataset(cfg.data.data_dir, "train", transform=preprocess_vis)
    valid_set = COCODataset(cfg.data.data_dir, "val", transform=preprocess_vis)
    train_loader = DataLoader(data.Subset(train_set, torch.arange(1000)), batch_size=4, num_workers=4)
    valid_loader = DataLoader(data.Subset(valid_set, torch.arange(1000)), batch_size=4, num_workers=4)
    test_loader = DataLoader(valid_set, num_workers=4)
    return train_loader, valid_loader, test_loader


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = ClipCap(cfg.clipcap)
    #ckpt = download_checkpoint(cfg.project_name, cfg.experiment_name)
    ckpt = None

    logger = WandbLogger(
        project=cfg.project_name, log_model="all"
    )
    logger.watch(model)

    seed_everything(cfg.seed, workers=True)
    trainer = Trainer(max_epochs=5, logger=logger, deterministic=True)
    train_loader, valid_loader, test_loader = get_splits(cfg)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt)

if __name__ == "__main__":
    main()
