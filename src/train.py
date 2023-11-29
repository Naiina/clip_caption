from typing import Tuple

import hydra
import lightning as L
import torch
import torch.utils.data as data
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ClipCap
from dataset import COCODataset
from utils import download_checkpoint


def get_splits(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set = COCODataset(cfg, "train")
    valid_set = COCODataset(cfg, "val")
    train_loader = DataLoader(data.Subset(train_set, torch.arange(1000)), num_workers=4)
    valid_loader = DataLoader(data.Subset(valid_set, torch.arange(1000)), num_workers=4)
    test_loader = DataLoader(valid_set, num_workers=4)
    return train_loader, valid_loader, test_loader


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = ClipCap(cfg)
    ckpt = download_checkpoint(cfg.project_name, cfg.experiment_name)

    logger = WandbLogger(
        project=cfg.project_name, log_model="all", id=cfg.experiment_name
    )
    logger.watch(model)

    trainer = L.Trainer(max_epochs=5, logger=logger)
    train_loader, valid_loader, test_loader = get_splits(cfg.data)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt)

if __name__ == "__main__":
    main()
