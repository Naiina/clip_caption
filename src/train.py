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
from utils import download_checkpoint, ImageCaptionLogger


def get_splits(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    _, preprocess_vis = clip.load(cfg.clipcap.vision_model.name)
    train_set = COCODataset(cfg.data.data_dir, "train", transform=preprocess_vis)
    valid_set = COCODataset(cfg.data.data_dir, "val", transform=preprocess_vis)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(valid_set, num_workers=cfg.num_workers)
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
    train_loader, valid_loader, test_loader = get_splits(cfg)
    sample = next(iter(valid_loader))
    trainer = Trainer(max_epochs=cfg.epochs, logger=logger, deterministic=True, callbacks=[ImageCaptionLogger(sample)])

    trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt)

if __name__ == "__main__":
    main()
