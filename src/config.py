from dataclasses import dataclass, field
from enum import Enum

import hydra
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError
from omegaconf import MISSING, DictConfig, OmegaConf
from pydantic import BaseModel, DirectoryPath, Field, validator, PositiveInt
from typing_extensions import Annotated


class VisionModelConfig(BaseModel):
    name: str
    image_size: PositiveInt
    output_dim: PositiveInt


class LanguageModelConfig(BaseModel):
    name: str
    output_dim: PositiveInt

    @validator("name")
    def must_be_huggingface_model(cls, v):
        try:
            info = model_info(v)
        except RepositoryNotFoundError:
            raise ValueError("Not a HuggingFace model")


class AdapterArchitecture(Enum):
    mlp = "mlp"
    transformer = "transformer"


class AdapterConfig(BaseModel):
    type: AdapterArchitecture
    hl1_dim: PositiveInt
    hl2_dim: PositiveInt


class CLIPCapConfig(BaseModel):
    vision_model: VisionModelConfig
    language_model: LanguageModelConfig
    adapter: AdapterConfig


class Language(Enum):
    english = "en"
    german = "de"
    french = "fr"


class Dataset(Enum):
    coco = "coco"
    multi30k = "multi30k"


class DataConfig(BaseModel):
    data_dir: DirectoryPath
    name: Dataset


class ExperimentConfig(BaseModel):
    clipcap: CLIPCapConfig
    data: DataConfig
    batch_size: PositiveInt
    epochs: PositiveInt
    language: Language
    lr: Annotated[float, Field(gt=0, lt=1)]
    project_name: str
    experiment_name: str


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.experiment_name)
    print(OmegaConf.to_yaml(cfg))

    OmegaConf.resolve(cfg)
    r_model = ExperimentConfig(**cfg)


if __name__ == "__main__":
    main()
