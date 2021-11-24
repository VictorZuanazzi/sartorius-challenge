import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence, Optional, List, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import (
    pil_to_tensor,
    InterpolationMode,
)
from yamldataclassconfig import YamlDataClassConfig, build_path


class Partition(Enum):
    Train = "train"
    Eval = "eval"
    Test = "test"


class SamplingStrategy(Enum):
    Uniform = "uniform"
    InverseClassFrequency = "inverse class frequency"
    SegmentationFrequency = "segmentation frequency"


class SartoriusClasses(IntEnum):
    astro = auto()
    cort = auto()
    shsy5y = auto()


class DataOutputs(Enum):
    Image = "image"
    Mask = "mask"
    Label = "label"

    @staticmethod
    def all():
        return set(*DataOutputs.__members__.values())


@dataclass
class DataloaderGlobalConfig(YamlDataClassConfig):
    num_workers: int


@dataclass
class DataloaderConfig(YamlDataClassConfig):
    batch_size: int
    p_random_transforms: float
    img_size: int
    num_samples: Optional[int] = None
    dataset_root: Path = field(
        default_factory=Path, metadata={"dataclasses_json": {"mm_field": Path}}
    )
    outputs: List = field(default_factory=list)  # typing List[Outputs] breaks =/

    def __post_init__(self):

        # get the absolute path ;)
        self.dataset_root: Path = build_path(
            path=self.dataset_root, path_is_absolute=True
        )

        # make outputs nicer =)
        if not self.outputs:
            self.outputs: List[DataOutputs] = DataOutputs.all()
        else:
            self.outputs: List[DataOutputs] = [
                DataOutputs[dt_out] for dt_out in self.outputs
            ]


class SatoriusDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: Path, partition: Partition, outputs: Sequence[DataOutputs] = None
    ):
        self.path_csv = root / "train.csv"
        self.partition = partition
        self.path_imgs = root / (
            partition.value if partition != Partition.Eval else Partition.Train.value
        )
        self.df_ann = self.load_csv()

        if outputs is None:
            if partition == Partition.Test:
                self.outputs = [DataOutputs.Image]
            else:
                self.outputs = DataOutputs.all()
        else:
            self.outputs = set(outputs)

    def load_csv(self):

        if self.partition == Partition.Test:
            ids = [id_.replace(".png", "") for id_ in os.listdir(self.path_imgs)]
            return pd.DataFrame(columns=["id"], data=ids)

        def running_pixels(ann_):
            start_end = [
                (start - 1, start - 1 + stroke)
                for start, stroke in zip(ann_[::2], ann_[1:][::2])
            ]
            return [pix for start, end in start_end for pix in range(start, end)]

        df_train = pd.read_csv(self.path_csv)
        df_ann = (
            df_train.groupby(["id", "cell_type"])["annotation"]
            .agg(list)
            .reset_index(drop=False)
        )
        df_ann["annotation"] = df_ann["annotation"].apply(
            lambda x: [int(item) for sublist in x for item in sublist.split()]
        )
        df_ann["pixels"] = df_ann["annotation"].apply(running_pixels)
        train_limit = len(df_ann) // 10

        if self.partition == Partition.Train:
            return df_ann[:-train_limit].reset_index(inplace=False)

        if self.partition == Partition.Eval:
            return df_ann[-train_limit:].reset_index(inplace=False)

    def get_sampling_weights(self, strategy):

        if (strategy == SamplingStrategy.Uniform) or (self.partition == Partition.Test):
            return [1 for __ in range(self.df_ann)]

        if strategy == SamplingStrategy.InverseClassFrequency:
            frequencies = (
                self.df_ann.groupby("cell_type")["cell_type"]
                .agg(lambda x: 1 / len(x))
                .to_dict()
            )
            return self.df_ann["cell_type"].apply(lambda x: frequencies[x]).to_list()

        if strategy == SamplingStrategy.SegmentationFrequency:
            return self.df_ann["pixels"].apply(lambda x: len(x)).to_list()

    def __len__(self):
        return len(self.df_ann)

    @staticmethod
    def get_mask(img, running_pixels):
        mask_flat = torch.zeros_like(img.view(-1))
        mask_flat[running_pixels] = 1
        return mask_flat.reshape(img.shape)

    @lru_cache(maxsize=1024)
    def __getitem__(self, index: int) -> Dict[DataOutputs, Tensor]:

        img_path = self.path_imgs / f"{self.df_ann.loc[index, 'id']}.png"
        img = pil_to_tensor(Image.open(img_path)) / 255.0

        output_data = {}
        if DataOutputs.Image in self.outputs:
            output_data[DataOutputs.Image] = img

        if DataOutputs.Mask in self.outputs:
            output_data[DataOutputs.Mask] = self.get_mask(
                img, self.df_ann.loc[index, "pixels"]
            )

        if DataOutputs.Label in self.outputs:
            output_data[DataOutputs.Label] = SartoriusClasses[
                self.df_ann.loc[index, "cell_type"]
            ].value

        return output_data


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        configuration_train: DataloaderConfig,
        configuration_eval: DataloaderConfig,
        configuration_test: DataloaderConfig,
    ):
        super().__init__()

        # passes configurations to self
        self.num_workers = num_workers
        self.config_train = configuration_train
        self.config_eval = configuration_eval
        self.config_test = configuration_test

        # Set list of dataset objects (prior to instanciation)
        self.datasets_train_not_init = [SatoriusDataset]
        self.datasets_eval_not_init = [SatoriusDataset]
        self.datasets_test_not_init = [SatoriusDataset]

        # placeholders for the instantiated datasets.
        self.dataset_train = None
        self.dataset_eval = None
        self.dataset_test = None

    @staticmethod
    def _make_dataset(datasets_not_init, batch_size, subset=None, **kwargs):
        datasets_ = []
        for dataset_ in datasets_not_init:
            datasets_.append(dataset_(**kwargs))

        if (subset is not None) and (subset > 0):
            idxs = [i for i in range(max(subset // len(datasets_not_init), batch_size))]
            datasets_ = [Subset(dataset_, idxs) for dataset_ in datasets_]

        return ConcatDataset(datasets_)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage in (None, "fit"):
            self.dataset_train = self._make_dataset(
                self.datasets_train_not_init,
                batch_size=self.config_train.batch_size,
                subset=self.config_train.num_samples,
                root=self.config_train.dataset_root,
                partition=Partition.Train,
                outputs=self.config_train.outputs,
            )

            self.dataset_eval = self._make_dataset(
                self.datasets_eval_not_init,
                batch_size=self.config_eval.batch_size,
                subset=self.config_eval.num_samples,
                root=self.config_train.dataset_root,
                partition=Partition.Eval,
                outputs=self.config_eval.outputs,
            )

        if stage in (None, "test"):
            self.dataset_test = self._make_dataset(
                self.datasets_eval_not_init,
                batch_size=self.config_test.batch_size,
                subset=self.config_test.num_samples,
                root=self.config_train.dataset_root,
                partition=Partition.Test,
                outputs=self.config_eval.outputs,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.config_train.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.dataset_eval,
            batch_size=self.config_eval.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.config_test.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        return loader

    @staticmethod
    def transform_identity():
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def transform_random_affine(p):
        if np.random.rand() < p:
            return transforms.RandomAffine(
                degrees=(0, 360),
                scale=None,
                shear=(-5, 5, -5, 5),
                interpolation=InterpolationMode.BILINEAR,
            )
        return nn.Identity()

    @staticmethod
    def transform_random_gaussian_blur(p):
        if np.random.rand() < p:
            return transforms.GaussianBlur(kernel_size=np.random.choice([3, 5, 7]))

        return transforms.Lambda(lambda x: x)

    def define_augmentations(self, p, img_size):

        if np.random.rand() > p:
            return transforms.RandomCrop(
                size=img_size, pad_if_needed=True, padding_mode="reflect"
            )

        augmentations = nn.Sequential(
            self.transform_random_affine(p),
            # self.transform_random_gaussian_blur(p),
            transforms.RandomHorizontalFlip(p=p),
            transforms.RandomVerticalFlip(p=p),
            # transforms.RandomAdjustSharpness(
            #     (np.random.rand() * 3) + 1, p=p
            # ),
            # transforms.RandomAutocontrast(p=p),
            transforms.RandomCrop(
                size=img_size, pad_if_needed=True, padding_mode="reflect",
            ),
        )
        return augmentations

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:

        if self.trainer.training and self.config_train.p_random_transforms:
            img_mask = torch.cat(
                (batch[DataOutputs.Image], batch[DataOutputs.Mask]), dim=1
            )
            img_mask = self.define_augmentations(
                p=self.config_train.p_random_transforms,
                img_size=self.config_train.img_size,
            )(img_mask)
            batch[DataOutputs.Image], batch[DataOutputs.Mask] = torch.split(
                img_mask, split_size_or_sections=1, dim=1
            )

        return batch

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    loader_train = DataLoader(
        SatoriusDataset(root=Path("./data/"), partition=Partition.Train),
        batch_size=4,
        num_workers=0,
    )
    loader_eval = DataLoader(
        SatoriusDataset(root=Path("./data/"), partition=Partition.Eval),
        batch_size=4,
        num_workers=0,
    )
    loader_test = DataLoader(
        SatoriusDataset(root=Path("./data/"), partition=Partition.Test),
        batch_size=3,
        num_workers=0,
    )

    batch_train = next(iter(loader_train))
    batch_eval = next(iter(loader_eval))
    batch_test = next(iter(loader_test))

    breakpoint()

    print("Done!")
