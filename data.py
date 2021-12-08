import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from yamldataclassconfig import YamlDataClassConfig, build_path

from generate_submissionfile import tensor2submission


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


class SatoriusDataset(Dataset):
    def __init__(
        self, root: Path, partition: Partition, outputs: Sequence[DataOutputs] = None
    ) -> None:


        """Satorius dataset as given by the challenger.

        More info: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data

        Args:
            root: root to the dataset. It is expected to containing sub-folders with test and training data and the
                annotations in a `train.csv` file.
            partition: train / eval or test partition.
            outputs: which outputs to load and return in __get_item__(). If None is give, then it defaults to all
                available outputs.
        """
        self.path_csv = root / "train.csv"
        self.partition = partition
        self.path_imgs = root / (
            partition.value if partition != Partition.Eval else Partition.Train.value
        )

        # load annotations
        self.df_ann = self.load_csv()

        # Get outputs
        if outputs is None:
            if partition == Partition.Test:
                self.outputs = [DataOutputs.Image]
            else:
                self.outputs = DataOutputs.all()
        else:
            self.outputs = set(outputs)

    def load_csv(self) -> pd.DataFrame:

        # The test partition has no annotation data!
        if self.partition == Partition.Test:
            ids = [id_.replace(".png", "") for id_ in os.listdir(self.path_imgs)]
            return pd.DataFrame(columns=["id"], data=ids)

        # Eval and Training data are in the same csv.
        df_train = pd.read_csv(self.path_csv)

        # Convert their weird annotation format into something useful.
        # 1. group all the annotations relevant to the same image.
        df_ann = df_train.groupby(["id", "cell_type"])["annotation"].agg(list).reset_index(drop=False)

        # 2. Convert the string annotations into integers.
        df_ann["annotation"] = df_ann["annotation"].apply(lambda x: [int(item) for item in ' '.join(x).split()])

        # 3. Convert their 'running pixel' into sequences of pixels in a flat image.
        def running_pixels(ann_):
            start_end = [
                (start - 1, start - 1 + stroke)
                for start, stroke in zip(ann_[::2], ann_[1:][::2])
            ]

            return list(set(pix for start, end in start_end for pix in range(start, end)))

        df_ann["pixels"] = df_ann["annotation"].apply(running_pixels)

        self.test_consistency(df_ann, running_pixels)  # only uncomment when unsure

        # Not so random split of train and eval
        train_limit = len(df_ann) // 10

        if self.partition == Partition.Train:
            return df_ann[:-train_limit].reset_index(inplace=False)

        if self.partition == Partition.Eval:
            return df_ann[-train_limit:].reset_index(inplace=False)

    def test_consistency(self, df_ann, running_pixels, test_size=50):
        """
        Makes sure that we can recreate the original data using our data transformation methods
        Args:
            df_ann:
            running_pixels:

        Returns:

        """
        for test_n in range(test_size):

            print(f"{test_n=}")
            index = random.randint(0, len(df_ann)-1)

            # Load image
            img_path = self.path_imgs / f"{df_ann.loc[index, 'id']}.png"
            img = pil_to_tensor(Image.open(img_path)) / 255.0

            mask = self.get_mask(
                img, df_ann.loc[index, "pixels"]
            )

            reconstructed_annotation = tensor2submission(mask, start_index=1)

            reconstructed_pixels = sorted(running_pixels([int(x) for x in reconstructed_annotation.split()]))
            original_pixels = sorted(df_ann.loc[index, "pixels"])
            assert (reconstructed_pixels == original_pixels), f"{img_path} proved not identical"
        print("Consistency tested")

    def get_sampling_weights(self, strategy: SamplingStrategy) -> List[float]:
        """Different sampling strategies.

        Args:
            strategy: the strategy to be used.

        Returns:
            a list with the sampling weights of each data sample.
        """

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
    def get_mask(img: Tensor, running_pixels: List[int]) -> Tensor:
        """Get a mask from an image and the flat pixel sequence.

        Args:
            img: the image for the mask.
            running_pixels: Flat list with the pixel indexes of a flat image.

        Returns:
            A binary mask with the same shape of the input image.
        """

        mask_flat = torch.zeros_like(img.view(-1))
        mask_flat[running_pixels] = 1

        return mask_flat.reshape(img.shape)

    @lru_cache(maxsize=1024)
    def __getitem__(self, index: int) -> Dict[DataOutputs, Tensor]:
        """Fetch one item."""

        # Load image
        img_path = self.path_imgs / f"{self.df_ann.loc[index, 'id']}.png"
        img = pil_to_tensor(Image.open(img_path)) / 255.0

        # Selects the outputs
        output_data = {}
        if DataOutputs.Image in self.outputs:
            # Add image to the output.
            output_data[DataOutputs.Image] = img

        if DataOutputs.Mask in self.outputs:
            # Add mask to the output.
            output_data[DataOutputs.Mask] = self.get_mask(
                img, self.df_ann.loc[index, "pixels"]
            )

        if DataOutputs.Label in self.outputs:
            # add the global label of the image to the output.
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
        """Constructor of the data module."""

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
    def _make_dataset(
        datasets_not_init: List[Dataset],
        batch_size: int,
        subset: Optional[int] = None,
        **kwargs,
    ):
        """Initialize multiple datasets and concatenate them into one dataset.

        Args:
            datasets_not_init: List of not initialized dataset classes.
            batch_size: the batch size.
            subset: the number of data points to consider instead of using the all the data of the dataset. If multiple
                datasets are given, each dataset will participate equality. Eg, if subset = 10 and there are two
                dataset, each one will contribute with 5 data samples.
                Additionally, the contribution of each dataset should be of one batch. If batch_size = 8 and
                subset = 10, then each dataset will contribute with 8 samples.
        """

        datasets_ = []
        for dataset_ in datasets_not_init:
            datasets_.append(dataset_(**kwargs))

        if (subset is not None) and (subset > 0):
            idxs = [i for i in range(max(subset // len(datasets_not_init), batch_size))]
            datasets_ = [Subset(dataset_, idxs) for dataset_ in datasets_]

        return ConcatDataset(datasets_)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the datasets."""

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
        return nn.Identity()

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
        """Define the augmentations for this batch."""

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
        """Operations before batch is transferred to GPU."""

        if self.trainer.training and self.config_train.p_random_transforms:
            # concatenates image and mask in the channel dimension.
            img_mask = torch.cat(
                (batch[DataOutputs.Image], batch[DataOutputs.Mask]), dim=1
            )

            # Augment the image.
            # :warning: **augmentations that alter pixel values will also alter the mask.**
            img_mask = self.define_augmentations(
                p=self.config_train.p_random_transforms,
                img_size=self.config_train.img_size,
            )(img_mask)

            # splits the augmented image into image and mask again.
            batch[DataOutputs.Image], batch[DataOutputs.Mask] = torch.split(
                img_mask, split_size_or_sections=1, dim=1
            )

        return batch


if __name__ == "__main__":

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
