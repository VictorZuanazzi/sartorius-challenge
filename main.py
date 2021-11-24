from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yamldataclassconfig import YamlDataClassConfig, build_path

from data import (DataloaderConfig, DataloaderGlobalConfig, DataOutputs,
                  GenericDataModule)
from metrics import (bce_weighted, calculate_acc, calculate_binary_dice,
                     calculate_competion_iou, calculate_iou,
                     inverse_frequency_weighting)
from model import DeepLabSegmeter, ModelConfiguration
from visualization import visualize_intersection


class SartoriousSegmentation(pl.LightningModule):
    """Main training module.

    Implements training, evaluation and inference logics.

    Training strategy:
        Binary segmentation (or mask prediction) for binary classification.
        Supervised learning for mask prediction.
        There is space for multitask / self-supervised methods.
    """

    def __init__(self, model: nn.Module) -> None:
        """Constructor.

        Args:
            model: initialized pytorch model.
        """

        super().__init__()
        self.model = model

    def forward(self, x: Tensor, head: str) -> Dict[str, Tensor]:
        """Inference forward.

        :warning: **This method was not implemented for the purposes of the challenge yet.**
        """

        return self.model(x)

    def log_images(
        self, mask_pred: Tensor, mask_gt: Tensor, img: Tensor, label: str
    ) -> None:
        """Log images into the logger.

        Args:
            mask_pred: The batch of  predicted masks. Shape: (B, 1, W, H)
            mask_gt: The ground truth mask. Shape: (B, 1, W, H)
            img: The image used for generating the mask. Shape: (B, 1, W, H)
            label: Watherver string to differentiate the images log, eg 'train' and 'eval'
        """

        intersection_raw, intersection_rgb, intersection_img = visualize_intersection(
            mask_pred, mask_gt, img, threshold=0.5
        )
        self.logger.experiment.add_image(
            f"{label} intersection_raw", intersection_raw[0], self.global_step
        )
        self.logger.experiment.add_image(
            f"{label} intersection_rgb", intersection_rgb[0], self.global_step
        )
        self.logger.experiment.add_image(
            f"{label} intersection_img", intersection_img[0], self.global_step
        )
        self.logger.experiment.add_image(
            f"{label} all",
            torch.cat(
                ((mask_pred[0].sigmoid() > 0.5).float(), mask_gt[0], img[0]), dim=-1
            ),
            self.global_step,
        )

    def training_step(
        self, batch: Dict[DataOutputs, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Performs one training step.

        Args:
            batch: Batch of examples for training.
            batch_idx: index of the batch.

        Returns:
            The loss of the training step.
        """

        # get the outputs we need
        img = batch[DataOutputs.Image]
        mask_gt = batch[DataOutputs.Mask]

        # forward step
        mask_pred = self.model(img)["mask"]

        # calculate losses
        balancing_importance = inverse_frequency_weighting(mask_gt)
        # loss_mask = nn.BCEWithLogitsLoss()(mask_pred, mask_gt)
        loss_mask = bce_weighted(mask_pred, mask_gt, balancing_importance)
        loss_dice = 1 - calculate_binary_dice(
            mask_pred, mask_gt, threshold=-1, weights=balancing_importance
        )
        train_loss = loss_mask + loss_dice

        # Calculate and log a bunch of metrics
        self.log("loss", {"mask": loss_mask, "dice ": loss_dice, "train": train_loss})
        self.log("train acc", calculate_acc(mask_pred, mask_gt))
        self.log("train IoU", calculate_iou(mask_pred, mask_gt, threshold=0.5))
        self.log("train DICE", calculate_binary_dice(mask_pred, mask_gt, threshold=0.5))
        self.log("train score", calculate_competion_iou(mask_pred, mask_gt))

        self.log_images(mask_pred, mask_gt, img, label="train")

        return train_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """Performs one evaluation step.

        Args:
            batch: Batch of examples for evaluation.
            batch_idx: index of the batch.
        """

        # Get the inputs
        img = batch[DataOutputs.Image]
        mask_gt = batch[DataOutputs.Mask]

        # forward step
        mask_pred = self.model(img)["mask"]

        # Calculate and log a bunch of metrics
        self.log("eval acc", calculate_acc(mask_pred, mask_gt))
        self.log("eval IoU", calculate_iou(mask_pred, mask_gt, threshold=0.5))
        self.log(
            "eval DICE",
            calculate_binary_dice(mask_pred, mask_gt, threshold=0.5),
            prog_bar=True,
        )
        self.log(
            "eval score", calculate_competion_iou(mask_pred, mask_gt), prog_bar=True
        )

        self.log_images(mask_pred, mask_gt, img, label="eval")

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4) -> Dict[Any, Any]:
        """Configure Optimizer.

        :warning: **I am still not sure how I am supposed to use this function in the context of pytorch lightining.**

        Args:
            lr: learning rate
            weight_decay: weight decay

        Returns:
            (From the documentatio:) **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
        """
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "eval score",
                "frequency": 10,
            },
        }


@dataclass
class TrainConfiguration(YamlDataClassConfig):
    random_seed: int
    n_gpus: Union[int, str]
    log_every_n_steps: int
    gradient_clip_val: float
    max_steps: int
    num_sanity_val_steps: int
    track_grad_norm: Union[int, str]
    resume_from_checkpoint: Path = field(
        default_factory=Path, metadata={"dataclasses_json": {"mm_field": Path}}
    )

    def __post_init__(self):

        # parse the gpus if the specific nodes are requested as string.
        if isinstance(self.n_gpus, str):
            self.n_gpus: List[int] = [int(gpu) for gpu in self.n_gpus]

        # makes the path absolute
        self.resume_from_checkpoint: Path = build_path(
            path=self.resume_from_checkpoint, path_is_absolute=True
        )


@dataclass
class Configuration(YamlDataClassConfig):
    training: TrainConfiguration = None
    model: ModelConfiguration = field(
        default=None, metadata={"dataclasses_json": {"mm_field": DataloaderConfig}}
    )
    dataloader_global: DataloaderGlobalConfig = None
    dataloader_train: DataloaderConfig = field(
        default=None, metadata={"dataclasses_json": {"mm_field": DataloaderConfig}}
    )
    dataloader_eval: DataloaderConfig = field(
        default=None, metadata={"dataclasses_json": {"mm_field": DataloaderConfig}}
    )
    dataloader_test: DataloaderConfig = field(
        default=None, metadata={"dataclasses_json": {"mm_field": DataloaderConfig}}
    )

    @staticmethod
    def load_from_config_file(path: Union[Path, str]) -> ["Configuration"]:
        config = Configuration()
        config.FILE_PATH = build_path(path)
        config.load(path=path)
        return config


if __name__ == "__main__":

    # Get configuration file.
    parser = ArgumentParser("Training script of Sartorius.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()
    config = Configuration.load_from_config_file(path=Path(args.config))

    # Set random seed.
    pl.seed_everything(config.training.random_seed, workers=True)

    # Initialize the Training module (I am never sure of what to call this PL module)
    satorious_masker = SartoriousSegmentation(
        model=DeepLabSegmeter(
            backbone=config.model.backbone,
            pretrained_backbone=config.model.pretrained_backbone,
            feat_map_n_dims=config.model.feat_map_n_dim,
            input_dim=config.model.input_dim,
            heads_segmentation=config.model.heads_segmentation,
            heads_global=config.model.heads_global,
        )
    )

    # Initialize the datamodule
    datamodule = GenericDataModule(
        num_workers=config.dataloader_global.num_workers,
        configuration_train=config.dataloader_train,
        configuration_eval=config.dataloader_eval,
        configuration_test=config.dataloader_test,
    )

    # Initialize the checkpoint callbacks
    checkpoint_callback_eval = ModelCheckpoint(
        filename="{epoch}-{step}-{eval score:.2f}",
        monitor="eval score",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True,
    )

    checkpoint_callback_train = ModelCheckpoint(
        filename="{epoch}-{step}-{train score:.2f}",
        monitor="train score",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True,
    )

    # Some back-end infra for defining how to parallelize multi-gpu training, in case it is required.
    multi_gpu_strategy = None
    if (isinstance(config.training.n_gpus, int) and config.training.n_gpus > 1) or (
        isinstance(config.training.n_gpus, list) and len(config.training.n_gpus) > 1
    ):
        multi_gpu_strategy = "ddp"

    # Initialize the PL trainer
    trainer = pl.Trainer(
        gpus=config.training.n_gpus,
        strategy=multi_gpu_strategy,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=config.training.log_every_n_steps,
        max_steps=config.training.max_steps,
        num_sanity_val_steps=config.training.num_sanity_val_steps,
        track_grad_norm=config.training.track_grad_norm,
        enable_model_summary=True,
        callbacks=[checkpoint_callback_eval, checkpoint_callback_train],
    )

    # That is where the magic happens!
    trainer.fit(
        model=satorious_masker,
        datamodule=datamodule,
        ckpt_path=config.training.resume_from_checkpoint,
    )
