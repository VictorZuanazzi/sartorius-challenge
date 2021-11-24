from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yamldataclassconfig import build_path, YamlDataClassConfig

from data import (
    DataOutputs,
    DataloaderGlobalConfig,
    DataloaderConfig,
    GenericDataModule,
)
from metrics import (
    calculate_competion_iou,
    calculate_iou,
    calculate_acc,
    calculate_binary_dice, inverse_frequency_weighting, bce_weighted,
)
from model import DeepLabSegmeter, ModelConfiguration
from visualization import visualize_intersection


class SartoriousSegmentation(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, head: str) -> Dict[str, Tensor]:
        return self.model(x)

    def log_images(self, mask_pred, mask_gt, img, label):
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        img = batch[DataOutputs.Image]
        mask_gt = batch[DataOutputs.Mask]

        mask_pred = self.model(img)["mask"]
        balancing_importance = inverse_frequency_weighting(mask_gt)
        # loss_mask = nn.BCEWithLogitsLoss()(mask_pred, mask_gt)
        loss_mask = bce_weighted(mask_pred, mask_gt, balancing_importance)
        loss_dice = 1 - calculate_binary_dice(mask_pred, mask_gt,
                                              threshold=-1, weights=balancing_importance)
        train_loss = loss_mask + loss_dice

        self.log("loss", {"mask": loss_mask, "dice ": loss_dice, "train": train_loss})
        self.log("train acc", calculate_acc(mask_pred, mask_gt))
        self.log("train IoU", calculate_iou(mask_pred, mask_gt, threshold=0.5))
        self.log("train DICE", calculate_binary_dice(mask_pred, mask_gt, threshold=0.5))
        self.log("train score", calculate_competion_iou(mask_pred, mask_gt))

        self.log_images(mask_pred, mask_gt, img, label="train")

        return train_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img = batch[DataOutputs.Image]
        mask_gt = batch[DataOutputs.Mask]

        mask_pred = self.model(img)["mask"]
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

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):
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

        if isinstance(self.n_gpus, str):
            self.n_gpus: List[int] = [int(gpu) for gpu in self.n_gpus]

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

    pretrained_model_path = Path(
        "./lightning_logs/version_9/checkpoints/epoch=18-step=322-eval score=0.34.ckpt"
    )

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

    datamodule = GenericDataModule(
        num_workers=config.dataloader_global.num_workers,
        configuration_train=config.dataloader_train,
        configuration_eval=config.dataloader_eval,
        configuration_test=config.dataloader_test,
    )

    # train_loader = DataLoader(dataset=SatoriusDataset(root=config.dataloader_train.dataset_root,
    #                                                   partition=Partition.Train,
    #                                                   outputs=config.dataloader_train.outputs),
    #                           batch_size=config.dataloader_train.batch_size,
    #                           num_workers=config.dataloader_global.num_workers,
    #                           shuffle=True,
    #                           drop_last=True)
    #
    # eval_loader = DataLoader(dataset=SatoriusDataset(root=config.dataloader_eval.dataset_root,
    #                                                  partition=Partition.Eval,
    #                                                  outputs=config.dataloader_eval.outputs),
    #                          batch_size=config.dataloader_eval.batch_size,
    #                          num_workers=config.dataloader_global.num_workers,
    #                          shuffle=False,
    #                          drop_last=True)

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

    multi_gpu_strategy = None
    if (isinstance(config.training.n_gpus, int) and config.training.n_gpus > 1) or (
        isinstance(config.training.n_gpus, list) and len(config.training.n_gpus) > 1
    ):
        multi_gpu_strategy = "ddp"

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

    trainer.fit(
        model=satorious_masker,
        datamodule=datamodule,
        ckpt_path=config.training.resume_from_checkpoint,
    )
