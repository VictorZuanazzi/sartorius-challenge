from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from yamldataclassconfig import YamlDataClassConfig


class Backbone(Enum):
    R50 = "resnet50"
    R101 = "resnet101"


@dataclass
class ModelConfiguration(YamlDataClassConfig):
    pretrained_backbone: bool
    feat_map_n_dim: int
    backbone: str  # typing Backbone breaks =/
    input_dim: int = 1
    heads_segmentation: Dict[str, int] = field(default_factory=dict)
    heads_global: Optional[Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        self.backbone: Backbone = Backbone[self.backbone]


class DeepLabSegmeter(nn.Module):
    """Model for segmentation."""

    def __init__(
        self,
        backbone: Backbone,
        pretrained_backbone: bool,
        feat_map_n_dims: int,
        input_dim: int = 1,
        heads_segmentation: Dict[str, int] = None,
        heads_global: Dict[str, int] = None,
    ):
        """Constructor of the segmentation model.

        Args:
            backbone: Which backbone to use (resnet50 or resnet101).
            pretrained_backbone: Use imagenet pretrained resnet if True.
            feat_map_n_dims: Number of dimensions of the feature map (this is the output of the DeepLabV3,
                this feature map can be used by multiple heads for multi-task learning)
            input_dim: number of channels of the input.
            heads_segmentation: Mapping of the name of the pixel-wise head and the number of dimensions.
            heads_global: Mapping of the name of the global head and the number of dimensions.
        """
        super().__init__()

        # number of dimensions of the resnet.
        self.global_n_dims = 2048

        # initialize segmentation model
        if backbone == Backbone.R50:
            self.main_model = deeplabv3_resnet50(
                pretrained=False,
                progress=True,
                num_classes=feat_map_n_dims,
                pretrained_backbone=pretrained_backbone,
            )

        elif backbone == Backbone.R101:
            self.main_model = deeplabv3_resnet101(
                pretrained=False,
                progress=True,
                num_classes=feat_map_n_dims,
                pretrained_backbone=pretrained_backbone,
            )

        # input layer maps the 1 channel input to 3 (RBG) channels
        self.input_layer = nn.Conv2d(
            in_channels=input_dim, out_channels=3, kernel_size=1
        )
        self.feature_extractor = nn.Sequential(self.input_layer, self.main_model)

        # final layers
        self.heads_segmentation = nn.ModuleDict()
        heads_segmentation = (
            dict() if heads_segmentation is None else heads_segmentation
        )
        for name, out_features in heads_segmentation.items():
            self.heads_segmentation[name] = nn.Conv2d(
                in_channels=feat_map_n_dims, out_channels=out_features, kernel_size=1
            )

        self.heads_global = nn.ModuleDict()
        heads_global = dict() if heads_global is None else heads_global
        for name, out_features in heads_global.items():
            self.heads_global[name] = nn.Linear(
                in_features=self.global_n_dims, out_features=out_features
            )

    def forward(self, x: Tensor, heads: Sequence[str] = None) -> Dict[str, Tensor]:
        x_ = self.feature_extractor(x)["out"]

        heads = self.heads_segmentation.keys() if heads is None else heads
        output = {"feature map": x_}
        for head in heads:
            output[head] = self.heads_segmentation[head](x_)

        return output

    def forward_global(
        self, x: Tensor, heads: Sequence[str] = None
    ) -> Dict[str, Tensor]:
        z_ = self.main_model.backbone(self.input_layer(x))["out"].mean(dim=(-1, -2))

        heads = self.heads_global.keys() if heads is None else heads
        output = {"feature global": z_}
        for head in heads:
            output[head] = self.heads_global[head](z_)

        return output


if __name__ == "__main__":
    breakpoint()

    model = DeepLabSegmeter(
        backbone=Backbone.R50,
        pretrained_backbone=True,
        feat_map_n_dims=128,
        input_dim=1,
        heads_segmentation={"mask": 1},
        heads_global={"area": 1},
    )

    batch = torch.rand(size=(4, 1, 32, 32))
    mask = model(batch)
    area = model.forward_global(batch)
