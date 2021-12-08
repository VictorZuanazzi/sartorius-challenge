from pathlib import Path

import torch

from main import SartoriousSegmentation
from model import DeepLabSegmeter, Backbone

model_file_path = Path('../lightning_logs/version_11/checkpoints/last.ckpt')
model_save_path = Path('./checkpoint/last.pt')


model_ = DeepLabSegmeter(
    backbone=Backbone.R50,
    pretrained_backbone=True,
    feat_map_n_dims=128,
    input_dim=1,
    heads_segmentation={"mask": 1},
    heads_global=None,
)

model_pl = SartoriousSegmentation.load_from_checkpoint(str(model_file_path), model=model_)

torch.save(model_pl.model.state_dict(), model_save_path)
