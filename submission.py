import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image, make_grid
from model import DeepLabSegmeter, Backbone

# from main import SartoriousSegmentation, Configuration
from generate_submissionfile import save_submission

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

testset_root = Path("../data/test")
pred_path = Path("../preds/")
pred_path.mkdir(parents=True, exist_ok=True)

config_path = Path("config_train_0.yaml")

model_file_path = Path("./checkpoint/last.pt")
# model_file_path = Path('../lightning_logs/version_11/checkpoints/last.ckpt')

device = "cuda" if torch.cuda.is_available() else "cpu"

# config = Configuration.load_from_config_file(path=config_path)

model_ = DeepLabSegmeter(
            backbone=Backbone.R50,
            pretrained_backbone=True,
            feat_map_n_dims=128,
            input_dim=1,
            heads_segmentation={"mask": 1},
            heads_global=None,
        )


model_.load_state_dict(torch.load(model_file_path))
model = model_
# model = SartoriousSegmentation.load_from_checkpoint(str(model_file_path),
#                                                     model=model_
#                                                     )
model.to(device)
model.eval()

masks_in_a_nice_list = []
img_name_in_a_nice_list = []
for img_name in os.listdir(testset_root):
    img_ = pil_to_tensor(Image.open(testset_root / img_name)) / 255.0
    img = img_.to(device).unsqueeze(0)
    # pred_ = model.model(torch.cat([img] * 2, dim=0))
    pred_ = model(torch.cat([img] * 2, dim=0))

    pred = pred_['mask'][0:1]

    vizualization = make_grid(
        torch.cat([(pred.sigmoid() > 0.5).float(), img, pred.sigmoid()], dim=0), nrow=1, pad_value=1
    )
    save_image(vizualization, fp=pred_path / img_name)

    masks_in_a_nice_list.append(pred.cpu())
    img_name_in_a_nice_list.append(img_name.replace(".png", ""))

save_submission(filename="submission.csv",
                tensors=masks_in_a_nice_list,
                identifications=img_name_in_a_nice_list,
                start_index=1)
