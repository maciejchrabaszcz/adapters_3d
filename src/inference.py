import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import glob
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.networks.nets import UNETR
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    ToTensord,
    EnsureTyped,
    Invertd,
    ActivationsD,
    SaveImageD,
    AsDiscreteD
)

from models.lora_finetuning import get_model_with_lora


os.environ["CUDA_MODULE_LOADING"] = "LAZY"

#DATA_PATH = "/net/pr2/projects/plgrid/plggsano/plgszymonplotka/PANORAMA_converted/val"
#MODEL_PATH = "/net/pr2/projects/plgrid/plggsano/plgszymonplotka/PANORAMA_converted/train/PANORAMA_UNet_250_scratch_2.pth"

DATA_PATH="/net/tscratch/people/plgszymonplotka/test_MSD"
MODEL_PATH="/net/tscratch/people/plgszymonplotka/PANORAMA_converted/fold1/PANORAMA_SuPreM_Swin_1000_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    device_name = torch.cuda.get_device_name(0)

inference_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        Spacingd(keys=["image"],
                 pixdim=(0.8, 0.8, 3.0),
                 mode="bilinear")
    ]
)

post_transform = Compose(
    [
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=inference_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True
        ),
        ActivationsD(keys="pred", softmax=True),
        AsDiscreteD(keys="pred", argmax=True),
        SaveImageD(keys="pred",
                   meta_keys="pred_meta_dict",
                   output_dir="inference_PanMamba_MSD",
                   output_postfix="predictions",
                   separate_folder=False,
                   resample=False)
    ]
)

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True,
    use_v2=True
    )

model = get_model_with_lora(
        model,
        8,
        8
    )

model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)["model"])
model.eval()

inferer = SlidingWindowInferer(
    roi_size=(192, 192, 64),
    sw_batch_size=1,
    sw_device="cuda",
    device="cuda",
    overlap=0.5,
    mode="gaussian",
    padding_mode="replicate",
)

data_dir = DATA_PATH
test_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
test_data = [{"image": image} for image in test_images]
test_dataset = Dataset(data=test_data, transform=inference_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

with torch.no_grad():
    for test_data in test_loader:
        print(test_data)
        test_inputs = test_data["image"].to("cuda")
        with autocast():
            test_data["pred"] = sliding_window_inference(test_inputs, (96, 96, 96), 4, model,
                                                         sw_device="cuda", device="cuda", overlap=0.5, progress=True)
        test_data = [post_transform(i) for i in decollate_batch(test_data)]
        torch.cuda.empty_cache()