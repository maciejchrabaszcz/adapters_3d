import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.networks.nets import SegResNet
from models.unet3d import UNet3D
from models.swin_smt import SwinSMT

from monai.data import (
    DataLoader,
    decollate_batch
)
import gc
from data_loader import PANORAMADataset
import argparse
from lightning.fabric import Fabric


parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    default="../dataset",
                    type=str,
                    help="Path to the training data.")
parser.add_argument("--data_train",
                    default="/net/tscratch/people/plgszymonplotka/PANORAMA_converted/train",
                    type=str,
                    help="Path to the training data.")
parser.add_argument("--data_val",
                    default="/net/tscratch/people/plgszymonplotka/PANORAMA_converted/valid",
                    type=str,
                    help="Path to the training data.")
parser.add_argument("--batch_size",
                    default=2,
                    type=int,
                    help="Number of batch size.")
parser.add_argument("--skip_val",
                    default=1,
                    type=int,
                    help="Skip validation step by N epochs.")
parser.add_argument("--classes",
                    default=3,
                    type=int,
                    help="Number of classes.")
parser.add_argument("--epochs",
                    default=300,
                    type=int,
                    help="Number of epochs.")
parser.add_argument("--lr",
                    default=1e-4,
                    type=float,
                    help="Learning rate value.")
parser.add_argument("--weight_decay",
                    default=1e-4,
                    type=float,
                    help="Weight decay value.")
parser.add_argument("--optimizer",
                    default="AdamW",
                    type=str,
                    help="Type of optimizer.")
parser.add_argument("--scheduler",
                    default="CALR",
                    type=str,
                    help="Type of learning rate scheduler.")
parser.add_argument("--k_fold",
                    default=5,
                    type=int,
                    help="Number of K-Fold splits.")
parser.add_argument("--patch_size",
                    default=(96, 96, 96),
                    type=list,
                    help="Patch size value.")
parser.add_argument("--feature_size",
                    default=48,
                    type=int,
                    help="Feature size of Transformer.")
parser.add_argument("--use_checkpoint",
                    default=True,
                    type=bool,
                    help="Use checkpoint in training model.")
parser.add_argument("--num_workers",
                    default=96,
                    type=int,
                    help="Number of workers.")
parser.add_argument("--pin_memory",
                    default=True,
                    type=bool,
                    help="Pin memory.")
parser.add_argument("--use_pretrained",
                    default=False,
                    type=bool,
                    help="Use pre-trained weights.")
parser.add_argument("--load_checkpoint",
                    default=False,
                    type=bool,
                    help="Load saved checkpoint.")
parser.add_argument("--checkpoint_name",
                    default="PancreasSurvival",
                    type=str,
                    help="Name of the checkpoint file.")
parser.add_argument("--model",
                    default="SwinUNETR",
                    type=str,
                    help="Type of model.")
parser.add_argument("--parallel",
                    default=True,
                    type=bool,
                    help="Use multi-GPU.")
parser.add_argument("--model_name",
                    default="Fullbody",
                    type=str,
                    help="File model name.")
parser.add_argument("--num_devices",
                    default=8,
                    type=int,
                    help="Number of devices.")
parser.add_argument("--strategy",
                    default="ddp",
                    type=str,
                    help="Strategy of training.")

args = parser.parse_args()


def load(model, model_dict):
    # make sure you load our checkpoints
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else
        current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model


def save_checkpoint(global_step, model, optimizer, scheduler, scaler):
    save_dict = {
        "step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict()
    }
    save_path = os.path.join(args.data_train, f"PANORAMA_{args.model}_{args.epochs}_final.pth")
    fabric.save(save_path, save_dict)


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = fabric.load(checkpoint)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    global_step = checkpoint["step"]
    return global_step


def simplify_key(k):
    """
    Simplify the key by removing common but irrelevant substrings
    Please modify this function according to your needs for your loaded model

    Args:
        k (str): the key to simplify

    Returns:
        str: the simplified key
    """
    for prefix in ['module.', 'features.', 'backbone.', 'model.']:
        k = k.replace(prefix, '')
    return k


def train(global_step, train_loader, valid_loader, dice_val_best, global_step_best, fabric: Fabric):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        torch.cuda.empty_cache()
        gc.collect()
        step += 1
        x, y = (batch["image"], batch["label"])
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)

        optimizer.zero_grad()
        fabric.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, len(train_loader) * args.epochs, loss)
        )
        if (
                global_step % (args.skip_val * len(train_loader)) == 0 and global_step != 0
        ) or global_step == len(train_loader) * args.epochs:
            epoch_iterator_val = tqdm(
                valid_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                save_checkpoint(global_step=global_step,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler)
                print(
                    "Model was saved...Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model was not saved...Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            scheduler.step()
        global_step += 1

    return global_step, dice_val_best, global_step_best


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, args.patch_size, 4, model, sw_device="cuda",
                                                       device="cuda")
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric1(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val1 = dice_metric1.aggregate().item()
        dice_metric1.reset()
    return mean_dice_val1


fabric = Fabric(devices=args.num_devices, strategy=args.strategy)
fabric.launch()
train_ds = PANORAMADataset(args.data_train,
                           mode="train",
                           patch_size=args.patch_size)
valid_ds = PANORAMADataset(args.data_val,
                           mode="valid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()

post_label = AsDiscrete(to_onehot=args.classes)
post_pred = AsDiscrete(argmax=True, to_onehot=args.classes)
dice_metric1 = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
train_loader = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=args.pin_memory)

valid_loader = DataLoader(valid_ds,
                          batch_size=1,
                          num_workers=0,
                          pin_memory=False)

if args.model == "VoCo_B":
    model = SwinUNETR(img_size=args.patch_size,
                      in_channels=1,
                      out_channels=3,
                      feature_size=48,
                      use_v2=True)
    pretrained_path = "/net/tscratch/people/plgszymonplotka/PANORAMA/VoCo_B_SSL_head.pt"
    print("Loading Weights from the Path {}".format(pretrained_path))
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)

elif args.model == "VoCo_L":
    model = SwinUNETR(img_size=args.patch_size,
                      in_channels=1,
                      out_channels=3,
                      feature_size=96,
                      use_v2=True)
    pretrained_path = "/net/tscratch/people/plgszymonplotka/PANORAMA/VoCo_B_SSL_head.pt"
    print("Loading Weights from the Path {}".format(pretrained_path))
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)

elif args.model == "VoComni_B":
    model = SwinUNETR(img_size=args.patch_size,
                      in_channels=1,
                      out_channels=3,
                      feature_size=48,
                      use_v2=True)
    pretrained_path = "/net/tscratch/people/plgszymonplotka/PANORAMA/VoComni_B.pt"
    print("Loading Weights from the Path {}".format(pretrained_path))
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)

elif args.model == "VoComni_L":
    model = SwinUNETR(img_size=args.patch_size,
                      in_channels=1,
                      out_channels=3,
                      feature_size=96,
                      use_v2=True)
    pretrained_path = "/net/tscratch/people/plgszymonplotka/PANORAMA/VoComni_L.pt"
    print("Loading Weights from the Path {}".format(pretrained_path))
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)

elif args.model == "SuPreM_Swin":
    model = SwinUNETR(img_size=args.patch_size,
                      in_channels=1,
                      out_channels=3,
                      feature_size=48,
                      use_v2=True)
    # Load pre-trained weights
    store_dict = model.state_dict()
    model_dict = torch.load(args.pretrain)['net']
    amount = 0
    for key in model_dict.keys():
        new_key = '.'.join(key.split('.')[1:])
        if 'backbone' in new_key:
            n_key = '.'.join(new_key.split('.')[1:])
            if n_key in store_dict.keys():
                store_dict[n_key] = model_dict[key]
                amount += 1
    model.load_state_dict(store_dict)
    print('Use SuPreM SwinUnetr backbone pretrained weights')

elif args.model == "SuPreM_3DUNet":
    model = UNet3D(n_class=3)
    model_dict = torch.load(args.pretrain)['net']
    store_dict = model.state_dict()
    amount = 0
    for key in model_dict.keys():
        new_key = '.'.join(key.split('.')[2:])
        if new_key in store_dict.keys():
            store_dict[new_key] = model_dict[key]
            amount += 1

    model.load_state_dict(store_dict)
    print('Use SuPreM U-Net backbone pretrained weights')

elif args.model == "SuPreM_SegResNet":
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=args.segresnet_init_filters,
        in_channels=1,
        out_channels=args.num_class,
        dropout_prob=0.0,
    )
    model_dict = torch.load(args.pretrain)['net']
    store_dict = model.state_dict()
    simplified_model_dict = {simplify_key(k): v for k, v in model_dict.items()}
    amount = 0
    for key in store_dict.keys():
        if key in simplified_model_dict and 'conv_final.2.conv' not in key:
            store_dict[key] = simplified_model_dict[key]
            amount += 1
    assert amount == (len(store_dict.keys()) - 2), 'the pre-trained model is not loaded successfully'
    print('loading weights', amount, len(store_dict.keys()))
    model.load_state_dict(store_dict)

elif args.model == "SwinSMT":
    model = SwinSMT(
        in_channels=1,
        out_channels=17,
        img_size=(128, 128, 128),
        spatial_dims=3,
        use_v2=True,
        feature_size=48,
        use_moe=True,
        num_experts=4,
        num_layers_with_moe=3
    )
    pretrained_path = "/net/tscratch/people/plgszymonplotka/PANORAMA/swin_smt.pt"
    print("Loading Weights from the Path {}".format(pretrained_path))
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)

else:
    raise NotImplementedError("This model not exists!")


if args.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  eps=1e-5)
elif args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
else:
    raise NotImplementedError("Optimizer not found! Please select one of [Adam, AdamW or SGD]")

model, optimizer = fabric.setup(model, optimizer)
train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)
if args.scheduler == "CALR":
    scheduler = CosineAnnealingLR(optimizer=optimizer,
                                  T_max=args.epochs // args.skip_val,
                                  verbose=True)
else:
    raise NotImplementedError("Learning rate scheduler not found! Please select one of [CALR]")

dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
global_step = 0

if args.load_checkpoint:
    global_step = load_checkpoint(checkpoint=args.checkpoint_name,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  scaler=scaler)

while global_step < len(train_loader) * args.epochs:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, valid_loader, dice_val_best, global_step_best, fabric
    )