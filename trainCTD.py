"""
Script  to train updated CameraTrapDetectoR model
"""

# Imports
import os
import torch
from utils.data_process import class_range, existing_images, \
    orient_boxes, format_vars, split_df
from utils import dicts
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from utils.data_setload import DetectDataset, get_class_weights
from torch.utils.data import DataLoader

from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn

from torch.optim import SGD, lr_scheduler
from utils.hyperparameters import get_lr

import time
import tqdm
from datetime import timedelta
from models.train_one_epoch import train_one_epoch
import copy
from utils.checkpoints import write_args, \
    create_checkpoint, save_checkpoint

import torchvision
from utils.metrics import prepare_results
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

#######
## -- Prepare System and Data for Model Training
#######

# Set location
if os.name == 'posix':
    local = False
else:
    local = True

# Set paths
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/trainimages"
    os.chdir('/project/cameratrapdetector')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set model type: options = 'general', 'family', 'species', 'pig_only'
model_type = 'species'

max_per_class, min_per_class = class_range(model_type)

# load annotations
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")

# prune df to existing images
df = existing_images(df, IMAGE_ROOT)

# confirm all bbox coordinates correspond to UL, LR corners
df = orient_boxes(df)

# format df
df = format_vars(df)

# create dictionary - create the dict corresponding to model type
df, label2target, columns2stratify = dicts.spec_dict(df, max_per_class, min_per_class)

# reverse dictionary to read into pytorch
target2label = {t: l for l, t in label2target.items()}

# define number of classes
num_classes = max(label2target.values()) + 1

# split df into train/val/test sets
train_df, val_df, test_df = split_df(df, columns2stratify)

# set image dimensions for training
# TODO: determine ideal training w and h
w = 408
h = 307

# define data augmentation pipelines
# note augmentations as a string to save in model arguments
transforms = 'shear, rotate, huesat, brightcont, safecrop, hflip'

train_transform = A.Compose([A.Affine(shear=(-30, 30), fit_output=True, p=0.3),
                             A.Affine(rotate=(-30, 30), fit_output=True, p=0.3),
                             A.HueSaturationValue(p=0.4),
                             A.RandomBrightnessContrast(p=0.4),
                             A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.4),
                             A.HorizontalFlip(p=0.4),
                             ToTensorV2()],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),)

val_transform = A.Compose([ToTensorV2()],
                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),)


# Load PyTorch Datasets
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=w, h=h,
                         label2target=label2target, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=w, h=h,
                       label2target=label2target, transform=val_transform)

# define anchor boxes based on image size
anchor_sizes, anchor_gen = get_anchors(h=h)

# define model backbone
cnn_backbone = 'resnet'

# initialize model
model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# define number of training epochs
num_epochs = 50

# define batch size
batch_size = 32
batch_size_org = batch_size

# boolean to use gradient accumulation
use_grad = False

# set denominator if using gradient accumulation
if use_grad:
    # set number of gradients to accumulate before updating weights
    grad_accumulation = 8
    # effective batch size = batch_size * grad_accumulation
    batch_size = batch_size_org // grad_accumulation


# set up class weights
sampler = get_class_weights(train_df)

# define dataloaders
train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn,
                          drop_last=True, sampler=sampler)
# do not oversample for validation, just for training
val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)

# define starting lr, wd, momentum
#TODO: review training results to confirm these
lr = 0.01
wd = 0.001
momentum = 0.9

# load optimizer
#TODO: create function to choose this based on optim
optim = 'SGD'
optimizer = SGD(params=params, lr=lr, weight_decay=wd, momentum=momentum)

# set learning rate scheduler
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# make output directory and filepath
output_path = "./output/" + "fasterRCNN_" + cnn_backbone + "_" + \
              time.strftime("%Y%m%d") + "_" + time.strftime("%H%M") + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# write model args to text file
write_args(cnn_backbone, w, h, transforms, anchor_sizes, batch_size_org, optim, lr, wd, lr_scheduler, output_path)

# write datasets to csv
train_df.to_csv(output_path + "train_df.csv")
val_df.to_csv(output_path + "val_df.csv")
test_df.to_csv(output_path + "test_df.csv")

#######
## -- Train the Model
#######


# define starting weights
# TODO: incorporate function for loading existing model weights when training iteratively
best_model_wts = copy.deepcopy(model.state_dict())

# define starting loss
best_loss = float('inf')

# create empty list to save train/val losses
loss_history = {
    'train': [],
    'val': []
}

# create empty list to save training time per epoch
training_time = []

# training/validation loop
for epoch in range(num_epochs):
    # record start time
    t_start = time.time()

    # set learning rate
    current_lr = get_lr(optimizer)

    # print epoch info
    print('Epoch {}, current lr={}'.format(epoch + 1, current_lr))

    # run training/validation
    # Note: arguments should already be specified!
    model, optimizer, loss_history, val_loss = train_one_epoch()

    ## -- UPDATE MODEL

    # record model weights based on val_loss
    if val_loss < best_loss:
        # update best loss
        best_loss = val_loss
        # update model weights
        best_model_wts = copy.deepcopy(model.state_dict())

    # adjust learning rate
    lr_scheduler.step(val_loss)

    # load best model weights if current epoch did not produce best weights
    if current_lr != get_lr(optimizer):
        model.load_state_dict(best_model_wts)

    # calculate training time per epoch
    elapsed = time.time() - t_start
    elapsed = str(timedelta(seconds=elapsed))
    training_time.append(elapsed)

    # save model state
    checkpoint = create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history,
                                   best_loss, model_type, num_classes, label2target, training_time)
    checkpoint_file = output_path + "checkpoint_" + str(epoch + 1) + "epochs.pth"
    save_checkpoint(checkpoint, checkpoint_file)


#######
## -- Evaluate the Model
#######

# list unique images in test dataset
image_infos = test_df.filename.unique()

# create placeholders for targets and predictions
pred_df = []
target_df = []

#TODO: set this apart as its own function?
# deploy model on test images
model.eval()
for i in tqdm(range(len(image_infos))):
    # define dataset and dataloader
    dfi = df[df['filename'] == image_infos[i]]
    dsi = DetectDataset(df=dfi, image_dir=IMAGE_ROOT, w=w, h=h, label2target=label2target, transform=val_transform)
    dli = DataLoader(dsi, batch_size=1, collate_fn=dsi.collate_fn, drop_last=True)

    # extract image, bbox, and label info
    input, target = next(iter(dli))
    tbs = dsi[0][1]['boxes']
    image = list(image.to(device) for image in input)

    # run input through the model
    output = model(image)[0]

    # extract prediction bboxes, labels, scores above score_thresh
    # format prediction data
    bbs = output['boxes'].cpu().detach()
    labels = output['labels'].cpu().detach()
    confs = output['scores'].cpu().detach()

    # id indicies of tensors to include in evaluation
    idx = torch.where(confs > 0.1)

    # filter to predictions that meet the threshold
    bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

    # perform non-maximum suppression on remaining predictions
    # set iou threshold low since training data does not contain overlapping ground truth boxes
    ixs = torchvision.optim.nms(bbs, confs, iou_threshold=0.1)

    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    # format predictions
    bbs = bbs.tolist()
    confs = confs.tolist()
    labels = labels.tolist()

    # save predictions and targets
    if len(bbs) == 0:
        pred_df_i = pd.DataFrame({
            'filename': image_infos[i],
            'file_id': image_infos[i][:-4],
            'class_name': 'empty',
            'confidence': 1,
            'bbox': [(0, 0, w, h)]
        })
    else:
        pred_df_i = pd.DataFrame({
            'filename': image_infos[i],
            'file_id': image_infos[i][:-4],
            'class_name': [target2label[a] for a in labels],
            'confidence': confs,
            'bbox': bbs
        })
    tar_df_i = pd.DataFrame({
        'filename': image_infos[i],
        'file_id': image_infos[i][:-4],
        'class_name': dfi['LabelName'].tolist(),
        'bbox': tbs.tolist()
    })
    pred_df.append(pred_df_i)
    target_df.append(tar_df_i)

# concatenate preds and targets into dfs
pred_df = pd.concat(pred_df).reset_index(drop=True)
target_df = pd.concat(target_df).reset_index(drop=True)

# save prediction and target dfs to csv
target_df.to_csv(output_path + "target_df.csv")
pred_df.to_csv(output_path + "pred_df.csv")

# define format to read bboxes
# use 'csv' if reloading bboxes from csv file; use 'env' if working with direct model output
format = 'csv'

if format == 'csv':
    pred_df = pd.read_csv(output_path + "pred_df.csv")
    target_df = pd.read_csv(output_path + "target_df.csv")

preds, targets = prepare_results()

# initialize metric
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
metric.update(preds, targets)
results = metric.compute()

# convert results to dataframe
results_df = pd.DataFrame({k: np.array(v) for k, v in results.items()}).reset_index().rename(columns={"index": "target"})

# add F1 score to results
results_df['f1_score'] = 2 * ((results_df['map_per_class'] * results_df['mar_100_per_class']) /
                              (results_df['map_per_class'] + results_df['mar_100_per_class']))

# Add class names to results
results_df['class_name'] = results_df['target'].map(target2label)
results_df = results_df.drop(['target'], axis = 1)

# save results df to csv
results_df.to_csv(output_path + "results_df.csv")

#######
## -- Save Model
#######


# Save model weights for loading into R package
path2weights = output_path + cnn_backbone + "_" + num_classes + "classes_weights_cpu.pth"
torch.save(dict(model.to(device='cpu').state_dict()), f=path2weights)

# save model architecture for loading into R package
model.eval()
s = torch.jit.script(model.to(device='cpu'))
torch.jit.save(s, output_path + cnn_backbone + "_" + num_classes + "classes_Arch_cpu.pt")
