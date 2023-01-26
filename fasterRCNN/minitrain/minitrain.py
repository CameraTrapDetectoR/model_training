"""
script to test model training with a mini dataset across grids of hyperparameters, model backbones,
and other coding choices
"""


import copy
from collections import Counter
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math


import torch
import torch.cuda
from PIL import Image, ImageFile, ImageDraw, ImageFont
import time
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pylab as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fasterRCNN.minitrain import data_prep
from fasterRCNN.minitrain import model_support
from fasterRCNN.minitrain.model_support import get_lr, create_checkpoint, save_checkpoint

from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


from torchvision.transforms.functional import to_pil_image
import random
from torchvision.ops import nms, batched_nms
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import matplotlib.pyplot as plt

# determine operating system, for batch vs. local jobs

if os.name == 'posix':
    local = False
else:
    local = True

# set path based on location, local machine or remote batch job
# for IMAGE_ROOT, specify full path to folder where all/only training images are located
if local:
    IMAGE_ROOT = 'C:/Users/Amira.Burns/OneDrive - USDA/Documents/CameraTrapDetectoR_Files/Test Images/minitrain'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/minitrain"
    os.chdir('/projects/cameratrapdetector')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set model type
model_type = 'species'

# load df
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")

# update fox labels
df.filename = df.filename.str.replace("Cascade_Red_Fox", "Red_Fox")

# filter df to images in the minitrain set
extant = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_ROOT) for f in fn]
extant = [x.replace(IMAGE_ROOT + '/', '/').strip('/') for x in extant]
df = df[df['filename'].isin(extant)]

# swap bbox coordinates if needed
df['YMin_org'] = df['YMin']
df['YMax_org'] = df['YMax']
df.drop(['YMax', 'YMin'], axis=1, inplace=True)
df['YMax'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMin_org']), df['YMax_org'])
df['YMin'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMax_org']), df['YMin_org'])

# update column names for common name
df['common.name_org'] = df['common.name']
df['common.name'] = df['common.name.general']

# create dictionary of species labels
label2target = {l: t + 1 for t, l in enumerate(df['common.name'].unique())}

# set background class
label2target['empty'] = 0
background_class = label2target['empty']

# reverse dictionary for pytorch input
target2label = {t: l for l, t in label2target.items()}
pd.options.mode.chained_assignment = None

# standardize label name
df['LabelName'] = df['common.name']

# stratify across species for train/val split
columns2stratify = ['common.name']

# define number of classes
num_classes = max(label2target.values()) + 1

# get list of distinct image filenames
df_unique_filename = df.drop_duplicates(subset='filename', keep='first')

# perform train-test split
train_ids, val_ids = train_test_split(df_unique_filename['filename'], shuffle=True,
                                        stratify=df_unique_filename[columns2stratify],
                                        test_size=0.3, random_state=22)
train_df = df[df['filename'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['filename'].isin(val_ids)].reset_index(drop=True)

# Review splits
Counter(train_df['LabelName'])
Counter(val_df['LabelName'])

# write datasets to csv
train_df.to_csv(output_path + "train_df.csv")
val_df.to_csv(output_path + "val_df.csv")


# set image size grid
#TODO: determine if aspect ratio needs to change depending on model backbone
w_grid = [408, 816, 1224, 1632, 2040]
h_grid = [307, 614, 921, 1228, 1535]

# Define width and height OUTSIDE any functions
w = w_grid[-2]
h = h_grid[-2]

# define data augmentation grid
#TODO: add image quality compression here?
transform_grid = ['none', 'horizontal_flip', 'rotate', 'shear', 'brightness_contrast',
                  'hue_sat_value', 'safe_bbox_crop', 'flip_crop', 'affine', 'affine_sat',
                  'affine_contrast', 'affine_crop', 'affine_sat_contrast']

# define augmentations to run this training round
transforms = transform_grid[0] # TODO loop through transform_grid

# get training augmentation pipeline
train_transform = data_prep.train_augmentations(w=w, h=h, transforms=transforms)

# define validation augmentation pipeline
val_transform = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)

# load pytorch datasets
train_ds = data_prep.DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=w, h=h, transform=train_transform)
val_ds = data_prep.DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=w, h=h, transform=val_transform)

# backbone grid
backbone_grid = ['resnet', 'vgg16', 'conv_s', 'conv_b', 'eff_b4', 'eff_v2m', 'swin_s', 'swin_b']
cnn_backbone = backbone_grid[-2]

# generate smaller anchor boxes:
# TODO: think about adjusting anchor sizes depending on input image size or model backbone
anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

# feature map to perform RoI cropping
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# load model
model = model_support.get_model(cnn_backbone=cnn_backbone, num_classes=num_classes).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# optimizer options
optim_dict = ["SGD", "Adam", "AdamW"]

# starting learning rate grid
lr_grid = [0.001, 0.005, 0.01, 0.05, 0.10]

# weight decay grid
wd_grid = [0, 0.0005, 0.001, 0.005, 0.01]

# load optimizer
optim = "AdamW"
lr = lr_grid[2] #TODO: loop through lr_grid
wd = wd_grid[2] #TODO: loop through wd_grid
momentum = 0.9 # only need this for optimizer = SGD

# load optimizer
optimizer = get_optimizer(optim, params, lr, wd, momentum)

# set learning rate scheduler
#TODO: explore other lr schedulers
lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# set number of epochs
# TODO implement early stopping and increase number of epochs?
# look at end of training loop and set criteria based on change in val_loss over epochs, or implement function from skorch, pytorch-lightning, etc.
n_epochs = 30

# set batch_size grid
# TODO: need to test gradient accumulation and available memory in Ceres/Atlas
batch_grid = [16, 32, 64]
batch_size = batch_grid[2]

# set gradient accumulation
grad_accumulation = 1

# set up class weights
# TODO: introduce imbalance into dataset to create different weights
s = dict(Counter(train_df['LabelName']))
#TODO: is there a more elegant way to execute the next 4-6 lines?
sdf = pd.DataFrame.from_dict(s, orient='index').reset_index()
sdf.columns = ['LabelName', 'counts']
sdf['weights'] = 1/sdf['counts']
swts = dict(zip(sdf.LabelName, sdf.weights))
train_unique = train_df.drop_duplicates(subset='filename', keep='first')
sample_weights = train_unique.LabelName.map(swts).tolist()

# load weighted random sampler
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_unique), replacement=True)

# define dataloaders
train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn,
                          drop_last=True, sampler=sampler)
# do not oversample for validation, just for training
val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)

# make output directory and filepath
#TODO: current format depends on models with the same backbone being initiated on different days
output_path = "./fasterRCNN/minitrain/output/" + "fasterRCNN_" + backbone + "_" + time.strftime("%Y%m%d")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# write txt file
#TODO: put this inside a function or no?

    """
    write txt file to output dir that has run values for changeable hyperparameters:
    - model backbone
    - image size
    - data augmentations and their ranges
    - anchor box sizes (may not change this)
    - optimizer
    - starting learning rate
    - weight decay
    - learning rate scheduler and parameters
    :return: txt file containing all values of changing arguments per model run
    """
# collect arguments in a dict
model_args = {'backbone': backbone,
              'image width': w,
              'image height': h,
              'data augmentations': transforms,
              'anchor box sizes': anchor_sizes,
              'optimizer': optim,
              'starting lr': lr,
              'weight decay': wd,
              'lr_scheduler': lr_scheduler.__class__
              }

# write args to text file
with open(output_path + '/model_args.txt', 'w') as f:
    for key, value in model_args.items():
        f.write('%s:%s\n' % (key, value))

# define starting weights, starting loss
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')

# create empty list to save losses
loss_history = {
    'train': [],
    'val': []
}

# train the model
for epoch in range(num_epochs):
    # set learning rate and print epoch number
    current_lr = get_lr(optimizer)
    print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

    # training pass
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        # send data to device
        images = list(image.to(device) for image in images)
        targets = [{'boxes':t['boxes'].to(device), 'labels':t['labels'].to(device)} for t in targets]

        # forward pass
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        # normalize loss to account for batch accumulation
        loss = loss / grad_accumulation

        # backward pass
        loss.backward()

        # optimizer step every x=grad_accumulation batches
        if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

        # update loss
        running_loss += loss.item()

    # record training loss
    loss_history['train'].append(running_loss/len(train_loader))
    print('train loss: %.6f' % (running_loss / len(train_loader)))

    # validation pass
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            model.train() # obtain losses without defining forward method
            # move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # collect losses
            val_losses = model(images, targets)
            val_loss = sum(loss for loss in val_losses.values())

            # normalize loss based on gradient accumulation
            val_loss = val_loss / grad_accumulation
            if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(val_loader)):
                optimizer.zero_grad()
                print(f'Batch {batch_idx} / {len(val_loader)} | Val Loss: {val_loss:.4f}')

            # update loss
            running_val_loss += float(val_loss)

        # record validation loss
        val_loss = running_val_loss / len(val_loader)
        loss_history['val'].append(running_val_loss / len(val_loader))
        print('val loss: %.6f' % (running_val_loss / len(val_loader)))

    # compare validation loss
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

    # save model state
    checkpoint = create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes, label2target)
    checkpoint_file = output_path + "checkpoint_" + str(epoch+1) + "epochs.pth"
    save_checkpoint(checkpoint, checkpoint_file)
