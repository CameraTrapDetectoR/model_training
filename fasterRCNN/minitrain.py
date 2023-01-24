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

from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import convnext_small, convnext_base, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights
from torchvision.models import efficientnet_b4, efficientnet_v2_m, EfficientNet_B4_Weights, EfficientNet_V2_M_Weights

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

# define data augmentation pipelines

#TODO determine whether grids need to be same length?
affine_grid = np.array([(x, y) for x in range(0, 35, 10) for y in range(0, 35, 10)]).transpose()
hue_sat_grid = list(range(0, 35, 10))

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-affine_grid[0, i], affine_grid[0, i]), fit_output=True, p=0.3), # loop through values of affine_grid[0]
    A.Affine(shear=(-affine_grid[1, i], affine_grid[0, i]), fit_output=True, p=0.3), # loop through values of affine_grid[1]
    A.RandomBrightnessContrast(brightness_limit= affine_grid[0, -i],
                               contrast_limit = affine_grid[1, -i], True, p=0.3), # loop (backwards?) through affine_grid for brightness_limit and contrast_limit
    A.HueSaturationValue(p=0.3), # loop through hue_sat grid for all parameter values
    # TODO: define height, width outside function and change these values to variables
    A.RandomSizedBBoxSafeCrop(height=307, width=408, erosion_rate=0.2, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)
val_transform = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)

# Create PyTorch dataset
class DetectDataset(torch.utils.data.Dataset):
    """
    Builds dataset with images and their respective targets, bounding boxes and class labels.
    DF must include: filename containing pathway to individual images; bbox ccordinates in format proportional to
    image size (i.e. all bbox coordinates [0,1]) with xmin, ymin corresponding to upper left corner and
    xmax, ymax corresponding to lower right corner.
    Images are resized, channels converted, and augmented according to data augmentation pipelines defined below.
    Bboxes also undergo corresponding data augmentation.
    Each filename corresponds to a 'target' dict of bboxes and labels.
    Images and targets are returned as Tensors.
    """

    def __init__(self, df, image_dir, w, h, transform):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h
        self.transform = transform

    def __getitem__(self, item):
        # create image id
        image_id = self.image_infos[item]
        # create full path to open each image file
        img_path = os.path.join(self.image_dir, image_id).replace("\\", "/")
        # open image
        img = cv2.imread(img_path)
        # reformat color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize image so bboxes can also be converted
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.
        # img = Image.open(img_path).convert("RGB").resize((self.w, self.h), resample=Image.Resampling.BILINEAR)
        # img = np.array(img, dtype="float32")/255.
        # filter df rows for img
        df = self.df
        data = df[df['filename'] == image_id]
        # extract label names
        labels = data['LabelName'].values.tolist()
        # extract bbox coordinates
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute values for model input
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        # convert coordinates to list
        boxes = data.tolist()
        # convert bboxes and labels to a tensor dictionary
        target = {
            'boxes': boxes,
            'labels': torch.tensor([label2target[i] for i in labels]).long()
        }
        # apply data augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, bboxes=target['boxes'], labels=labels)
            img = augmented['image']
            target['boxes'] = augmented['bboxes']
        target['boxes'] = torch.tensor(target['boxes']).float()  # ToTensorV2() isn't working on bboxes
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)

# set image size grid
#TODO: determine if aspect ratio needs to change depending on model backbone
w_grid = [408, 816, 1224, 1632, 2040]
h_grid = [307, 614, 921, 1228, 1535]

# load pytorch datasets
#TODO: loop through height/width grids for image size
# include a training run where train_ds transform=val_transform
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=w_grid, h=h_grid, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=w_grid, h=h_grid, transform=val_transform)

# generate smaller anchor boxes:
# TODO: think about adjusting anchor sizes depending on input image size
anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

# feature map to perform RoI cropping
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# define model
def get_model(cnn_backbone, num_classes):
    # generate smaller anchor boxes:
    # TODO: think about adjusting anchor sizes depending on input image size
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

    # feature map to perform RoI cropping
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', rpn_anchor_generator=anchor_gen)
        #TODO: debug this code
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    if cnn_backbone == 'vgg16':
        backbone = vgg16_bn(weights='DEFAULT').features
        backbone.out_channels = 512
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'conv_s':
        backbone = convnext_small(weights='DEFAULT').features
        #TODO: determine which ConvNext model to use
        backbone.out_channels = 768
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'conv_b':
        backbone = convnext_base(weights='DEFAULT').features
        #TODO: determine which ConvNext model to use
        backbone.out_channels = 1024
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'eff_b4':
        backbone = efficientnet_b4(weights='DEFAULT').features
        #TODO: verify out channels
        backbone.out_channels = 448
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'eff_v2m':
        backbone = efficientnet_v2_m(weights='DEFAULT').features
        #TODO: determine which ConvNext model to use
        backbone.out_channels = 512
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'swin_s':
        backbone = swin_s(weights='DEFAULT').features
        backbone.out_channels = 768
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'swin_b':
        backbone = swin_b(weights='DEFAULT').features
        backbone.out_channels = 1024
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

# backbone grid
backbone_grid = ['resnet', 'vgg16', 'conv_s', 'conv_b', 'eff_b4', 'eff_v2m', 'swin_s', 'swin_b']

# load model
#TODO loop through backbone grid
model = get_model(backbone_grid[i], num_classes).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# optimizer grid
optim_dict = ["SGD", "Adam"]

# starting learning rate grid
lr_grid = [0.001, 0.005, 0.01, 0.05, 0.10]

# weight decay grid
wd_grid = [0, 0.0001, 0.0005, 0.001, 0.005]

optim_grid = np.array([(x, y) for x in lr_grid for y in wd_grid]).transpose()

# load optimizer
#TODO: loop through optim_grid
if optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=optim_grid[0], momentum=momentum, weight_decay=optim_grid[1])
    return optimizer
if optim=="Adam":
    optimizer = torch.optim.Adam(params=params, lr=optim_grid[0], weight_decay=optim_grid[1])


# -- Data Processing / Augmentation Parameters

# weighted random sampling: research a few different options for 3-4 total including no oversampling
# image size: need to confirm if image size must be determined by backbone
# image resolution - signal to noise ratio


# -- Model backbones
#TODO: read papers from pytoch pretrained models to determine which ones to try

# resnet - as baseline comparable to previous versions
# convnext - any version
# vgg16 - may have too many parameters, but was the backbone used in the Faster-RCNN paper
# efficientnet - try efficientnetV2-sm, efficientnet-b4
# swin transformer


# -- hyperparameters
# starting learning rate
# weight decay
# momentum (not applicable to all optimzers)
# optimizer: SGD, Adam
# effective batch size