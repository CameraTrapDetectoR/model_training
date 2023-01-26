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

from fasterRCNN.minitrain.data_augmentation import train_augmentation

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
train_transform = train_augmentations(w=w, h=h, transforms=transforms)

# define validation augmentation pipeline
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


# load pytorch datasets
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=w, h=h, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=w, h=h, transform=val_transform)

# generate smaller anchor boxes:
# TODO: think about adjusting anchor sizes depending on input image size
anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

# feature map to perform RoI cropping
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# define model
def get_model(cnn_backbone, num_classes):
    """
    function to load Faster-RCNN model with a specified backbone
    :param cnn_backbone: options from backbone_grid identify different CNN backbone architectures
    to load underneath the region proposal network
    :param num_classes: number of classes in the model
    :return: loaded model
    """

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
backbone = backbone_grid[-1]

# load model
#TODO loop through backbone grid
model = get_model(cnn_backbone=backbone, num_classes).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# optimizer options
optim_dict = ["SGD", "Adam"]

# starting learning rate grid
lr_grid = [0.001, 0.005, 0.01, 0.05, 0.10]

# weight decay grid
wd_grid = [0, 0.0001, 0.0005, 0.001, 0.005]

# load optimizer
optim = "Adam"
lr = lr_grid[2] #TODO: loop through lr_grid
wd = wd_grid[2] #TODO: loop through wd_grid
momentum = 0.9 # only need this for optimizer = SGD

# load optimizer
if optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
if optim=="Adam":
    optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=wd)

# set learning rate scheduler
#TODO: explore other lr schedulers
lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# set number of epochs
# TODO implement early stopping and increase number of epochs
# look at end of training loop and set criteria based on change in val_loss over epochs, or implement function from skorch, pytorch-lightning, etc.
n_epochs = 25

# set batch_size grid
# TODO: need to test gradient accumulation and available memory in Ceres/Atlas
batch_grid = [8, 16, 32, 64]
batch_size = batch_grid[0]

# set up class weights
# TODO: introduce imbalance into dataset to create different weights
s = dict(Counter(train_df['LabelName'])
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
output_path = "./minitrain/output/" + "fasterRCNN_" + backbone + "_" + time.strftime("%Y%m%d")\
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
with open(output_path + 'model_args.txt', 'w') as f:
    for key, value in model_args.items():
        f.write('%s:%s\n' % (key, value))
