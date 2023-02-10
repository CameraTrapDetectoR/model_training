"""
script to test model training with a mini dataset across grids of hyperparameters, model backbones,
and other coding choices
"""


import copy
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.cuda
from PIL import ImageFile
import time
import random
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import convnext_small, convnext_base, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights
from torchvision.models import efficientnet_b4, efficientnet_v2_m, EfficientNet_B4_Weights, EfficientNet_V2_M_Weights
from tqdm import tqdm
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from datetime import timedelta

# from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
    os.chdir('/project/cameratrapdetector')


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set model type
model_type = 'species'

# load df
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")
df['bbox.area'] = (df['XMax'] - df['XMin']) * (df['YMax'] - df['YMin'])

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

# set boolean to create unbalanced dataset
unbalanced = True

# option to create imbalance in the dataset
if unbalanced:
    # save copy of original df
    df_filename_org = df_unique_filename
    # randomly select class to downsample
    downsize = random.sample(df.LabelName.unique().tolist(), k=1)[0]
    # separate images from other classes
    df_bal = df_unique_filename[df_unique_filename['LabelName'] != downsize]
    # downsample images of that class to 10% of original size
    df_downsize = df_unique_filename[df_unique_filename['LabelName'] == downsize]
    df_downsize = df_downsize.sample(frac=0.1)
    # join dfs into new df
    df_unique_filename = pd.concat([df_bal, df_downsize])

# perform train-test split
train_ids, val_ids = train_test_split(df_unique_filename['filename'], shuffle=True,
                                        stratify=df_unique_filename[columns2stratify],
                                        test_size=0.3, random_state=22)
train_df = df[df['filename'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['filename'].isin(val_ids)].reset_index(drop=True)

# Review splits
# Counter(train_df['LabelName'])
# Counter(val_df['LabelName'])

# set image size grid
#TODO: determine if aspect ratio needs to change depending on model backbone
w_grid = [408, 510, 612, 816, 1224]
h_grid = [307, 384, 460, 614, 921]

# Define width and height OUTSIDE any functions
w = w_grid[-1]
h = h_grid[-1]

# define data augmentation grid
#TODO: add image quality compression here?
transform_grid = ['none', 'horizontal_flip', 'rotate', 'shear', 'brightness_contrast',
                  'hue_sat_value', 'safe_bbox_crop', 'flip_crop', 'affine', 'affine_sat',
                  'affine_contrast', 'affine_crop', 'affine_sat_contrast']

# define data augmentation for train_df
def train_augmentations(w, h, transforms):
    """
    return different data augmentation pipelines based on transformations

    :param w: image width
    :param h: image height
    :param transforms: data augmentations

    :return: training augmentation pipeline

    """
    # single transforms
    if transforms == 'none':
        train_transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'horizontal_flip':
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'rotate':
        train_transform = A.Compose([
            A.Affine(rotate=(-30, 30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'shear':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'brightness_contrast':
        train_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'hue_sat_value':
        train_transform = A.Compose([
            A.HueSaturationValue(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'safe_bbox_crop':
        train_transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'flip_crop':
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'affine':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'affine_sat':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.HueSaturationValue(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'affine_contrast':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'affine_crop':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

    if transforms == 'affine_sat_contrast':
        train_transform = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.HueSaturationValue(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform

# define augmentations to run this training round
transforms = transform_grid[-1] # TODO loop through transform_grid

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
            img = (augmented['image'])
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

# backbone grid
#TODO revisit these backbone choices; troubleshoot connecting CNN to RPN
backbone_grid = ['resnet', 'resnet_nofeatures', 'vgg16', 'conv_s', 'conv_b',
                 'eff_b4', 'eff_v2m', 'swin_s', 'swin_b']
cnn_backbone = backbone_grid[0]

# generate anchor boxes based on image size:
# TODO: think about adjusting anchor sizes depending on model backbone
anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) # ((16,), (32,), (64,), (128,), (256,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

# feature map to perform RoI cropping
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# define model
#TODO: need to research specific implementation of each backbone onto Faster-RCNN
def get_model(cnn_backbone, num_classes, anchor_gen):
    """
    function to load Faster-RCNN model with a specified backbone
    :param cnn_backbone: options from backbone_grid identify different CNN backbone architectures
    to load underneath the region proposal network
    :param num_classes: number of classes in the model
    :return: loaded model
    """

    # feature map to perform RoI cropping
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        # update anchor boxes
        model.rpn.anchor_generator = anchor_gen
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
        backbone.out_channels = 768
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'conv_b':
        backbone = convnext_base(weights='DEFAULT').features
        backbone.out_channels = 1024
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'eff_b4':
        backbone = efficientnet_b4(weights='DEFAULT').features
        backbone.out_channels = 448
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_gen,
                           box_roi_pool=roi_pooler)
        return model

    if cnn_backbone == 'eff_v2m':
        backbone = efficientnet_v2_m(weights='DEFAULT').features
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

# load model
#TODO load model w/o weights and try again
model = get_model(cnn_backbone=cnn_backbone, num_classes=num_classes, anchor_gen=anchor_gen).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# starting learning rate grid
lr_grid = [0.001, 0.005, 0.01, 0.05, 0.10]

# weight decay grid
wd_grid = [0, 0.0005, 0.001, 0.005, 0.01]


# define optimizer method
def get_optimizer(optim, params, lr, wd, momentum):
    if optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        return optimizer
    if optim == "Adam":
        optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=wd)
        return optimizer
    if optim == "AdamW":
        optimizer = torch.optim.AdamW(params=params, lr=lr, weight_decay=wd)
        return optimizer



# set number of epochs
# TODO implement early stopping and increase number of epochs?
# look at end of training loop and set criteria based on change in val_loss over epochs, or implement function from skorch, pytorch-lightning, etc.
num_epochs = 50

# set batch_size grid
# TODO: need to test gradient accumulation and available memory in Ceres/Atlas
batch_grid = [16, 32, 64, 80]
batch_size = batch_grid[1]
# make copy of batch_size for gradient accumulation
batch_size_org = batch_size

# boolean to use gradient accumulation
use_grad = True

# set denominator if using gradient accumulation
if use_grad:
    # set number of gradients to accumulate before updating weights
    grad_accumulation = 8
    # effective batch size = batch_size * grad_accumulation
    batch_size = batch_size_org // grad_accumulation


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

# write arguments to text file
def write_args(cnn_backbone, w, h, unbalanced, transforms, anchor_sizes, batch_size, optim, lr, wd, lr_scheduler, output_path):
    """
    write txt file to output dir that has run values for changeable hyperparameters:
    - model backbone
    - image size
    - data augmentations and their ranges
    - imbalance
    - anchor box sizes (may not change this)
    - optimizer
    - starting learning rate
    - weight decay
    - learning rate scheduler and parameters
    :return: txt file containing all values of changing arguments per model run
    """

    # collect arguments in a dict
    model_args = {'backbone': cnn_backbone,
                  'image width': w,
                  'image height': h,
                  'unbalanced': unbalanced,
                  'data augmentations': transforms,
                  'anchor box sizes': anchor_sizes,
                  'batch size': batch_size,
                  'optimizer': optim,
                  'starting lr': lr,
                  'weight decay': wd,
                  'lr_scheduler': lr_scheduler.__class__
                  }

    # write args to text file
    with open(output_path + '/model_args.txt', 'w') as f:
        for key, value in model_args.items():
            f.write('%s:%s\n' % (key, value))

# obtain current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# define checkpoint functions
def create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes,label2target, training_time):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch + 1,
                  'lr': current_lr,
                  'scheduler': lr_scheduler.state_dict(),
                  'loss_history': loss_history,
                  'best_loss': best_loss,
                  'model_type': model_type,
                  'num_classes': num_classes,
                  'label2target': label2target,
                  'training_time': training_time}
    return checkpoint

def save_checkpoint(checkpoint, checkpoint_file):
    print(" Saving model state")
    torch.save(checkpoint, checkpoint_file)

# define optimizer options
optim_dict = ["SGD", "Adam", "AdamW"]

# loop training through different optimizers
for i in range(len(optim_dict)):
    # set starting lr and wd
    lr = lr_grid[2]  # TODO: loop through lr_grid
    wd = wd_grid[2]  # TODO: loop through wd_grid
    momentum = 0.9  # only need this for optimizer = SGD

    # define optimizer
    optim = optim_dict[i]
    optimizer = get_optimizer(optim, params, lr, wd, momentum)

    # set learning rate scheduler
    #TODO: explore other lr schedulers
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # make output directory and filepath
    #TODO: current format depends on models with the same backbone being initiated on different days
    output_path = "./minitrain/output/" + "fasterRCNN_" + cnn_backbone + "_" + \
                  time.strftime("%Y%m%d") + "_" + time.strftime("%H%M") + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # write test arguments to file
    write_args(cnn_backbone, w, h, unbalanced, transforms, anchor_sizes, batch_size_org, optim, lr, wd, lr_scheduler, output_path)

    # write datasets to csv
    train_df.to_csv(output_path + "train_df.csv")
    val_df.to_csv(output_path + "val_df.csv")

    # define starting weights, starting loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # create empty list to save losses
    loss_history = {
        'train': [],
        'val': []
    }

    # create empty list to save training time per epoch
    training_time = []

    # train the model
    for epoch in range(num_epochs):
        # start time
        t_start = time.time()
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

            if use_grad:
                # normalize loss to account for batch accumulation
                loss = loss / grad_accumulation
                # backward pass
                loss.backward()
                # optimizer step every x=grad_accumulation batches
                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

            else:
                # backward pass
                loss.backward()
                # optimizer step
                optimizer.step()
                # reset gradients
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

                if use_grad:
                    # normalize loss based on gradient accumulation
                    val_loss = val_loss / grad_accumulation
                    if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(val_loader)):
                        # reset gradients
                        optimizer.zero_grad()
                        print(f'Batch {batch_idx} / {len(val_loader)} | Val Loss: {val_loss:.4f}')

                else:
                    # reset gradients
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

        # calculate training time per epoch
        elapsed = time.time() - t_start
        elapsed = str(timedelta(seconds=elapsed))
        training_time.append(elapsed)

        # save model state
        checkpoint = create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history,
                                       best_loss, model_type, num_classes, label2target, training_time)
        checkpoint_file = output_path + "checkpoint_" + str(epoch+1) + "epochs.pth"
        save_checkpoint(checkpoint, checkpoint_file)

        print("End model training round")
