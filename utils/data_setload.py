# Functions to set up data

import os
from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import torch.cuda
import numpy as np
import cv2

import pandas as pd
from collections import Counter

from torchvision.transforms import ToTensor

# Create PyTorch dataset
class DetectDataset(Dataset):
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

    def __init__(self, df, image_dir, w, h, label2target, transform):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h
        self.label2target = label2target
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
            'labels': torch.tensor([self.label2target[i] for i in labels]).long()
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


def get_class_weights(train_df, model_type):
    """
    introduce class weights to oversample minority classes
    and avoid overfitting majroity classes
    :param train_df: train df
    :param model_type: model_type
    :return: weighted random sampler to pass to dataloader
    """
    
    # collect class counts in a dataframe
    s = dict(Counter(train_df['LabelName']))
    sdf = pd.DataFrame.from_dict(s, orient='index').reset_index()
    sdf.columns = ['LabelName', 'counts']

    # take the inverse to define class weights; smaller counts -> higher weights
    ## perform additional oversampling for pig_only model
    if model_type == 'pig_only':
        sdf['weights'] = [0.5, 0.5]
    else:
        sdf['weights'] = 1/sdf['counts']

    swts = dict(zip(sdf.LabelName, sdf.weights))
    train_unique = train_df.drop_duplicates(subset='filename', keep='first')

    # assign a weight to each image
    sample_weights = train_unique.LabelName.map(swts).tolist()

    # load weighted random sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_unique), replacement=True)

    return sampler

# Create PyTorch dataset for unlabeled images
class EvalDataset(Dataset):
    """
    Builds dataset for images without labels.
    DF must include: filename containing pathway to individual images;
    Images are resized, channels converted, and converted to tensors.
    """

    def __init__(self, df, image_dir, w, h):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h

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
        # convert image to tensor
        img = ToTensor(img)
        return img

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)
