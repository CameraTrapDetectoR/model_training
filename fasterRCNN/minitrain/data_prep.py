"""
supporting functions for data augmentation in minitraining
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch

# train_transform = A.Compose([
#     # A.HorizontalFlip(p=0.5),
#     # A.Affine(rotate=(-30, 30), fit_output=True, p=0.3),
#     # A.Affine(shear=(-20, 20), fit_output=True, p=0.3),
#     # A.RandomBrightnessContrast(p=0.3),
#     # A.HueSaturationValue(p=0.3), # loop through hue_sat grid for all parameter values
#     # A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
#     ToTensorV2()
# ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
# )
#
# # define grids of augmentation options
# transform_grid = ['none', 'horizontal_flip', 'rotate', 'shear', 'brightness_contrast',
#                   'hue_sat_value', 'safe_bbox_crop', 'affine_only', 'flip_crop', ]
# define data augmentation pipelines

# grids of augmentation parameter values
# affine_grid = np.array([(x, y) for x in range(0, 35, 10) for y in range(0, 35, 10)]).transpose()
# hue_sat_grid = list(range(0, 35, 10))

def train_augmentations(w, h, transforms):
    """
    return different data augmentation pipelines based on transformations

    :param w: image width
    :param h: image height
    :param transforms: data augmentations

    :return: training augmentation pipeline

    """
    # single transforms
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

# Create PyTorch dataset
# TODO figure out how to import dataset from here so it's not on main page
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