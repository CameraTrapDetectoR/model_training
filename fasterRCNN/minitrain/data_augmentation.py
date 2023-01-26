"""
supporting functions for data augmentation in minitraining
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# t_pipeline = A.Compose([
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
        t_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'rotate':
        t_pipeline = A.Compose([
            A.Affine(rotate=(-30, 30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'shear':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'brightness_contrast':
        t_pipeline = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'hue_sat_value':
        t_pipeline = A.Compose([
            A.HueSaturationValue(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'safe_bbox_crop':
        t_pipeline = A.Compose([
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'flip_crop':
        t_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'affine':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'affine_sat':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.HueSaturationValue(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'affine_contrast':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'affine_crop':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

    if transforms == 'affine_sat_contrast':
        t_pipeline = A.Compose([
            A.Affine(shear=(-30, 30), rotate=(-30,30), fit_output=True, p=0.4),
            A.HueSaturationValue(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return t_pipeline

