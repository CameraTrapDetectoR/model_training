# functions to set model hyperparameters


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# define data augmentation for train_df
def augmentations(w, h, transforms):
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
        return train_transform, val_transform

    if transforms == 'horizontal_flip':
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )
        return train_transform, val_transform

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

# define validation transform pipeline
def val_augmentation():
    val_transform = A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    )
    return val_transform