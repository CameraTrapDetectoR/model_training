# functions to set model hyperparameters

from torchvision.models.detection.anchor_utils import AnchorGenerator
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# define anchor boxes based on image size
def get_anchors(h):
    """
    define anchor boxes based on image size
    :param h: image height
    :return: anchor box generator
    """
    # use smaller anchor boxes for smaller images
    if h < 512:
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    else:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

    return anchor_sizes, anchor_gen


# obtain current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def data augmentation pipelines
def get_transforms(transforms):
    #TODO: update so transforms is a list
    """
    Return data augmentation pipelines
    :param transforms:
    :return: train_transform, val_transform
    """

    if transforms == 'shear, rotate, huesat, brightcont, safecrop, hflip':
        train_transform = A.Compose([A.Affine(shear=(-30, 30), fit_output=True, p=0.3),
                                     A.Affine(rotate=(-30, 30), fit_output=True, p=0.3),
                                     A.HueSaturationValue(p=0.4),
                                     A.RandomBrightnessContrast(p=0.4),
                                     A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.4),
                                     A.HorizontalFlip(p=0.4),
                                     ToTensorV2()],
                                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )
        val_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )

    if transforms == 'none':
        train_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )
        val_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )

    if transforms == 'hflip':
        train_transform = A.Compose([A.HorizontalFlip(p=0.4),
                                     ToTensorV2()],
                                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )
        val_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )

    if transforms == 'safecrop':
        train_transform = A.Compose([A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.4),
                                     ToTensorV2()],
                                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )
        val_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )

    if transforms == 'safecrop, hflip':
        train_transform = A.Compose([A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.2, p=0.4),
                                     A.HorizontalFlip(p=0.4),
                                     ToTensorV2()],
                                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )
        val_transform = A.Compose([ToTensorV2()],
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), )


    return train_transform, val_transform