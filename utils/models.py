# function to load different models

import torch
import mmdet
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import convnext_small, convnext_base, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights
from torchvision.models import efficientnet_b4, efficientnet_v2_m, EfficientNet_B4_Weights, EfficientNet_V2_M_Weights
from torchvision.ops import MultiScaleRoIAlign

def get_model(cnn_backbone, num_classes, anchor_gen):


    # feature map to perform RoI cropping
    #roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        # update anchor boxes
        model.rpn.anchor_generator = anchor_gen
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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

