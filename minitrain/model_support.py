"""
functions to support model functionality for mini train sessions
"""

import torch
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import convnext_small, convnext_base, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from torchvision.models import swin_s, swin_b, Swin_S_Weights, Swin_B_Weights
from torchvision.models import efficientnet_b4, efficientnet_v2_m, EfficientNet_B4_Weights, EfficientNet_V2_M_Weights


# define model
#TODO: need to research specific implementation of each backbone onto Faster-RCNN
def get_model(cnn_backbone, num_classes):
    """
    function to load Faster-RCNN model with a specified backbone
    :param cnn_backbone: options from backbone_grid identify different CNN backbone architectures
    to load underneath the region proposal network
    :param num_classes: number of classes in the model
    :return: loaded model
    """
    # generate smaller anchor boxes:
    # TODO: think about adjusting anchor sizes depending on input image size or model backbone
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

    # feature map to perform RoI cropping
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
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

# write arguments to text file
def write_args(cnn_backbone, w, h, transforms, anchor_sizes, batch_size, optim, lr, wd, lr_scheduler, output_path):
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
    model_args = {'backbone': cnn_backbone,
                  'image width': w,
                  'image height': h,
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
def create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes,label2target):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch + 1,
                  'lr': current_lr,
                  'scheduler': lr_scheduler.state_dict(),
                  'loss_history': loss_history,
                  'best_loss': best_loss,
                  'model_type': model_type,
                  'num_classes': num_classes,
                  'label2target': label2target}
    return checkpoint


def save_checkpoint(checkpoint, checkpoint_file):
    print(" Saving model state")
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(checkpoint_file):
    print(" Loading saved model state")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    best_loss = checkpoint['best_loss']
    model_type = checkpoint['model_type']
    label2target = checkpoint['label2target']
    return model, optimizer, lr_scheduler, epoch, loss_history, best_loss, model_type, label2target