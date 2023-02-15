# functions to set model hyperparameters

from torchvision.models.detection.anchor_utils import AnchorGenerator


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

    return anchor_gen


# obtain current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']