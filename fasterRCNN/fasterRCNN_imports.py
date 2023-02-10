############################
## - CameraTrapDetectoR
## - Model Imports
############################


# import data processing libs
import copy
from collections import Counter
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# import dataset prep libs
import torch
import torch.cuda
from PIL import Image, ImageFile, ImageDraw, ImageFont
import time
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pylab as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# import model libs
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

# import deploy/eval libs
from torchvision.transforms.functional import to_pil_image
import random
from torchvision.ops import nms, batched_nms
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import matplotlib.pyplot as plt