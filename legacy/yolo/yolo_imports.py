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
import torch.nn as nn

# import deploy/eval libs
from torchvision.transforms.functional import to_pil_image
import random
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes

# YOLO test imports
from IPython import display
from IPython.display import clear_output
from pathlib import Path