"""
Script  to train updated CameraTrapDetectoR model
"""

## Imports
import os
import torch
from utils.data_process import class_range
from utils.data_process import existing_images
from utils.data_process import format_vars
from utils.dicts import spec_dict
import pandas as pd
import numpy as np



# Set location
if os.name == 'posix':
    local = False
else:
    local = True

# Set paths
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/trainimages"
    os.chdir('/project/cameratrapdetector')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set model type: options = 'general', 'family', 'species', 'pig_only'
model_type = 'species'

max_per_class, min_per_class = class_range(model_type)

# load annotations
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")

# prune df to existing images
df = existing_images(df, IMAGE_ROOT)

# format df
df = format_vars(df)

# create dictionary - create the dict corresponding to model type
df, label2target, columns2stratify = spec_dict(df, max_per_class, min_per_class)

# reverse dictionary to read into pytorch
target2label = {t: l for l, t in label2target.items()}




