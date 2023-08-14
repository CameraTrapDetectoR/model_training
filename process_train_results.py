###############
### Process Results from trainCTD.py
###############


## Project Setup

import os
import torch
import pandas as pd


# Set location
if os.name == 'posix':
    local = False
else:
    local = True

# Set paths
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/trainimages"
    os.chdir('/project/cameratrapdetector')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Extract completed results from HPC training
results_path = './output/general_fasterRCNN_resnet_20230607_0929'

checkpoint_path = results_path + "/checkpoint_50epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

results_df = checkpoint['results_df']
results_df.to_csv(results_path + '/fam_v2_test_results_50epochs.csv', index=False)

# extract and format pred_df, target_df
pred_df = checkpoint['pred_df']
pred_df.to_csv(results_path + '/pred_df_50epochs.csv', index=False)

test_df = pd.read_csv(results_path + "/test_df.csv")
test_df = test_df.assign(bbox="[" + test_df['XMin'].astype(str) + ", " + \
                              test_df['YMin'].astype(str) + ", " + \
                              test_df['XMax'].astype(str) + ", " + \
                              test_df['YMax'].astype(str) + "]")
target_df = pd.DataFrame({
    'filename': test_df['filename'],
    'class_name': test_df['class'],
    'bbox': test_df['bbox']
})
target_df.to_csv(results_path + "/target_df.csv")

# get average training time per epoch
from datetime import datetime

train_time = checkpoint['training_time']
train_times = pd.Series([datetime.strptime(t, '%H:%M:%S.%f').time() for t in train_time])

# TODO: troubleshoot how to calculate mean(train_times)