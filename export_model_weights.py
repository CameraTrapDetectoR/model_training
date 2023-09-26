"""
Export final model weights for deploying in R package
Separate script for after model has been trained and evaluated
"""

# Imports

import os
import torch
from utils import dicts

from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn



# set os path
os.chdir("C:/Users/amira.burns/OneDrive - USDA/Projects/CameraTrapDetectoR")

# set device to cpu for exporting
device = torch.device('cpu')
print(device)

# set path to model run being deployed
output_path = "./output/family_fasterRCNN_resnet_20230606_1122/"

# open model arguments file
with open(output_path + 'model_args.txt') as f:
    model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
model_args['image width'] = int(model_args['image width'])
model_args['image height'] = int(model_args['image height'])
model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))
cnn_backbone = model_args['backbone']

# define image dimensions
w = model_args['image width']
h = model_args['image height']

# load model checkpoint
checkpoint_path = output_path + "checkpoint_50epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# reload anchor generator
anchor_sizes, anchor_gen = get_anchors(h=h)

# initiate model
cnn_backbone = 'resnet'
num_classes = checkpoint['num_classes']
model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen)

# load model weights
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

# # Save model weights for loading into R package

# create folder for model version
model_version = '/family_v2'
os.mkdir(output_path + model_version)
model_folder = output_path + model_version
path2weights = model_folder + "/model_weights.pth"
torch.save(dict(model.to(device='cpu').state_dict()), f=path2weights)

# # save model architecture for loading into R package
model.eval()
s = torch.jit.script(model.to(device='cpu'))
torch.jit.save(s, model_folder + "/model_arch.pt")

# save label encoder for loading into R package
dicts.encode_labels(label2target, model_folder)

## END