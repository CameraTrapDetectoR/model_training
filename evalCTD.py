"""
Script to deploy CameraTrapDetectoR model on out of sample data
"""

import os
import torch
from PIL import ImageFile
import numpy as np
import pandas as pd
import cv2

import json
from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn
from tqdm import tqdm
from torchvision.ops import nms



#######
## -- Prepare System and Data for Model Training
#######

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

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set path to model run being deployed
output_path = "./output/fasterRCNN_resnet_20230221_1612/"

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
checkpoint_path = output_path + "50epochs/checkpoint_50epochs.pth"
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

# set image directory
IMAGE_PATH = IMAGE_ROOT + '/Yancy/Control/NFS12'
# load image names
image_infos = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_PATH) for f in fn if os.path.splitext(f)[1].lower() == '.jpg']


#######
## -- Evaluate Model on Test Data
#######

# create placeholder for predictions
pred_df = []

# TODO: add option here to load partial results and only run images that have not already been run through model
resume_from_checkpoint = False
if resume_from_checkpoint == True:
    # search IMAGE_ROOT for checkpoint file
    # throw error if cannot find file
    # load file and set as pred_df
    f = open(chkpt_pth)
    pred_checkpoint = json.load(f) 
    pred_df = pd.DataFrame.from_dict(pred_checkpoint)
    # filter through image infos and update list to images not in pred_df


else:
    # define checkpoint path
    chkpt_pth = IMAGE_PATH + "_pred_checkpoint.json"

# deploy model
with torch.no_grad():
    model.eval()
    for i in tqdm(range(len(image_infos))):
        # set image path
        img_path = image_infos[i]
        # open image
        img = cv2.imread(img_path)
        # reformat color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize image so bboxes can also be converted
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.
        # convert array to tensor 
        img = torch.from_numpy(img)
        # shift channels to be compatible with model input
        image = img.permute(2, 0, 1)
        image = image.unsqueeze_(0)
        # send input to CUDA if available
        image = image.to(device)

        # run input through the model
        with torch.no_grad():
            output = model(image)[0]
        output = model(image)[0]

        # extract prediction bboxes, labels, scores above score_thresh
        # format prediction data
        bbs = output['boxes'].cpu().detach()
        labels = output['labels'].cpu().detach()
        confs = output['scores'].cpu().detach()

        # id indicies of tensors to include in evaluation
        idx = torch.where(confs > 0.01)

        # filter to predictions that meet the threshold
        bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

        # perform non-maximum suppression on remaining predictions
        ixs = nms(bbs, confs, iou_threshold=0.5)

        bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

        # format predictions
        bbs = bbs.tolist()
        confs = confs.tolist()
        labels = labels.tolist()

        if len(bbs) == 0:
            pred_df_i = pd.DataFrame({
                'filename': image_infos[i],
                'file_id': image_infos[i][:-4],
                'class_name': 'empty',
                'confidence': 1,
                'bbox': [(0, 0, w, h)]
            })
        else:
            pred_df_i = pd.DataFrame({
                'filename': image_infos[i],
                'file_id': image_infos[i][:-4],
                'class_name': [target2label[a] for a in labels],
                'confidence': confs,
                'bbox': bbs
            })

        # add image predictions to existing list
        pred_df.append(pred_df_i)

        # save results every 100 images 
        # TODO troubleshoot this
         if i % 100 == 0:
            # concatenate preds into df
             pred_df = pd.concat(pred_df).reset_index(drop=True)
            # save to json
             pred_df.to_json(path_or_buf=chkpt_pth, orient='records')

# concatenate preds and targets into dfs
pred_df = pd.concat(pred_df).reset_index(drop=True)

# save prediction and target dfs to csv
pred_df.to_csv(IMAGE_PATH + "_pred_df.csv")

# remove checkpoint file
# os.remove(chkpt_pth)


#######
## -- Post Processing
#######

