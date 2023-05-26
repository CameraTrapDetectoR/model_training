"""
Script to deploy CameraTrapDetectoR model on out of sample data
"""

import os
import torch
from PIL import ImageFile
import numpy as np
import albumentations as A
from models.model_inference import plot_losses
from albumentations.pytorch.transforms import ToTensorV2
from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn
from tqdm import tqdm
from torchvision.ops import nms
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import ToTensor
import cv2


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

# set path to model run being evaluated
output_path = "./output/fasterRCNN_resnet_20230221_1612/"

# open model arguments file
with open(output_path + 'model_args.txt') as f:
    model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
model_args['image width'] = int(model_args['image width'])
model_args['image height'] = int(model_args['image height'])
model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))
cnn_backbone = model_args['backbone']

# load image info
#TODO: make this easily changeable

# set image directory
IMAGE_PATH = IMAGE_ROOT + '/Yancy/Control/NFS02'
# create holder for file names
image_infos = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_PATH) for f in fn 
               if os.path.splitext(f)[1].lower() == '.jpg']

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

#######
## -- Evaluate Model on Test Data
#######

# create placeholders for targets and predictions
pred_df = []

# deploy model on test images
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
        # set iou threshold low since training data does not contain overlapping ground truth boxes
        ixs = nms(bbs, confs, iou_threshold=0.1)

        bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

        # format predictions
        bbs = bbs.tolist()
        confs = confs.tolist()
        labels = labels.tolist()

        # save predictions and targets
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
        # tar_df_i = pd.DataFrame({
        #     'filename': image_infos[i],
        #     'file_id': image_infos[i][:-4],
        #     'class_name': dfi['LabelName'].tolist(),
        #     'bbox': tbs.tolist()
        # })
        pred_df.append(pred_df_i)
        # target_df.append(tar_df_i)


# concatenate preds and targets into dfs
pred_df = pd.concat(pred_df).reset_index(drop=True)
# target_df = pd.concat(target_df).reset_index(drop=True)

# save prediction and target dfs to csv
# target_df.to_csv(output_path + "target_df.csv")
pred_df.to_csv(IMAGE_PATH + "pred_df.csv")

#######
## -- Calculate Evaluation Metrics
#######

# # extract predicted bboxes, confidence scores, and labels
# preds = []
# targets = []

# # create list of dictionaries for targets, preds for each image
# for i in tqdm(range(len(image_infos))):
#     # extract predictions and targets for an image
#     p_df = pred_df[pred_df['filename'] == image_infos[i]]
#     t_df = target_df[target_df['filename'] == image_infos[i]]

#     # # format boxes based on input
#     # if format == 'csv':
#     #     # pred detections
#     #     pred_boxes = [box.strip('[').strip(']').strip(',') for box in p_df['bbox']]
#     #     pred_boxes = np.array([np.fromstring(box, sep=', ') for box in pred_boxes])
#     #     # ground truth boxes
#     #     target_boxes = [box.strip('[').strip(']').strip(', ') for box in t_df['bbox']]
#     #     target_boxes = np.array([np.fromstring(box, sep=', ') for box in target_boxes])
#     # if format == 'env':
#     #     # pred detections
#     #     pred_boxes = [box for box in p_df['bbox']]
#     #     # ground truth boxes
#     #     target_boxes = [box for box in t_df['bbox']]

#     # pred boxes
#     pred_boxes = [box for box in p_df['bbox']]
#     # ground truth boxes
#     target_boxes = [box for box in t_df['bbox']]

#     # format scores and labels
#     pred_scores = p_df['confidence'].values.tolist()
#     pred_labels = p_df['class_name'].map(label2target)
#     target_labels = t_df['class_name'].map(label2target)

#     # convert preds to dictionary of tensors
#     pred_i = {
#         'boxes': torch.tensor(pred_boxes),
#         'scores': torch.tensor(pred_scores),
#         'labels': torch.tensor(pred_labels.values)
#     }

#     # convert targets to tensor dictionary
#     target_i = {
#         'boxes': torch.tensor(target_boxes),
#         'labels': torch.tensor(target_labels.values)
#     }

#     # add current image preds and targets to dictionary list
#     preds.append(pred_i)
#     targets.append(target_i)

# # initialize metric
# metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
# metric.update(preds, targets)
# results = metric.compute()

# # convert results to dataframe
# results_df = pd.DataFrame({k: np.array(v) for k, v in results.items()}).reset_index().rename(columns={"index": "target"})

# # add F1 score to results
# results_df['f1_score'] = 2 * ((results_df['map_per_class'] * results_df['mar_100_per_class']) /
#                               (results_df['map_per_class'] + results_df['mar_100_per_class']))

# # Add class names to results
# results_df['class_name'] = results_df['target'].map(target2label)
# results_df = results_df.drop(['target'], axis = 1)

# # save results df to csv
# results_df.to_csv(output_path + "results_df.csv")

# # re-initiate the model on CPU so it can be loaded into R package
# device = 'cpu'
# model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen)
# model.load_state_dict(checkpoint['state_dict'])
# model.to(device)

# # Save model weights for loading into R package
# path2weights = output_path + cnn_backbone + "_" + str(num_classes) + "classes_weights_cpu.pth"
# torch.save(dict(model.to(device='cpu').state_dict()), f=path2weights)

# # save model architecture for loading into R package
# model.eval()
# s = torch.jit.script(model.to(device='cpu'))
# torch.jit.save(s, output_path + cnn_backbone + "_" + str(num_classes) + "classes_Arch_cpu.pth")
