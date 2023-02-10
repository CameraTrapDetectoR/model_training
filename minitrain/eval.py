"""
Test model trainings from the mini dataset
"""

import os
import torch
import pandas as pd
from PIL import ImageFile
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.ops import nms
from tqdm import tqdm
import torch.cuda
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchmetrics.detection.mean_ap import MeanAveragePrecision

if os.name == 'posix':
    local = False
else:
    local = True

# set path based on location, local machine or remote batch job
# for IMAGE_ROOT, specify full path to folder where all/only training images are located
if local:
    IMAGE_ROOT = 'C:/Users/Amira.Burns/OneDrive - USDA/Documents/CameraTrapDetectoR_Files/Test Images/minitrain'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/minitrain"
    os.chdir('/project/cameratrapdetector')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set path to model run being evaluated
output_path = "./minitrain/output/fasterRCNN_resnet_20230206_1142/"

# open model arguments file
with open(output_path + 'model_args.txt') as f:
    model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
model_args['image width'] = int(model_args['image width'])
model_args['image height'] = int(model_args['image height'])
model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))

# load image info
df = pd.read_csv(output_path + "val_df.csv")
image_infos = df.filename.unique()

# define image dimensions
w = model_args['image width']
h = model_args['image height']

# load model checkpoint
checkpoint_path = output_path + "checkpoint_50epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Review loss history
# loss_history = checkpoint['loss_history']
# plot_losses(model_type, loss_history)
#
# # rreview training times
# training_time = checkpoint['training_time']

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# reload anchor generator
anchor_sizes = model_args['anchor box sizes']
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

# define validation augmentation pipeline
val_transform = A.Compose([ToTensorV2()],
                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),)

# initiate model
num_classes = checkpoint['num_classes']
model = fasterrcnn_resnet50_fpn_v2()
# model.rpn.anchor_generator = anchor_gen
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load model weights
model.load_state_dict(checkpoint['state_dict'])
model.to(device)


# Create PyTorch dataset
class DetectDataset(torch.utils.data.Dataset):
    """
    Builds dataset with images and their respective targets, bounding boxes and class labels.
    DF must include: filename containing pathway to individual images; bbox ccordinates in format proportional to
    image size (i.e. all bbox coordinates [0,1]) with xmin, ymin corresponding to upper left corner and
    xmax, ymax corresponding to lower right corner.
    Images are resized, channels converted, and augmented according to data augmentation pipelines defined below.
    Bboxes also undergo corresponding data augmentation.
    Each filename corresponds to a 'target' dict of bboxes and labels.
    Images and targets are returned as Tensors.
    """

    def __init__(self, df, image_dir, w, h, label2target, transform):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h
        self.label2target = label2target
        self.transform = transform

    def __getitem__(self, item):
        # create image id
        image_id = self.image_infos[item]
        # create full path to open each image file
        img_path = os.path.join(self.image_dir, image_id).replace("\\", "/")
        # open image
        img = cv2.imread(img_path)
        # reformat color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize image so bboxes can also be converted
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.
        # img = Image.open(img_path).convert("RGB").resize((self.w, self.h), resample=Image.Resampling.BILINEAR)
        # img = np.array(img, dtype="float32")/255.
        # filter df rows for img
        df = self.df
        data = df[df['filename'] == image_id]
        # extract label names
        labels = data['LabelName'].values.tolist()
        # extract bbox coordinates
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute values for model input
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        # convert coordinates to list
        boxes = data.tolist()
        # convert bboxes and labels to a tensor dictionary
        target = {
            'boxes': boxes,
            'labels': torch.tensor([self.label2target[i] for i in labels]).long()
        }
        # apply data augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, bboxes=target['boxes'], labels=labels)
            img = (augmented['image'])
            target['boxes'] = augmented['bboxes']
        target['boxes'] = torch.tensor(target['boxes']).float()  # ToTensorV2() isn't working on bboxes
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)


# create placeholders for targets and predictions
pred_df = []
target_df = []

# deploy model on test images
model.eval()
for i in tqdm(range(len(image_infos))):
    # define dataset and dataloader
    dfi = df[df['filename'] == image_infos[i]]
    dsi = DetectDataset(df=dfi, image_dir=IMAGE_ROOT, w=w, h=h, label2target=label2target, transform=val_transform)
    dli = DataLoader(dsi, batch_size=1, collate_fn=dsi.collate_fn, drop_last=True)

    # extract image, bbox, and label info
    input, target = next(iter(dli))
    tbs = dsi[0][1]['boxes']
    image = list(image.to(device) for image in input)

    # run input through the model
    output = model(image)[0]

    # extract prediction bboxes, labels, scores above score_thresh
    # format prediction data
    bbs = output['boxes'].cpu().detach()
    labels = output['labels'].cpu().detach()
    confs = output['scores'].cpu().detach()

    # id indicies of tensors to include in evaluation
    idx = torch.where(confs > 0.1)

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
    tar_df_i = pd.DataFrame({
        'filename': image_infos[i],
        'file_id': image_infos[i][:-4],
        'class_name': dfi['LabelName'].tolist(),
        'bbox': tbs.tolist()
    })
    pred_df.append(pred_df_i)
    target_df.append(tar_df_i)

# concatenate preds and targets into dfs
pred_df = pd.concat(pred_df).reset_index(drop=True)
target_df = pd.concat(target_df).reset_index(drop=True)

# save prediction and target dfs to csv
target_df.to_csv(output_path + "target_df.csv")
pred_df.to_csv(output_path + "pred_df.csv")

# define format to read bboxes
# use 'csv' if reloading bboxes from csv file; use 'env' if working with direct model output
format = 'csv'

if format == 'csv':
    pred_df = pd.read_csv(output_path + "pred_df.csv")
    target_df = pd.read_csv(output_path + "target_df.csv")

# extract predicted bboxes, confidence scores, and labels
preds = []
targets = []

# create list of dictionaries for targets, preds for each image
for i in tqdm(range(len(image_infos))):
    # extract predictions and targets for an image
    p_df = pred_df[pred_df['filename'] == image_infos[i]]
    t_df = target_df[target_df['filename'] == image_infos[i]]

    # format boxes based on input
    if format == 'csv':
        # pred detections
        pred_boxes = [box.strip('[').strip(']').strip(',') for box in p_df['bbox']]
        pred_boxes = np.array([np.fromstring(box, sep=', ') for box in pred_boxes])
        # ground truth boxes
        target_boxes = [box.strip('[').strip(']').strip(', ') for box in t_df['bbox']]
        target_boxes = np.array([np.fromstring(box, sep=', ') for box in target_boxes])
    if format == 'env':
        # pred detections
        pred_boxes = [box for box in p_df['bbox']]
        # ground truth boxes
        target_boxes = [box for box in t_df['bbox']]


    # format scores and labels
    pred_scores = p_df['confidence'].values.tolist()
    pred_labels = p_df['class_name'].map(label2target)
    target_labels = t_df['class_name'].map(label2target)

    # convert preds to dictionary of tensors
    pred_i = {
        'boxes': torch.tensor(pred_boxes),
        'scores': torch.tensor(pred_scores),
        'labels': torch.tensor(pred_labels.values)
    }

    # convert targets to tensor dictionary
    target_i = {
        'boxes': torch.tensor(target_boxes),
        'labels': torch.tensor(target_labels.values)
    }

    # add current image preds and targets to dictionary list
    preds.append(pred_i)
    targets.append(target_i)

# initialize metric
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
metric.update(preds, targets)
results = metric.compute()

# Add class names to results
results_df = pd.DataFrame({k: np.array(v) for k, v in results.items()}).reset_index().rename(columns={"index": "target"})
results_df['class_name'] = results_df['target'].map(target2label)
results_df.drop(['target'], axis = 1)

# add F1 score to results
results_df['f1_score'] = 2 * ((results_df['map_per_class'] * results_df['mar_100_per_class']) /
                              (results_df['map_per_class'] + results_df['mar_100_per_class']))

# save results df to csv
results_df.to_csv(output_path + "results_df.csv")


# Save model weights for loading into R package
path2weights = output_path + cnn_backbone + "_" + num_classes + "classes_weights_cpu.pth"
torch.save(dict(model.to(device='cpu').state_dict()), f=path2weights)

# save model architecture for loading into R package
model.eval()
s = torch.jit.script(model.to(device='cpu'))
torch.jit.save(s, output_path + cnn_backbone + "_" + num_classes + "classes_Arch_cpu.pt")

