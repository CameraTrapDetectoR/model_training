"""
Script to deploy CameraTrapDetectoR model in eval mode via command line

This module allows users to run the suite of CameraTrapDetectoR models on a personal computer or
on a high-performance computing (HPC) system.

Need to find a way to share the appropriate folder with model_args.txt and a model checkpoint


"""

import os
import torch
from PIL import ImageFile
import numpy as np
import pandas as pd
import cv2

from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn
from tqdm import tqdm
from torchvision.ops import nms

from collections import Counter

import argparse
import sys

#######
## -- Prepare System and Data for Model Training
#######

# Set location


# Set paths
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/trainimages"
    os.chdir('/project/cameratrapdetector')

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# determine if using CPU or GPU
def get_device():
    # load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print message
    if device.type == 'cuda':
        print('Images will be run with GPU')
    else:
        print('Images will be run with CPU')

    return device




# set path to model run being deployed
model_path = "./output/species_v2/"

# open model arguments file
with open(model_path + 'model_args.txt') as f:
    model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
model_args['image width'] = int(model_args['image width'])
model_args['image height'] = int(model_args['image height'])
model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))
cnn_backbone = model_args['backbone']

# define image dimensions
w = model_args['image width']
h = model_args['image height']

# load model checkpoint
checkpoint_path = model_path + "checkpoint_50epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# load model type
model_type = checkpoint['model_type']

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# reload anchor generator
anchor_sizes, anchor_gen = get_anchors(h=h)

# initiate model
def get_model(checkpoint):
    cnn_backbone = 'resnet'
    num_classes = checkpoint['num_classes']
    model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen)

    # load model weights
    model.load_state_dict(checkpoint['state_dict'])

    return model.to(device)



# set image directory


# load image filepaths
def image_infos(image_dir):
    # walk through image_dir and make list of all jpg, jpeg files
    image_jpgs = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(image_dir) for f in fn if
                   os.path.splitext(f)[1].lower() == '.jpg']
    image_jpegs = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(image_dir) for f in fn if
                  os.path.splitext(f)[1].lower() == '.jpeg']
    # combine lists
    image_jpgs += image_jpegs
    # remove duplicates
    image_infos = [i for n, i in enumerate(image_jpgs) if i not in image_jpgs[:n]]

    return image_infos



# define checkpoint path
# chkpt_pth = IMAGE_PATH + "_" + model_type + '_pred_checkpoint.csv'
chkpt_pth = IMAGE_ROOT + "/TPWD_Gallagher/2016/2016.05.19/SAB309_" + model_type + '_pred_checkpoint.csv'

#######
## -- Evaluate Model on Test Data
#######

# create placeholder for predictions
pred_df = pd.DataFrame(columns=['filename', 'file_id', 'class_name', 'confidence', 'bbox'])

resume_from_checkpoint = True
if resume_from_checkpoint == True:
    # load checkpoint file
    pred_checkpoint = pd.read_csv(chkpt_pth)

    # turn pred_checkpoint into list of dataframes and add to pred_df
    pred_df = pd.concat([pred_df, pred_checkpoint], ignore_index=True)

    # filter through image infos and update list to images not in pred_df
    also_rans = pred_df.filename.unique().tolist()
    image_infos = [x for x in image_infos if x not in also_rans]

# deploy model
count = 0
with torch.no_grad():
    model.eval()
    for i in tqdm(range(len(image_infos))):
        try:
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
                    'bbox': [[0, 0, w, h]]
                })
            else:
                pred_df_i = pd.DataFrame({
                    'filename': image_infos[i],
                    'file_id': image_infos[i][:-4],
                    'class_name': [target2label[a] for a in labels],
                    'confidence': confs,
                    'bbox': bbs
                })
        except Exception as err:
            pred_df_i = pd.DataFrame({
                'filename': image_infos[i],
                'file_id': image_infos[i][:-4],
                'class_name': "Image error",
                'confidence': 0,
                'bbox': [[0, 0, 0, 0]]
            })
            pass

        # add image predictions to existing df
        pred_df = pd.concat([pred_df, pred_df_i], ignore_index=True)

        # save results every 10 images
        count += 1
        if count % 10 == 0:
            # save to checkpoint
            pred_df.to_csv(chkpt_pth, index=False)

# save prediction and target dfs to csv
# pred_df.to_csv(IMAGE_ROOT + "_" + model_type + '_results_raw.csv', index=False)
pred_df.to_csv(IMAGE_ROOT + "/TPWD_Gallagher/2016/2016.05.19/SAB307_" + model_type + '_results_raw.csv', index=False)

# remove checkpoint file
os.remove(chkpt_pth)

#######
## -- Post Processing
#######

# # Drop bboxes
pred_df = pred_df.drop(['bbox'], axis=1)
#
# # Rename and remove columns
pred_df = pred_df.rename(columns={'filename': 'file_path', 'class_name': 'prediction'}).drop(['file_id'], axis=1)

# # extract image name/structure from file_path
image_names = pred_df['file_path']
image_names = image_names.str.replace('D:/2016.05.19/', '')
pred_df['image_name'] = image_names

# split image name to extract site, camera, date info
image_parts = image_names.str.rsplit("/", n=3, expand=True)

# site name
pred_df['site'] = image_parts[0].str.slice(stop=3)

# camera name
pred_df['cam_id'] = image_parts[0]

# timestamp
pred_df['timestamp'] = image_parts[1].str.replace(".JPG", "").str.replace("-", ":")

#
# # get prediction counts for each image
cts = Counter(pred_df['file_path']).items()
pred_counts = pd.DataFrame.from_dict(cts)
pred_counts.columns = ['file_path', 'count']
pred_df = pred_df.merge(pred_counts, on='file_path', how='left')

# # separate images with one prediction and images with >1 predictions
single_preds = pred_df[pred_df['count'] == 1]
multi_preds = pred_df[pred_df['count'] > 1]

# # format single preds
single_preds.loc[single_preds['prediction'] == 'empty', 'count'] = 0
#
# # drop counts from multi_preds
multi_preds = multi_preds.drop(['count'], axis=1)

# # get new counts based on image + predicted class
multi_cts = multi_preds.groupby(['file_path', 'prediction'])['prediction'].count().reset_index(name='count')
#
# # join multi_preds to new counts
multi_preds = multi_preds.merge(multi_cts, on=['file_path', 'prediction'], how='left', copy=False)
#
# # filter multi_preds to one prediction per image + class group - take highest confidence
filtr_preds = multi_preds.groupby(['file_path', 'prediction']).apply(
    lambda x: x[x['confidence'] == max(x['confidence'])])

# join filtered multi_preds to single_preds
preds = pd.concat([single_preds, filtr_preds], ignore_index=True).sort_values(['file_path'])

# reorder image_name column
preds = preds.loc[:, ['file_path', 'image_name', 'prediction', 'confidence', 'count']]

# add columns for manual review: true_class, true_count, comments
preds['true_class'] = ""
preds['true_count'] = ""
preds['comments'] = ""

# save with new formatted name
# preds.to_csv(IMAGE_PATH + "_" + model_type + '_results_formatted.csv', index=False)
preds.to_csv(IMAGE_ROOT + "/TPWD_Gallagher/2016/2016.05.19/SAB307_" + model_type + '_results_formatted.csv',
             index=False)


# END


##### -- COMMAND-LINE Driver

def main():
    ## -- Add model arguments
    parser = argparse.ArgumentParser(
        description='Module to run CameraTrapDetectoR models on HPC via command line arguments.' + \
                    'Two final results files will be provided. Raw results will contain one row for each detection with bounding box.' + \
                    'Formatted results will include one row for each detected class per image with predicted count.'
    )
    parser.add_argument(
        'model_folder',
        help='Path to model files'
    )
    parser.add_argument(
        'image_dir',
        help='Path to image directory. The script will automatically recurse into sub-folders of this directory.' + \
             'Currently only .jpg files are accepted.'
    )
    parser.add_argument(
        'output_dir',
        help='Path to output directory where your results files will be stored - a new folder will be created.' + \
             'If left NULL, results will be stored in your image_dir.'
    )
    parser.add_argument(
        '--plot_bboxes',
        action='store_true',
        help='Create image copies with bounding boxes in your output_dir.'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0,
        help='Filter out predictions below this confidence threshold. Default is 0'
    )
    parser.add_argument(
        '--plot_images',
        action='store_true',
        help='Plot image copies with bounding boxes and predicted classes drawn.'
    )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=10,
        help='Write raw results to a temporary file every N images; default is 10.' + \
             'Setting the value to -1 will disable checkpointing but is not recommended.'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume model run from checkpoint file. Provide full path.'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # confirm model folder exists
    assert os.path.isdir(args.model_folder), \
        'model_folder {} does not exist'.format(args.model_folder)
    # confirm all files in model folder
    assert os.path.exists(model_path + '/model_args.txt'), \
        'Model args text file does not exist in model_folder'
    assert os.path.exists(model_path + '/model_checkpoint.pth'), \
        'Model checkpoint .pth file does not exist in model_folder'

    # confirm image dir exists
    assert os.path.isdir(args.image_dir), \
        'image_dir {} does not exist'.format(args.image_dir)

    # confirm score_threshold is between [0, 1]
    assert 0.0 < args.score_threshold <= 1.0, 'Confidence threshold must be between 0 and 1'

    # load checkpoint if available
    if args.resume_from_checkpoint is not None:
        # confirm checkpoint path exists
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'

        # load checkpoint file
        pred_checkpoint = pd.read_csv(args.resume_from_checkpoing)

        # get list of images already run
        also_rans = pred_checkpoint.filename.unique().tolist()

        # inform user of checkpoint length
        print('Loaded checkpointed results from {} previously-run images'.format(len(also_rans)))
