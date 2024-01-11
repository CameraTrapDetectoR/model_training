"""
Script to deploy CameraTrapDetectoR model in eval mode via command line

This module allows users to run the suite of CameraTrapDetectoR models on a personal computer or
on a high-performance computing (HPC) system.

Need to find a way to share the appropriate folder with model_args.txt and a model checkpoint


"""

import os
import warnings
import torch
from PIL import ImageFile, Image
import numpy as np
import pandas as pd
import cv2

from datetime import datetime

from utils.hyperparameters import get_anchors
from models.backbones import load_fasterrcnn
from tqdm import tqdm
from torchvision.ops import nms

from utils.post_process import format_evals, plot_image, normalize_bboxes, get_metadata

import argparse
import sys

from pathlib import Path

#######
## -- Prepare System and Data for Model Training
#######

# suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set default thresholds
DEFAULT_SCORE_THRESHOLD = 0.05
DEFAULT_OVERLAP_THRESHOLD = 0.5


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


# get target2label
def get_target2label(label2target):
    # reverse label2target
    target2label = {t: l for l, t in label2target.items()}

    # make sure the empty entry is at the beginning of the dictionary
    keys = sorted(target2label.keys())
    vals = [target2label[k] for k in keys]
    target2label = dict(zip(keys, vals))

    return target2label


# initiate model
def get_model(checkpoint, anchor_gen):
    cnn_backbone = 'resnet'
    num_classes = checkpoint['num_classes']
    model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen)

    # load model weights
    model.load_state_dict(checkpoint['state_dict'])

    return model


# load image filepaths
def get_image_infos(image_dir):
    # walk through image_dir and make list of all jpg, jpeg files
    image_jpgs = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(image_dir) for f in fn if
                  os.path.splitext(f)[1].lower() == '.jpg']
    image_jpegs = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(image_dir) for f in fn if
                   os.path.splitext(f)[1].lower() == '.jpeg']
    # combine lists
    image_jpgs += image_jpegs
    # remove duplicates
    image_infos = [i for n, i in enumerate(image_jpgs) if i not in image_jpgs[:n]]
    # remove any images in a prediction_plot folder
    image_infos = [f for f in image_infos if not 'prediction_plots' in f]

    return image_infos


# make predictions for a single image
def prepare_image(img_org, w, h, device):
    # reformat color channels
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
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

    return image


# format output
def format_output(output, target2label, w, h, score_threshold, overlap_threshold):
    """
    take raw tensor output and return objects that can be saved more easilty
    :param output: output from sending image through model
    :param target2label: dictionary of class labels
    :param score_threshold: threshold below which to reject predictions
    :param overlap_threshold: overlap threshold by which to perform nms
    :param w: image width for normalizing bbox
    :param h: image height for normalizing bbox
    :return: a series of lists of bounding boxes, confidence scores, and class labels for each prediction for a given image
    """
    # format prediction data
    bbs = output['boxes'].cpu().detach()
    labels = output['labels'].cpu().detach()
    confs = output['scores'].cpu().detach()

    # id indicies of tensors to include in evaluation
    idx = torch.where(confs > score_threshold)

    # filter to predictions that meet the threshold
    bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

    # perform non-maximum suppression on remaining predictions
    ixs = nms(bbs, confs, iou_threshold=overlap_threshold)

    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    # normalize bboxes
    bbs = normalize_bboxes(w, h, bbs)

    # format predictions
    bbs = bbs.tolist()
    confs = confs.tolist()
    labels = labels.tolist()
    class_names = [target2label[a] for a in labels]

    return bbs, confs, class_names


# make pred_df_i for single image
def make_pred_df_i(img_path, class_names, confs, bbs):
    """
    combine outputs for a single image into a dataframe
    :param img_path: full file path to image
    :param class_names: class names of predictions
    :param confs: confidence scores of predictions
    :param bbs: bounding box coordinates of the predictions
    :return: dataframe of results for an image
    """
    if len(bbs) == 0:
        pred_df_i = pd.DataFrame({
            'filename': img_path,
            'file_id': img_path[:-4],
            'class_name': 'empty',
            'confidence': 1,
            'bbox': [[0, 0, 0, 0]]
        })
    else:
        pred_df_i = pd.DataFrame({
            'filename': img_path,
            'file_id': img_path[:-4],
            'class_name': class_names,
            'confidence': confs,
            'bbox': bbs
        })

    pred_df_i['timestamp'] = get_metadata(img_path)

    return pred_df_i


##### -- COMMAND-LINE Driver

def main():
    ## -- Add model arguments
    parser = argparse.ArgumentParser(
        description='Module to run CameraTrapDetectoR models on HPC via command line arguments.' + \
                    'Two final results files will be provided. Raw results will contain one row for each detection with bounding box. ' + \
                    'Formatted results will include one row for each detected class per image with predicted count.' + \
                    'For more detailed documentation, visit the Github repo: https://github.com/CameraTrapDetectoR/model_training'
    )
    parser.add_argument(
        'model_folder',
        help='Path to model files. Should contain files named "model_args.txt" and "model_checkpoint.pth". See Github documentation for ' + \
             'accessing this information from AG Data Commons.'
    )
    parser.add_argument(
        'image_dir',
        help='Path to image directory. The script will automatically recurse into sub-folders of this directory. ' + \
             'Currently only .jpg files are accepted.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Path to output directory where your results files will be stored - a new folder will be created. ' + \
             'If left NULL, results will be stored in your image_dir.'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help='Filter out predictions below this confidence threshold. Default is 0.05'
    )

    parser.add_argument(
        '--overlap_threshold',
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help='Iteratively remove lower scoring boxes which have an IoU greater than the overlap threshold ' + \
             ' with another (higher scoring) box. Default is 0.50'
    )
    ## PLOT IMAGES WORKING IN PYTHON INTERPRETER BUT NOT FROM COMMAND LINE.
    # parser.add_argument(
    #     '--plot_images',
    #     action='store_true',
    #     help='Plot image copies with bounding boxes and predicted classes drawn. A folder named after the model version ' + \
    #          'and "prediction_plots" will be created inside your output_dir or image_dir if output_dir is null.'
    # )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=10,
        help='Write raw results to a temporary file every N images; default is 10. ' + \
             'Setting the value to -1 will disable checkpointing but is not recommended.'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume model run from checkpoint file. Provide full path.'
    )

    ## -- ARGUMENT CHECKS

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # confirm model folder exists
    assert os.path.isdir(args.model_folder), \
        'model_folder {} does not exist'.format(args.model_folder)
    # confirm all files in model folder
    assert os.path.exists(args.model_folder + '/model_args.txt'), \
        'Model args text file does not exist in model_folder'
    assert os.path.exists(args.model_folder + '/model_checkpoint.pth'), \
        'Model checkpoint .pth file does not exist in model_folder'
    MODEL_FOLDER = args.model_folder

    # confirm image dir exists
    assert os.path.isdir(args.image_dir), \
        'image_dir {} does not exist'.format(args.image_dir)
    IMAGE_DIR = args.image_dir

    # confirm score_threshold is between [0, 1]
    assert 0.0 < args.score_threshold <= 1.0, 'Confidence threshold must be between 0 and 1'
    SCORE_THRESHOLD = args.score_threshold

    # confirm overlap_threshold is between [0, 1]
    assert 0.0 < args.overlap_threshold <= 1.0, 'Overlap threshold must be between 0 and 1'
    OVERLAP_THRESHOLD = args.overlap_threshold

    # confirm output dir is directory
    if args.output_dir is not None:
        assert os.path.isdir(args.output_dir), \
            'output_dir {} does not exist. Please create this folder and try again.'.format(args.output_dir)
        OUTPUT_DIR = args.output_dir

    ## -- LOAD MODEL

    # get device
    device = get_device()

    # load model args
    with open(MODEL_FOLDER + '/model_args.txt') as f:
        model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
    model_args['image width'] = int(model_args['image width'])
    model_args['image height'] = int(model_args['image height'])
    model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))

    cnn_backbone = model_args['backbone']

    # define image dimensions
    w = model_args['image width']
    h = model_args['image height']

    # load model checkpoint
    checkpoint_path = MODEL_FOLDER + "/model_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # load model type
    model_type = checkpoint['model_type']
    model_version = model_args['model_version']

    # load dictionaries
    label2target = checkpoint['label2target']
    target2label = get_target2label(label2target)

    # reload anchor generator
    anchor_sizes, anchor_gen = get_anchors(h=h)

    # initiate model
    model = get_model(checkpoint, anchor_gen).to(device)

    ## -- LOAD CHECKPOINT AND PLOT DIRECTORY

    # create placeholder for predictions
    pred_df = pd.DataFrame(columns=['filename', 'file_id', 'class_name', 'confidence', 'bbox'])

    if args.resume_from_checkpoint is not None:
        # confirm checkpoint path exists
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'

        # load checkpoint file
        pred_checkpoint = pd.read_csv(args.resume_from_checkpoint)

        # get list of images already run
        also_rans = pred_checkpoint.filename.unique().tolist()

        # join checkpoint to placeholder df
        pred_df = pd.concat([pred_df, pred_checkpoint], ignore_index=True)

        # inform user of checkpoint length
        print('Loaded results from {} previously-run images'.format(len(also_rans)))

    # create new checkpoint filename
    if args.checkpoint_frequency != -1:
        if args.resume_from_checkpoint is not None:
            chkpt_pth = args.resume_from_checkpoint
        elif args.output_dir is not None:
            chkpt_pth = OUTPUT_DIR + "/" + model_version + "_checkpoint_" + datetime.utcnow().strftime(
                "%Y%m%d%H%M%S") + ".csv"
        else:
            chkpt_pth = IMAGE_DIR + "/" + model_version + "_checkpoint_" + datetime.utcnow().strftime(
                "%Y%m%d%H%M%S") + ".csv"
        print('New and existing results will be checkpointed in the filepath: {}'.format(Path(chkpt_pth)))

    # create prediction plot folder
    # if args.plot_images:
    #     if args.output_dir is not None:
    #         PRED_PATH = OUTPUT_DIR + '/' + model_version + '_prediction_plots/'
    #         if os.path.exists(PRED_PATH) == False:
    #             os.mkdir(PRED_PATH)
    #     else:
    #         PRED_PATH = IMAGE_DIR + '/' + model_version + '_prediction_plots/'
    #         if os.path.exists(PRED_PATH) == False:
    #             os.mkdir(PRED_PATH)
    #     print('Image copies with plotted predictions will be saved in real time in the filepath: {}'.format(PRED_PATH))

    ## -- LOAD IMAGE FILES

    # get image filenames
    image_infos = get_image_infos(IMAGE_DIR)

    print('Found {} total image files in your image directory'.format(len(image_infos)))

    # filter out also_rans, if applicable
    try:
        image_infos = [x for x in image_infos if x not in also_rans]
    except NameError:
        pass

    print('Model will run over {} images.'.format(len(image_infos)))

    ## -- Run Model

    # deploy model
    count = 0
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(image_infos))):
            try:
                # set image path
                img_path = image_infos[i]
                # open image
                img_org = cv2.imread(img_path)

                # prepare model input
                input = prepare_image(img_org, w, h, device)

                # run input through the model
                output = model(input)[0]

                # format output
                bbs, confs, class_names = format_output(output, target2label, w, h, SCORE_THRESHOLD, OVERLAP_THRESHOLD)

                # plot image if argument selected
                # if args.plot_images is not None:
                #     if (len(bbs) > 0):
                #         plot_image(image=img_org, bbs=bbs, confs=confs, labels=class_names,
                #                    img_path=img_path, PRED_PATH=PRED_PATH, IMAGE_PATH=IMAGE_DIR)

                pred_df_i = make_pred_df_i(img_path, class_names, confs, bbs)

            except Exception as err:
                pred_df_i = pd.DataFrame({
                    'filename': image_infos[i],
                    'file_id': image_infos[i][:-4],
                    'class_name': "Image error",
                    'confidence': 0,
                    'bbox': [[0, 0, 0, 0]],
                    'timestamp': "NA"
                })
                pass

            # add image predictions to existing df
            pred_df = pd.concat([pred_df, pred_df_i], ignore_index=True)

            # increase count
            count += 1

            # save checkpoint if requested
            if count % args.checkpoint_frequency == 0:
                pred_df.to_csv(chkpt_pth, index=False)

    # Finalize raw results
    if args.output_dir is not None:
        raw_results = OUTPUT_DIR + '/' + model_version + "_results_raw.csv"
    else:
        raw_results = IMAGE_DIR + '/' + model_version + "_results_raw.csv"
    pred_df.to_csv(raw_results, index=False)
    print("Raw results with proportional bounding boxes can be found in the path {}".format(Path(raw_results)))

    ## -- Format Results

    # aggregate by image, prediction to get counts
    format_df = format_evals(pred_df, IMAGE_DIR)

    # save formatted results
    if args.output_dir is not None:
        format_results = OUTPUT_DIR + '/' + model_version + "_results_formatted.csv"
    else:
        format_results = IMAGE_DIR + '/' + model_version + "_results_formatted.csv"
    format_df.to_csv(format_results, index=False)

    # remove checkpoint
    os.remove(Path(chkpt_pth))

    # Final update
    print("Model run complete! Formatted results can be found in the path {}".format(Path(format_results)) + \
          " Checkpoint files have been removed.")

    ## END


if __name__ == '__main__':
    main()

    # TODO add option to write metadata tags
