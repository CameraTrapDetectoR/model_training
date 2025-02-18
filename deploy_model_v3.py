"""
Script to deploy CameraTrapDetector V3 model in eval mode via command line

This module allows users to run the latest CameraTrapDetector models on a personal computer or
on a high-performance computing (HPC) system.

"""

# import libraries and functions
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
import torch
import cv2
from tqdm import tqdm
import argparse
import sys
import glob
from pathlib import Path
from datetime import datetime

##### -- UTILITY FUNCTIONS

# set default thresholds
DEFAULT_SCORE_THRESHOLD = 0.05
DEFAULT_OVERLAP_THRESHOLD = 0.5

# get device
def get_device():
    """
    Determine whether model will run on CPU or GPU
    Args: 
        none
    Returns:
        str: device for loading model
    """
    # load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print message
    if device.type == 'cuda':
        print('Images will be run on GPU')
    else:
        print('Images will be run on CPU')
    return device

# load label dict
def label_dict():
    """
    Define class names and organize them into a df with corresponding label number
    Args:
        none
    Returns:
        pandas DataFrame: class label dataframe
    """
    label2target = {0: 'American_Badger', 1: 'American_Black_Bear', 2: 'American_Marten', 3: 'Bighorn_Sheep', 4: 'Blackbird_Cowbird_Grackle', 
                    5: 'Bobcat', 6: 'Canada_Lynx', 7: 'Caribou', 8: 'Collared_Peccary',9: 'Common_Raccoon', 10: 'Coyote', 11: 'Crow_Raven', 
                    12: 'Domestic_Cat', 13: 'Domestic_Chicken', 14: 'Domestic_Cow', 15: 'Domestic_Dog', 16: 'Domestic_Goat', 17: 'Domestic_Sheep', 
                    18: 'Dove_Pigeon', 19: 'Eagle_Osprey', 20: 'Egret', 21: 'Fisher', 22: 'Fox', 23: 'Francolin', 24: 'Grizzly_Bear', 25: 'Grouse', 
                    26: 'Heron', 27: 'Horse_Donkey', 28: 'Human', 29: 'Iguana', 30: 'Jaguar_Jaguarundi',31: 'Jay', 32: 'Margay', 33: 'Marmot_Woodchuck', 
                    34: 'Moose', 35: 'Mountain_Lion', 36: 'Mouse_Rat', 37: 'Mule_Deer', 38: 'Nilgai',39: 'Nine-Banded_Armadillo', 
                    40: 'North_American_Beaver', 41: 'North_American_Porcupine', 42: 'Ocelot', 43: 'Owl', 44: 'Polar_Bear', 45: 'Prairie_Dog', 
                    46: 'Pronghorn', 47: 'Quail', 48: 'Rabbit_Hare', 49: 'River_Otter', 50: 'Robin_Thrush', 51: 'Rocky_Mountain_Elk', 52: 'Skunk', 
                    53: 'Squirrel_Chipmunk', 54: 'Vehicle', 55: 'Virginia_Opossum', 56: 'Vulture', 57: 'Weasel_Mink', 58: 'White-Nosed_Coati', 
                    59: 'White-Tailed_Deer', 60: 'Wild_Pig', 61: 'Wild_Turkey', 62: 'Wolf', 63: 'Wolverine', 'Empty':'Empty'}
    l2t_df = pd.DataFrame.from_dict(label2target.items()).rename(columns={0:"class_label", 1:"prediction"})
    return(l2t_df)

# get image files
def get_img_files(image_dir):
    """
    Return list of all image files in user-defined image directory. Will search directory recursively
    Args:
        image_dir (str): Full path to image directory
    Returns:
        list: a list of all .jpg files in the image directory
    """
    # define empty list for image files
    img_files = []
    # walk recursively through user-defined image dir
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            img_files.append(os.path.join(root, file))
    # filter list to accepted file types
    img_files = [pic for pic in img_files if pic.endswith((".JPG", ".jpg", ".JPEG", ".jpeg"))]
    return img_files

## -- FUNCTION TO PREDICT ONE IMAGE
def predict_one_image(file_path, model, l2t_df, confidence_score, iou_threshold): 
    '''
        function to deploy the model on a single image
        Args:
            file_path: full path to image 
            model: loaded model
            l2t_df: label dictionary 
            confidence_score: confidence threshold for predictions
            iou_threshold: intersection over union threshold for aggregating predictions  
        Returns:
            dataframe: classes and bounding box coordinates for each detection in a single image
    '''
    # deploy model
    res = model.predict(file_path, conf = confidence_score, iou = iou_threshold, agnostic_nms=True)

    # extract predictions
    for r in res:
        boxes = r.boxes.cpu()
    
    # format label, confidence, bbox coordinates
    box_class = torch.Tensor.numpy(boxes.cls)
    box_conf = torch.Tensor.numpy(boxes.conf)
    box_xywhn = torch.Tensor.numpy(boxes.xywhn)

    # make dataframe of prediction
    df_i = pd.DataFrame({'class_label': [c for c in box_class], 
            'confidence': [conf for conf in box_conf],
            'x_center': [x for x in box_xywhn[:,0]],
            'y_center': [x for x in box_xywhn[:,1]],
            'box_w': [w for w in box_xywhn[:,2]],
            'box_h': [h for h in box_xywhn[:,3]]})
    if box_class.size==0:
        df_i = pd.DataFrame({'class_label':['Empty'],
                            'confidence': ['NA'], 
                            'x_center': ['NA'],
                            'y_center': ['NA'], 
                            'box_w': ['NA'], 
                            'box_h': ['NA']})
    
    # add filename
    df_i['filename'] = file_path

    # get image metadata
    #TODO: keep working on extracting more metadata
    # df_i['timestamp'] = get_timestamp(file_path)

    # join label name
    df_i = df_i.merge(l2t_df, how = 'left', on='class_label')

    # reorder columns
    df_i = df_i[['filename', 'prediction', 'confidence', 'x_center', 'y_center', 'box_w', 'box_h']]

    return df_i

## -- FUNCTION TO PLOT PREDICTIONS
def plot_image(df_i, IMAGE_DIR):
    """
    Save image with labeled bounding boxes drawn
    Args:
        df_i: df of predictions from `predict_one_image`
        IMAGE_DIR: path to image dir for renaming files
    Returns:
        Image file
    """
    # load image
    im = cv2.imread(df_i.loc[0,'filename'])

    # get image height and width
    im_w = im.shape[1]
    im_h = im.shape[0]

    # draw boxes and labels
    for row in range(len(df_i)):
        x_cen = df_i.loc[row]['x_center'] * im_w
        w = df_i.loc[row]['box_w'] *im_w
        y_cen = df_i.loc[row]['y_center'] * im_h
        h = df_i.loc[row]['box_h'] * im_h
        x_min, x_max, y_min, y_max = int(x_cen - (w/2)), int(x_cen + (w/2)), int(y_cen - (h/2)), int(y_cen + (h/2))
        cv2.rectangle(im, (x_min, (y_min + 5)), (x_max, y_max), color=(0, 0, 255), thickness=2)
        cv2.putText(im, df_i.loc[row]['prediction'] + "=" + str(round(df_i.loc[row]['confidence'], 2)), 
            (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # create new new filename
    pred_file = df_i.iloc[0]['filename'].replace(IMAGE_DIR, "").replace("/", "_").replace("\\", "")

    return pred_file, im


## -- FORMAT PREDICTIONS  
def format_predictions(pred_df):
    """
    Aggregate detections by image, predicted class
    Args:
        pred_df
    Returns:
        format_df
    """
    # extract relevant columns
    form_df = pred_df[['filename', 'prediction', 'confidence']]
    # create df of class counts per image
    cts = form_df.groupby(['filename', 'prediction']).size().reset_index(name = 'predicted_count')
    # update empty count to 0
    cts.loc[cts.prediction == 'Empty', 'predicted_count'] = 0
    # merge counts to predictions
    form_df = form_df.merge(cts, how='left', on = ['filename', 'prediction'])
    # remove duplicates for multiple counts
    form_df = form_df.sort_values(by = ['filename', 'prediction', 'confidence']).drop_duplicates(subset = ['filename', 'prediction'])
    return form_df.reset_index(drop=True)   

##### -- COMMAND-LINE Driver

def main():
    ## -- Add model arguments
    parser = argparse.ArgumentParser(
        description='Module to run CameraTrapDetector models (V3 and later) via command line arguments.' + \
                    'Two final results files will be provided. Raw results will contain one row for each detection with bounding box. ' + \
                    'Formatted results will include one row for each detected class per image with predicted count.' + \
                    'You can optionally return plotted bounding boxes on your images.' + \
                    'For more detailed documentation visit our Github repo: https://github.com/CameraTrapDetectoR/'
    )
    parser.add_argument(
        'model_file',
        help='Full path to model weights file titled weights.pt.'
    )
    parser.add_argument(
        'image_dir',
        help='Path to image directory. The script will automatically recurse into sub-folders of this directory. ' + \
             'Currently only .jpg/.jpeg files are accepted.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Path to output directory where your results files will be stored - a new folder inside this directory will be created. ' + \
             'If left NULL, the results folder will be created inside your image_dir.'
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
    # PLOT IMAGES WORKING IN PYTHON INTERPRETER BUT NOT FROM COMMAND LINE.
    parser.add_argument(
        '--plot_images',
        action='store_true',
        help='Plot image copies with bounding boxes and predicted classes drawn. A folder named after the model version ' + \
             'and "prediction_plots" will be created inside your output_dir/image_dir to hold these plots.'
    )
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
        help='Resume model run from checkpoint file. Provide full path to raw predictions file with bounding box coordinates.'
    )

    ## -- ARGUMENT CHECKS

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # confirm model file exists
    assert os.path.exists(args.model_file), \
        'model_file {} does not exist'.format(args.model_file)
    MODEL_FILE = args.model_file

    # confirm image dir exists
    assert os.path.isdir(args.image_dir), \
        'image_dir {} does not exist'.format(args.image_dir)
    IMAGE_DIR = args.image_dir

    # confirm score_threshold is between [0, 1]
    assert 0.0 < args.score_threshold <= 1.0, 'Confidence score threshold must be between 0 and 1'
    SCORE_THRESHOLD = args.score_threshold

    # confirm overlap_threshold is between [0, 1]
    assert 0.0 < args.overlap_threshold <= 1.0, 'Overlap threshold must be between 0 and 1'
    OVERLAP_THRESHOLD = args.overlap_threshold

    # define output dir
    if args.output_dir is not None:
        assert os.path.isdir(args.output_dir), \
            'output_dir {} does not exist. Please create this folder and try again.'.format(args.output_dir)
        OUTPUT_DIR = args.output_dir
        if OUTPUT_DIR[-1] != "/":
            OUTPUT_DIR = OUTPUT_DIR + '/'
    else:
        OUTPUT_DIR = IMAGE_DIR + '/CameraTrapDetector_V3_Results/'
        if os.path.isdir(OUTPUT_DIR) == False:
            os.mkdir(OUTPUT_DIR)
    print('All results will be stored in the directory: '.format(len(OUTPUT_DIR)))


    ## -- GET DEVICE
    device = get_device()

    ## -- LOAD MODEL
    model = YOLO(MODEL_FILE)

    ## -- LOAD LABEL DICTIONARY
    l2t_df = label_dict()
  
    ## -- LOAD IMAGES TO RUN
    image_files = get_img_files(IMAGE_DIR)
    print('Found {} total image files in your image directory'.format(len(image_files)))

    ## -- CREATE PLACEHOLDER PRED DF
    pred_df = pd.DataFrame(columns = ['filename', 'prediction', 'confidence', 'x_center', 'y_center', 'box_w', 'box_h'])
    
    ## -- LOAD CHECKPOINT
    if args.resume_from_checkpoint is not None:
        # confirm checkpoint path exists
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'

        # load checkpoint file
        pred_checkpoint = pd.read_csv(args.resume_from_checkpoint)

        #confirm checkpoint loaded is the raw predictions with bboxes
        assert all(pred_checkpoint.columns == ['filename', 'prediction', 'confidence', 'x_center', 'y_center', 'box_w', 'box_h']), 'Please load the raw checkpoint file with bounding box coordinates'

        # get list of images already run
        also_rans = pred_checkpoint.filename.unique().tolist()

        # join checkpoint to placeholder df
        pred_df = pd.concat([pred_df, pred_checkpoint], ignore_index=True)

        # inform user of checkpoint length
        print('Loaded results from {} previously-run images'.format(len(also_rans)))

        # filter also_rans out of image files
        image_files = [x for x in image_files if x not in also_rans]
    
    ## -- DEFINE CHECKPOINT FILENAMES
    if args.checkpoint_frequency != -1:
        checkpoint_time = datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
        # set raw checkpoint file
        checkpoint_raw = OUTPUT_DIR + 'CTDv3_predictions_raw_checkpoint_' + checkpoint_time
        # set formatted checkpoint file
        checkpoint_format = OUTPUT_DIR + 'CTDv3_predictions_formatted_checkpoint_' + checkpoint_time
        print('New and existing raw results will be checkpointed in the filepath: {}'.format(Path(checkpoint_raw)))
        print('New and existing formatted results will be checkpointed in the filepath: {}'.format(Path(checkpoint_format)))

    ## -- CREATE FOLDER FOR PREDICTIONS
    if args.plot_images:
        plot_folder = OUTPUT_DIR + 'prediction_plots/'
        if os.path.isdir(plot_folder) == False:
            os.mkdir(plot_folder)
        print('Image copies with nonempty predictions will be saved in real time in the folder: {}'.format(plot_folder))


    ## -- RUN MODEL
    count = 0
    for f in tqdm(range(len(image_files))):
        # run the model
        try:
            df_i = predict_one_image(image_files[f], model=model, l2t_df=l2t_df,
                                     confidence_score=SCORE_THRESHOLD, iou_threshold=OVERLAP_THRESHOLD)
        except Exception as e:
            df_i = pd.DataFrame({'filename': image_files[f], 'prediction':['file_error'], 'confidence': ['NA'], 
                                'x_center': ['NA'], 'y_center': ['NA'], 'box_w': ['NA'], 'box_h': ['NA']})
        # plot predictions
        if args.plot_images:
            # plot prediction if pred is not empty or file error
            if df_i.loc[0]['prediction'] not in ['Empty', 'file_error']:
                pred_file, im = plot_image(df_i=df_i, IMAGE_DIR=IMAGE_DIR)
                # save image
                cv2.imwrite(plot_folder + 'pred_' + pred_file, im)
        
        # concat df_i to df
        pred_df = pd.concat([pred_df, df_i], ignore_index=True)

        # increase count
        count += 1

        # Checkpointing
        if count % args.checkpoint_frequency == 0:
            # save raw checkpoint
            pred_df.to_csv(checkpoint_raw, index=False)
            # save formatted checkpoint
            form_df = format_predictions(pred_df)
            form_df.to_csv(checkpoint_format, index=False)
    
    ## -- FINALIZE PREDICTIONS

    # Save raw predictions
    raw_results = OUTPUT_DIR + 'CTDv3_predictions_raw_final.csv'
    pred_df.to_csv(raw_results, index=False)
    print("Raw results with YOLO-format bounding boxes can be found in the path {}".format(Path(raw_results)))

    # Save formatted predictions
    form_results = OUTPUT_DIR + 'CTDv3_predictions_formatted_final.csv'
    form_df = format_predictions(pred_df)
    form_df.to_csv(form_results, index=False)
    print("Formatted results with class counts can be found in the path {}".format(Path(form_results)))

    # Point to plots
    if args.plot_images:
        print("Plotted predictions can be found in: {}".format(Path(plot_folder)))
    # Delete checkpoints
    checkpoint_files = [OUTPUT_DIR + fn for fn in os.listdir(OUTPUT_DIR) if 'checkpoint' in fn]
    if len(checkpoint_files) > 0:
        print("Deleting checkpoint files: {}".format(checkpoint_files))
        for f in range(len(checkpoint_files)):
            os.remove(checkpoint_files[f])
        print("All checkpoints deleted.")

    ## -- GOODBYE
    print("Model run complete! All results can be found in the directory: {}".format(Path(OUTPUT_DIR)))

if __name__ == '__main__':
    main()

# - END

