# Functions for evaluating model performance


import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd
import matplotlib.pyplot as plt


def plot_losses(model_type, cnn_backbone, loss_history):
    # extract losses and number of epochs
    train_loss = [loss for loss in loss_history['train']]
    val_loss = [loss for loss in loss_history['val']]
    epochs = range(1, len(train_loss) + 1)
    # format and plot
    plt.plot(epochs, train_loss, 'bo', label='Train Loss')
    plt.plot(epochs, val_loss, 'b', label='Val Loss')
    plt.title(model_type + " " + cnn_backbone + " Faster R-CNN Loss History")
    plt.legend()


# filter predictions with low probability scores
def filter_preds(output, threshold):
    """
    filter output based on probability score; exclude predictions less than threshold
    :param output: model output of all predictions for a particular image
    :param threshold: probability score below which to exclude all predictions
    :return:
    """

    # format prediction data
    bbs = output['boxes'].cpu().detach()
    labels = output['labels'].cpu().detach()
    confs = output['scores'].cpu().detach()

    # id indicies of tensors to include in evaluation
    idx = torch.where(confs > threshold)

    # filter to predictions that meet the threshold
    bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

    return bbs, labels, confs

def prepare_results(pred_df, target_df, image_infos, label2target):
    """
    collects predictions and ground truth boxes into tensor dictionaries for calculating evaluation metrics
    :return:
    """
    # extract predicted bboxes, confidence scores, and labels
    preds = []
    targets = []

    # create list of dictionaries for targets, preds for each image
    for i in tqdm(range(len(image_infos))):
        # extract predictions and targets for an image
        p_df = pred_df[pred_df['filename'] == image_infos[i]]
        t_df = target_df[target_df['filename'] == image_infos[i]]

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

    return preds, targets

def calculate_metrics(preds, targets, target2label):
    """
    calculate evaluation metrics for test results

    :return: classwise results in df
    """
    # initialize metric
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    metric.update(preds, targets)
    results = metric.compute()

    # convert results to dataframe
    results_df = pd.DataFrame({k: np.array(v) for k, v in results.items()}).reset_index().\
        rename(columns={"index": "target"})

    # add F1 score to results
    results_df['f1_score'] = 2 * ((results_df['map_per_class'] * results_df['mar_100_per_class']) /
                                  (results_df['map_per_class'] + results_df['mar_100_per_class']))

    # Add class names to results
    results_df['class_name'] = results_df['target'].map(target2label)
    results_df = results_df.drop(['target'], axis=1)

    return results_df