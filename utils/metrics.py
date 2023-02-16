# functions to evaluate model performace

import torch


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

def prepare_results(image_infos=image_infos, format='env', label2target=label2target):
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

    return preds, targets