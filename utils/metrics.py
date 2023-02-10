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

