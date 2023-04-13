"""
Functions to create, save, and load model checkpoints during model training and evaluation
"""

import torch


# write model arguments to text file
def write_args(cnn_backbone, w, h, transforms, anchor_sizes, batch_size, optim, lr, wd, lr_scheduler, output_path):
    """
    write txt file to output dir that has run values for changeable hyperparameters:
    - model backbone
    - image size
    - data augmentations and their ranges
    - anchor box sizes (may not change this)
    - optimizer
    - starting learning rate
    - weight decay
    - learning rate scheduler and parameters
    :return: txt file containing all values of changing arguments per model run
    """

    # collect arguments in a dict
    model_args = {'backbone': cnn_backbone,
                  'image width': w,
                  'image height': h,
                  'data augmentations': transforms,
                  'anchor box sizes': anchor_sizes,
                  'batch size': batch_size,
                  'optimizer': optim,
                  'starting lr': lr,
                  'weight decay': wd,
                  'lr_scheduler': lr_scheduler.__class__
                  }

    # write args to text file
    with open(output_path + '/model_args.txt', 'w') as f:
        for key, value in model_args.items():
            f.write('%s:%s\n' % (key, value))

# load saved model args to python dict
def load_args(output_path, filename="model_args.txt"):
    """
    :param output_path: path to folder where model args are stored
    :param filename: name of text file; probably "model_args.txt" but allow for customization
    :return:
    """
    model_args = {}
    with open(output_path + filename) as f:
        model_args = {k: v for line in f for (k, v) in [line.strip().split(":")]}
    model_args['image width'] = int(model_args['image width'])
    model_args['image height'] = int(model_args['image height'])
    model_args['batch size'] = int(model_args['batch size'])
    model_args['unbalanced'] = bool(model_args['unbalanced'])
    model_args['anchor box sizes'] = tuple(eval(model_args['anchor box sizes']))

    return model_args

# create checkpoint
def create_checkpoint(model, optimizer, epoch, current_lr, loss_history, best_loss,
                      model_type, num_classes,label2target, training_time, pred_df, results_df):
    '''
    creates checkpoint at the end of each epoch that can be used for continuing model training or
    for model evaluation

    :param model: model (see model_args.txt for backbone)
    :param optimizer: optimizer parameters
    :param epoch: training epoch just completed
    :param current_lr: current learning rate
    :param loss_history: train/val loss history
    :param best_loss: best loss
    :param model_type: model type being trained
    :param num_classes: number classes in the model
    :param label2target: label encoder dictionary
    :param training_time: list of training times (train + val) per epoch
    :param pred_df: df of test predictions
    :param results_df: df of evaluation metrics on test df
    :return: checkpoint
    '''
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch + 1,
                  'current_lr': current_lr,
                  'loss_history': loss_history,
                  'best_loss': best_loss,
                  'model_type': model_type,
                  'num_classes': num_classes,
                  'label2target': label2target,
                  'training_time': training_time,
                  'pred_df': pred_df,
                  'results_df': results_df}
    return checkpoint

# save checkpoint
def save_checkpoint(checkpoint, checkpoint_file):
    """
    save checkpoint to output directory
    :param checkpoint:
    :param checkpoint_file:
    :return:
    """
    print(" Saving model state")
    torch.save(checkpoint, checkpoint_file)

# load checkpoint
def load_checkpoint(checkpoint_file, device):
    """
    load a model checkpoint
    :param checkpoint_file: full path to a saved checkpoint
    :return: checkpoint dictionary
    """
    print(" Loading saved model state")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    return checkpoint