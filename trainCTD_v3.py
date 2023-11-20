"""
Script to set up training sample for training CameraTrapDetectoR version 3
Last Updated: 11/15/2023
Code Authors: Amira Burns
"""

# Imports
import os
import torch
import pandas as pd

from utils.data_setload import DetectDataset
from torch.utils.data import DataLoader

from utils.hyperparameters import get_anchors, get_transforms
from models.backbones import load_fasterrcnn

from torchvision.ops import nms

from models.model_inference import prepare_results, \
    calculate_metrics

from torch.optim import SGD, lr_scheduler
from utils.hyperparameters import get_lr

import time
from datetime import timedelta
from tqdm import tqdm
import copy
from utils.checkpoints import write_args, \
    create_checkpoint, save_checkpoint


#######
## -- Prepare System and Data for Model Training
#######


# Set paths
IMAGE_ROOT = 'path/to/!VarifiedPhotos'
os.chdir("path/to/Projects/CameraTrapDetectoR")


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Set model type: options = 'general', 'family', 'species', 'pig_only'
model_type = 'species'

# load annotations - samples created in the R script `create_training_sample_v3`
train_df = pd.read_csv("./labels/v3_species_train_df.csv")
val_df = pd.read_csv("./labels/v3_species_val_df.csv")
test_df = pd.csv("./labels/v3_species_test_df.csv")

# create LabelName column based on model_type
train_df['LabelName'] = train_df[model_type]
val_df['LabelName'] = val_df[model_type]
test_df['LabelName'] = test_df[model_type]

# create dictionary of species labels
label2target = {l: t + 1 for t, l in enumerate(train_df['species'].unique())}
# set background class
label2target['empty'] = 0
# reverse dictionary to read into pytorch
target2label = {t: l for l, t in label2target.items()}
# define number of classes
num_classes = max(label2target.values()) + 1

# set image dimensions for training
# TODO: determine ideal training w and h
w = 408
h = 307

# define data augmentation pipelines
# note augmentations as a string to save in model arguments
transforms = 'shear, rotate, huesat, brightcont, safecrop, hflip'

train_transform, val_transform = get_transforms(transforms, w, h)

# Load PyTorch Datasets
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=w, h=h,
                         label2target=label2target, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=w, h=h,
                       label2target=label2target, transform=val_transform)

# define anchor boxes based on image size
anchor_sizes, anchor_gen = get_anchors(h=h)

# define model backbone
cnn_backbone = 'resnet'

# initialize model
model = load_fasterrcnn(cnn_backbone, num_classes, anchor_gen).to(device)
params = [p for p in model.parameters() if p.requires_grad]

# define number of training epochs
num_epochs = 50

# define batch size
batch_size = 32
batch_size_org = batch_size

# boolean to use gradient accumulation
use_grad = True

# set denominator if using gradient accumulation
if use_grad:
    # set number of gradients to accumulate before updating weights
    grad_accumulation = 8
    # effective batch size = batch_size * grad_accumulation
    batch_size = batch_size_org // grad_accumulation

# define dataloaders
train_loader = DataLoader(train_ds, batch_size=batch_size,
                          collate_fn=train_ds.collate_fn,
                          drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size,
                        collate_fn=train_ds.collate_fn, drop_last=True)

# define starting lr, wd, momentum
# TODO: review training results to confirm these
lr = 0.01
wd = 0.001
momentum = 0.9

# load optimizer
# TODO: create function to choose this based on optim
optim = 'SGD'
optimizer = SGD(params=params, lr=lr, weight_decay=wd, momentum=momentum)

# set learning rate scheduler
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# make output directory and filepath
output_path = "./output/v3/" + model_type + "_fasterRCNN_" + cnn_backbone + "_" + \
              time.strftime("%Y%m%d") + "_" + time.strftime("%H%M") + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# write model args to text file
write_args(model_type, cnn_backbone, w, h, transforms, anchor_sizes,
           batch_size_org, optim, lr, wd, lr_scheduler, output_path)

#######
## -- Train the Model
#######


# define starting weights
# TODO: incorporate function for loading existing model weights when training iteratively
best_model_wts = copy.deepcopy(model.state_dict())

# define starting loss
best_loss = float('inf')

# create empty list to save train/val losses
loss_history = {
    'train': [],
    'val': []
}

# create empty list to save training time per epoch
training_time = []

# define starting epoch
epoch = range(num_epochs)[0]

# training/validation loop
for epoch in range(epoch, num_epochs):
    # record start time
    t_start = time.time()

    # set learning rate
    current_lr = get_lr(optimizer)

    # print epoch info
    print('Epoch {}, current lr={}'.format(epoch + 1, current_lr))

    ## -- TRAIN PASS
    # initialize training pass
    model.train()
    running_loss = 0.0

    # loop through train loader
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        # send data to device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]

        # forward pass, calculate losses
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        # TODO: determine if optimizer.zero_grad() needs to move
        if use_grad:
            # normalize loss to account for batch accumulation
            loss = loss / grad_accumulation
            # backward pass
            loss.backward()
            # optimizer step every x=grad_accumulation batches
            if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

        else:
            # backward pass
            loss.backward()
            # optimizer step
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
            print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

        # update loss
        running_loss += loss.item()

    # average loss across entire epoch
    epoch_train_loss = running_loss / len(train_loader)

    # record training loss
    loss_history['train'].append(epoch_train_loss)

    ## -- VALIDATION PASS

    # initialize validation pass, validation loss
    model.train(False)
    running_val_loss = 0.0

    # loop through val loader
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            model.train()  # obtain losses without defining forward method
            # move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # collect losses
            val_losses = model(images, targets)
            val_loss = sum(loss for loss in val_losses.values())

            if use_grad:
                # normalize loss based on gradient accumulation
                val_loss = val_loss / grad_accumulation
                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(val_loader)):
                    # reset gradients
                    optimizer.zero_grad()
                    print(f'Batch {batch_idx} / {len(val_loader)} | Val Loss: {val_loss:.4f}')

            else:
                # reset gradients
                optimizer.zero_grad()
                print(f'Batch {batch_idx} / {len(val_loader)} | Val Loss: {val_loss:.4f}')

            # update loss
            running_val_loss += float(val_loss)

    # average val loss across the entire epoch
    epoch_val_loss = running_val_loss / len(val_loader)

    # record validation loss
    loss_history['val'].append(epoch_val_loss)

    ## -- Run evaluation on test set every 10 epochs

    if (epoch + 1) %% 10 == 0:
        # define test image list
        test_infos = test_df.filename.unique()

        with torch.no_grad():
            # create placeholders for targets and predictions
            pred_df = []
            target_df = []

            # deploy model on test images
            model.eval()
            for i in tqdm(range(len(test_infos))):
                # define dataset and dataloader
                dfi = test_df[test_df['filename'] == test_infos[i]]
                dsi = DetectDataset(df=dfi, image_dir=IMAGE_ROOT, w=w, h=h, label2target=label2target,
                                    transform=val_transform)
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
                idx = torch.where(confs > 0.01)

                # filter to predictions that meet the threshold
                bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

                # perform non-maximum suppression on remaining predictions
                ixs = nms(bbs, confs, iou_threshold=0.5)

                bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

                # normalize bboxes
                norms = torch.tensor([1 / w, 1 / h, 1 / w, 1 / h])
                bbs *= norms

                # format predictions
                bbs = bbs.tolist()
                confs = confs.tolist()
                labels = labels.tolist()
                class_names = [target2label[a] for a in labels]

                if len(bbs) == 0:
                    pred_df_i = pd.DataFrame({
                        'filename': test_infos[i],
                        'class_name': 'empty',
                        'confidence': 1,
                        'bbox': [[0, 0, 0, 0]]
                    })
                else:
                    pred_df_i = pd.DataFrame({
                        'filename': test_infos[i],
                        'class_name': class_names,
                        'confidence': confs,
                        'bbox': bbs
                    })
                tar_df_i = pd.DataFrame({
                    'filename': test_infos[i],
                    'class_name': dfi['LabelName'].tolist(),
                    'bbox': tbs.tolist()
                })
                pred_df = pd.concat([pred_df, pred_df_i], ignore_index=True)
                target_df = pd.concat([target_df, tar_df_i], ignore_index=True)

            # calculate eval metrics
            preds, targets = prepare_results(pred_df=pred_df, target_df=target_df,
                                             image_infos=test_infos, label2target=label2target)

            results_df = calculate_metrics(preds=preds, targets=targets, target2label=target2label)

    ## -- UPDATE MODEL

    # record model weights based on val_loss
    if val_loss < best_loss:
        # update best loss
        best_loss = val_loss
        # update model weights
        best_model_wts = copy.deepcopy(model.state_dict())

    # adjust learning rate
    lr_scheduler.step(val_loss)

    # load best model weights if current epoch did not produce best weights
    if current_lr != get_lr(optimizer):
        model.load_state_dict(best_model_wts)

    # calculate training time per epoch
    elapsed = time.time() - t_start
    elapsed = str(timedelta(seconds=elapsed))
    training_time.append(elapsed)

    # save model state
    checkpoint = create_checkpoint(model, optimizer, epoch, current_lr, loss_history,
                                   best_loss, model_type, num_classes, label2target, training_time,
                                   pred_df, results_df)
    checkpoint_file = output_path + "checkpoint_" + str(epoch + 1) + "epochs.pth"
    save_checkpoint(checkpoint, checkpoint_file)

    print('Model trained for {} epochs'.format(epoch + 1))




