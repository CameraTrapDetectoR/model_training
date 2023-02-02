########################
## - CameraTrapDetectoR
## - RCNN Model Training
#######################

## DISCLAIMER!!!
# Use this file only for initiating a *new* model training session
# If resuming a previous training session, use './fasterRCNN_resume_training.py' file

## System Setup

import os

# determine operating system, for batch vs. local jobs
import torch.cuda

if os.name == 'posix':
    local = False
else:
    local = True

# set path based on location, local machine or remote batch job
# for IMAGE_ROOT, specify full path to folder where all/only training images are located
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/90daydata/cameratrapdetector/fasterrcnn"
    os.chdir('/project/cameratrapdetector')

# Import packages
exec(open('./fasterRCNN/fasterRCNN_imports.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model functions
exec(open('./fasterRCNN/fasterRCNN_model_functions.py').read())

# Set model type
# options: 'general', 'family', 'species', 'pig_only'
model_type = 'species'

# Set min/max number images per category
max_per_category, min_per_category = class_range(model_type)

## Load and format labels
# Load label .csv file - check to confirm most recent version
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")

# wrangle df
df = wrangle_df(df, IMAGE_ROOT)

# create balanced training set, class dictionary, split stratifier
df, label2target, target2label, columns2stratify = define_dictionary(df, model_type)
num_classes = max(label2target.values()) + 1

# review sample df
# df.shape
# Counter(df['LabelName'])
# print(label2target)

# split the dataset into training and validation sets
train_df, val_df, test_df = split_df(df, columns2stratify)
# len(train_df), len(val_df), len(test_df)


# Load PyTorch dataset
# Note: if no data augmentation desired for training data, use val_transform - performs only necessary pre-processing
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=408, h=307, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=408, h=307, transform=val_transform)

# test image
# Note: execute this block with the same image a few times to see the augmentations
# img, target = train_ds[44]
# print('image size:', img.shape, type(img))
# print('target: ', target)
# show_img_bbox(img, target)

# initialize model
model = get_model(num_classes).to(device)

# define hyperparameters
#TODO: revisit learning rate and weight decay once training is complete, if needed
lr = 0.005
momentum = 0.9
weight_decay = 0.0005
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)
num_epochs = 30
batch_size = 4 # Note: effective batch size = batch_size * grad_accumulation
grad_accumulation = 4

# define PyTorch data loaders
# note: weighted random sampler may be producing training errors
train_loader, val_loader = get_dataloaders_even(train_df, train_ds, val_ds, num_classes, batch_size)
# train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
# val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collate_fn, drop_last=True)


# make output directory and filepaths
output_path = "./output/" + time.strftime("%Y%m%d_") + "fasterRCNN_" + model_type + "_" + \
              str(batch_size) + 'bs_' + str(grad_accumulation) + 'gradaccumulation_' + \
              str(momentum).replace('0.', '') + "momentum_" + str(weight_decay).replace('0.', '') + \
              "weight_decay_" + str(lr).replace('0.', '') + "lr/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save datasets for reference
train_df.to_csv(output_path + "train_df.csv")
val_df.to_csv(output_path + "val_df.csv")
test_df.to_csv(output_path + "test_df.csv")

################
## - TRAIN MODEL
################
print("training model on ", device)
# define starting weights, starting loss
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
# create empty list to save losses
loss_history = {
    'train': [],
    'val': []
}

# train the model
for epoch in range(num_epochs):
    # set learning rate and print epoch number
    current_lr = get_lr(optimizer)
    print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

    # training pass
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        # send data to device
        images = list(image.to(device) for image in images)
        targets = [{'boxes':t['boxes'].to(device), 'labels':t['labels'].to(device)} for t in targets]

        # forward pass
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        # normalize loss to account for batch accumulation
        loss = loss / grad_accumulation

        # backward pass
        loss.backward()

        # optimizer step every x=grad_accumulation batches
        if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

        # update loss
        running_loss += loss.item()

    # record training loss
    loss_history['train'].append(running_loss/len(train_loader))
    print('train loss: %.6f' % (running_loss / len(train_loader)))

    # validation pass
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            model.train() # obtain losses without defining forward method
            # move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # collect losses
            val_losses = model(images, targets)
            val_loss = sum(loss for loss in val_losses.values())

            # normalize loss based on gradient accumulation
            val_loss = val_loss / grad_accumulation
            if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(val_loader)):
                optimizer.zero_grad()
                print(f'Batch {batch_idx} / {len(val_loader)} | Val Loss: {val_loss:.4f}')

            # update loss
            running_val_loss += float(val_loss)

        # record validation loss
        val_loss = running_val_loss / len(val_loader)
        loss_history['val'].append(running_val_loss / len(val_loader))
        print('val loss: %.6f' % (running_val_loss / len(val_loader)))

    # compare validation loss
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

    # save model state
    checkpoint = create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes, label2target)
    checkpoint_file = output_path + "checkpoint_" + str(epoch+1) + "epochs.pth"
    save_checkpoint(checkpoint, checkpoint_file)

# END
print("Model training complete!")
# For inference, proceed to './fasterRCNN_eval.py'
# To resume a previous training session, proceed to './fasterRCNN_resume_training.py'