#############################
## - CameraTrapDetectoR
## - Resume Previous Training
#############################

## DISCLAIMER!!!
# Use this file only for resuming a previously-initiated training session
# If initiating a new training session, use './fasterRCNN_training.py' file

## System Setup
import os

# determine operating system, for batch vs. local jobs
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
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu"
    os.chdir('/projects/burnsal@colostate.edu/CameraTrapDetectoR')

# Import packages
exec(open('./fasterRCNN/fasterRCNN_imports.py').read())

# Load model functions
exec(open('./fasterRCNN/fasterRCNN_model_functions.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True


# set pathways
# Note: Manually change output_path and checkpoint_file to the training session being resumed and its latest checkpoint
output_path = "./output/test/startdate_20220810_fasterRCNN_species_4bs_4gradaccumulation_9momentum_0005weight_decay_005lr/"
checkpoint_file = output_path + "checkpoints/modelstate_12epochs.pth"

# load training and validation data files
train_df = pd.read_csv(output_path + "train_df.csv")
val_df = pd.read_csv(output_path + "val_df.csv")

# define latest checkpoint
checkpoint = torch.load(checkpoint_file)

# initialize the model
num_classes = checkpoint['num_classes']
model = get_model(num_classes).to(device)

# set number of epochs
# Note: this number should reflect total epochs across this and all previous training sessions
# Ex: if model ran for 5 epochs and num_epochs = 10, then this training session will run for 5 epochs
num_epochs = 18
assert checkpoint['epoch'] < num_epochs, "Training already completed for set num_epochs"

# define other hyperparameters
lr = checkpoint['lr']
# momentum, weight decay, batch_size, grad_accumulation should be the same as initialized training
momentum = 0.9
weight_decay = 0.0005
batch_size = 4 # Note: effective batch size = batch_size * grad_accumulation
grad_accumulation = 4
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)

# load checkpoint
model, optimizer, lr_scheduler, epoch, loss_history, best_loss, model_type, label2target = load_checkpoint(checkpoint_file)
target2label = {t: l for l, t in label2target.items()}  # reverse dictionary for pytorch input


# Load datasets
train_ds = DetectDataset(df=train_df, image_dir=IMAGE_ROOT, w=408, h=307, transform=train_transform)
val_ds = DetectDataset(df=val_df, image_dir=IMAGE_ROOT, w=408, h=307, transform=val_transform)

# Get dataloaders
# note: weighted random sampler may be producing training errors
# train_loader, val_loader = get_dataloaders(train_df, train_ds, val_ds, model_type, num_classes, batch_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collate_fn, drop_last=True)

# train the model
print("training model on ", device)
for epoch in range(epoch, num_epochs):
    # set learning rate and print epoch number
    current_lr = get_lr(optimizer)
    print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

    # training pass
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        # send data to device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]

        # forward pass
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss = loss / grad_accumulation  # normalize loss to account for batch accumulation

        # backward pass
        loss.backward()

        # optimizer step every x=grad_accumulation batches
        if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            print(f'Batch {batch_idx} / {len(train_loader)} | Train Loss: {loss:.4f}')

        # update loss
        running_loss += float(loss)

    # record training loss
    loss_history['train'].append(running_loss / len(train_loader))
    print('train loss: %.6f' % (running_loss / len(train_loader)))

    # validation pass
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            model.train()  # obtain losses without defining forward method
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
    checkpoint = create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type,
                                   num_classes, label2target)
    checkpoint_file = output_path + "checkpoint_" + str(epoch + 1) + "epochs.pth"
    save_checkpoint(checkpoint, checkpoint_file)

# END
print("Model training complete!")
# For additional training, update checkpoint/paths and rerun this script
# For inference, proceed to './fasterRCNN_eval.py'
