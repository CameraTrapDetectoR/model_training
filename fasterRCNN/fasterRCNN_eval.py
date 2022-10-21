#############################
## - CameraTrapDetectoR
## - Model Evaluation
#############################

## DISCLAIMER!!!
# Use this file only for evaluating a trained model
# If a model needs additional training, refer to './fasterRCNN_resume_training.py'

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

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model functions
exec(open('./fasterRCNN/fasterRCNN_model_functions.py').read())

# set pathways
# Note: Manually change output_path and checkpoint_file to the training session being evaluated and its latest checkpoint
output_path = "./output/20221006_fasterRCNN_species_2bs_8gradaccumulation_9momentum_001weightdecay_01lr/"
checkpoint_file = output_path + "checkpoint_" + "18epochs.pth"
eval_path = output_path + "evals/"
if not os.path.exists(eval_path):
    os.makedirs(eval_path)

# load test df
test_df = pd.read_csv(output_path + "test_df.csv")
image_infos = test_df.filename.unique()

# define latest checkpoint
checkpoint = torch.load(checkpoint_file, map_location=device)

# Load model type
model_type = checkpoint['model_type']

# Review loss history
loss_history = checkpoint['loss_history']
plot_losses(model_type, loss_history)

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# make output dictionaries based on model type
if model_type == 'pig_only':
    # save new column to modify
    test_df['class_name'] = test_df['family']
    # map not-pig to all non-Suidae labels
    test_df.loc[test_df['class_name'] != 'Suidae', ['class_name']] = 'Non-Suidae'
    # make new output dict
    out_label2target = {l: t + 1 for t, l in enumerate(test_df['class_name'].unique())}
    out_label2target['empty'] = 0
    out_target2label = {t: l for l, t in out_label2target.items()}
else:
    out_label2target = label2target
    out_target2label = target2label

# re-initialize the model
num_classes = checkpoint['num_classes']
model = get_model(num_classes).to(device)

# load model weights
model.load_state_dict(checkpoint['state_dict'])

# set dummy height and width for empty prediction bboxes
w = 408
h = 307

# setup holders for predictions and targets
pred_df = []
target_df = []

# deploy model on test images
model.eval()
for i in tqdm(range(len(image_infos))):
    # define dataset and dataloader
    dfi = test_df[test_df['filename'] == image_infos[i]]
    dsi = DetectDataset(df=dfi, image_dir=IMAGE_ROOT, w=408, h=307, transform=val_transform)
    dli = DataLoader(dsi, batch_size=1, collate_fn=dsi.collate_fn, drop_last=True)

    # extract image, bbox, and label info
    input, target = next(iter(dli))
    tbs = dsi[0][1]['boxes']
    image = list(image.to(device) for image in input)

    # run input through the model
    output = model(image)[0]

    # extract prediction bboxes, labels, scores
    bbs, labels, confs = filter_preds(output, threshold=0.1)

    # relabel predictions and targets for pig_only model
    if model_type == 'pig_only':
        # index the three new classes
        pigs = labels == 31
        nonpig = labels != (0 or 31)
        # reassign labels
        labels[pigs] = 2
        labels[nonpigs] = 1

    # perform classwise non-maximum suppression based on IOU threshold
    # TODO: rethink how this works since dataset does not have any overlapping predictions
    ixs = batched_nms(bbs, confs, labels, 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    # format predictions
    bbs = bbs.tolist()
    confs = confs.tolist()
    labels = labels.tolist()

    # save predictions and targets
    if len(bbs) == 0:
        pred_df_i = pd.DataFrame({
            'filename': image_infos[i],
            'file_id': image_infos[i][:-4],
            'class_name': 'empty',
            'confidence': 1,
            'bbox': [0, 0, w, h]
        })
    else:
        pred_df_i = pd.DataFrame({
            'filename': image_infos[i],
            'file_id': image_infos[i][:-4],
            'class_name': [out_target2label[a] for a in labels],
            'confidence': confs,
            'bbox': bbs
        })
    tar_df_i = pd.DataFrame({
        'filename': image_infos[i],
        'file_id': image_infos[i][:-4],
        'class_name': [out_target2label[a] for a in dfi['class_name'].tolist()],
        'bbox': tbs.tolist()
    })
    pred_df.append(pred_df_i)
    target_df.append(tar_df_i)

# concatenate preds and targets into dfs
pred_df = pd.concat(pred_df).reset_index(drop=True)
target_df = pd.concat(target_df).reset_index(drop=True)

# save dfs to csv
target_df.to_csv(eval_path + "target_df.csv")
pred_df.to_csv(eval_path + "pred_df.csv")

# update pig_only class labels before computing evaluation metrics
#TODO: test upstream edits then delete this

# if model_type == 'pig_only':
#     # save original values of class name
#     pred_df['class_name_org'] = pred_df['class_name']
#     target_df['class_name_org'] = target_df['class_name']
#
#     # replace non-pig families with the same generic class name
#     pred_df.loc[(pred_df['class_name'] != 'Suidae') & (pred_df['class_name'] != 'empty'), ['class_name']] = 'Non-Suidae'
#     target_df.loc[target_df['class_name'] != 'Suidae', ['class_name']] = 'Non-Suidae'
#
#     # update class dictionaries
#     label2target = {l: t + 1 for t, l in enumerate(target_df['class_name'].unique())}
#     label2target['empty'] = 0
#     target2label = {t: l for l, t in label2target.items()}

# open target and pred dfs if working in a new session
eval_path = "./output/20221006_fasterRCNN_species_2bs_8gradaccumulation_001weightdecay_01lr/evals/"
pred_df = pd.read_csv(eval_path + "pred_df.csv")
target_df = pd.read_csv(eval_path + "target_df.csv")


# extract predicted bboxes, confidence scores, and labels
preds = []
targets = []

for i in tqdm(range(len(image_infos))):
    # extract predictions and targets for an image
    p_df = pred_df[pred_df['filename'] == image_infos[i]]
    t_df = target_df[target_df['filename'] == image_infos[i]]

    # if reloaded as .csv
    pred_boxes = [box.strip('[').strip(']').strip(',') for box in p_df['bbox']]  # this needed if reloading .csv
    pred_boxes = np.array([np.fromstring(box, sep=', ') for box in pred_boxes])
    # if continuing from env
    # pred_boxes = [box for box in p_df['bbox']]
    pred_scores = p_df['confidence'].values.tolist()
    pred_labels = p_df['class_name'].map(label2target)

    # convert preds to dictionary of tensors
    pred_i = {
        'boxes': torch.tensor(pred_boxes),
        'scores': torch.tensor(pred_scores),
        'labels': torch.tensor(pred_labels.values)
    }

    # extract target bboxes and labels
    # if reloaded as .csv
    target_boxes = [box.strip('[').strip(']').strip(', ') for box in t_df['bbox']]
    target_boxes = np.array([np.fromstring(box, sep=', ') for box in target_boxes])
    # if continuing from env
    # target_boxes = [box for box in t_df['bbox']]
    target_labels = t_df['class_name'].map(label2target)
    # convert targets to tensor dictionary
    target_i = {
        'boxes': torch.tensor(target_boxes),
        'labels': torch.tensor(target_labels.values)
    }

    # add current image preds and targets to dictionary list
    preds.append(pred_i)
    targets.append(target_i)

# initialize metric
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
metric.update(preds, targets)
results = metric.compute()

# save results to file
# TODO: associate classes in mAP output to class names
results_df = pd.DataFrame(results)
results_df.to_csv(eval_path + "results_df.csv")

# TODO: add detection-level metrics from paper: precision, recall, and F1 score


# If evaluation provides evidence of strong performance, save weights for loading into R package
path2weights = output_path + "weights_" + model_type + "_cpu.pth"
torch.save(dict(model.to(device='cpu').state_dict()), path2weights)

# Model evaluation complete!
