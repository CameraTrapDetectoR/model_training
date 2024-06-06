# est pretrained weights on iWildCam images



# import packages
from ultralytics import YOLO
import pandas as pd


### GET PRETRAINED WEIGHTS ###

# TODO: insert code for selecting labels


# initiate yolo model
model = YOLO('yolov8s.pt')

# load  annotations
train_df = pd.read_csv('/path/to/iwildcam2022_pre_train_df.csv')
val_df = pd.read_csv('/path/to/iwildcam2022_pre_val_df.csv')

# make class dictionary
label2target = {l: t for t, l in enumerate(sorted(train_df['common_name_general'].unique()))}
# reverse dictionary to read into pytorch
target2label = {t: l for l, t in label2target.items()}

# join label2target to df
labeldf = pd.DataFrame.from_dict(label2target.items())
labeldf = labeldf.rename(columns={0:"common_name_general", 1:"label"})
train_df = train_df.merge(labeldf, how = 'left', on='common_name_general')
val_df = val_df.merge(labeldf, how = 'left', on='common_name_general')

# save list of file names for yaml
train_files = pd.DataFrame('/path/to/project/' + train_df.file_name.unique())
train_files.to_csv('/path/to/project/train_imgs.txt', index=False, header=False)
val_files = pd.DataFrame('/path/to/project/' + val_df.file_name.unique())
val_files.to_csv('/path/to/project/val_imgs.txt', index=False, header=False)

# reformat bboxes to yolo
train_df['box_w'] = train_df['xmax'] - train_df['xmin']
train_df['box_h'] = train_df['ymax'] - train_df['ymin']
train_df['x_center'] = ((train_df['xmax'] + train_df['xmin']) / 2)
train_df['y_center'] = ((train_df['ymax'] + train_df['ymin']) / 2)

val_df['box_w'] = val_df['xmax'] - val_df['xmin']
val_df['box_h'] = val_df['ymax'] - val_df['ymin']
val_df['x_center'] = ((val_df['xmax'] + val_df['xmin']) / 2)
val_df['y_center'] = ((val_df['ymax'] + val_df['ymin']) / 2)

# write labels to txt files
for i in range(len(train_list)):
    image_info = train_df[train_df.file_name == train_list[i]]
    txt_name = image_info.iloc[0][['file_name']].str.replace(".jpg", ".txt").item()
    image_info[['label', 'x_center', 'y_center', 'box_w', 'box_h']].to_csv('/path/to/project/labels/' + txt_name, sep=" ", index=False, header=False)
for i in range(len(val_list)):
    image_info = val_df[val_df.file_name == val_list[i]]
    txt_name = image_info.iloc[0][['file_name']].str.replace(".jpg", ".txt").item()
    image_info[['label', 'x_center', 'y_center', 'box_w', 'box_h']].to_csv('/path/to/project/labels/' + txt_name, sep=" ", index=False, header=False)

# train the model
results = model.train(data='/path/to/project/pretrain_iwilcam_yolov8.yaml', epochs=300, imgsz=640, rect=True, optimizer='SGD', name ="test1")

# Validate the model
best_wts = YOLO('/path/to/project/test1/weights/best.pt')
metrics = best_wts.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category


### HYPERPARAMETER TUNING ###

# load packages
from collections import Counter
import shutil
import numpy as np
import os
from PIL import Image

# set dir roots
IMAGE_ROOT = '/project/90daydata/cameratrapdetector/trainimages/'
script_root = '/project/cameratrapdetector/train_runs/species_20240516/'

# load annotations
train_df = pd.read_csv('/project/cameratrapdetector/labels/species_train.csv')
val_df = pd.read_csv('/project/cameratrapdetector/labels/species_val.csv')

train_df['LabelName'] = train_df['common.name.combined']
val_df['LabelName'] = val_df['common.name.combined']

# make label dict
label2target = {l: t for t, l in enumerate(sorted(train_df['LabelName'].unique()))}
# reverse dictionary to read into pytorch
target2label = {t: l for l, t in label2target.items()}

# attach label number to df
def attach_label(df, label2target=label2target):
    labeldf = pd.DataFrame.from_dict(label2target.items())
    labeldf = labeldf.rename(columns={0:"LabelName", 1:"label"})
    df = df.merge(labeldf, how = 'left', on='LabelName')
    return df
train_df = attach_label(train_df)
val_df = attach_label(val_df)

# write filenames to img.txt files; shuffle filenames 
def new_filepaths(df):
    file_df = pd.DataFrame({'LabelName' : df.LabelName, 'photo_name' : df.photo_name}).drop_duplicates()
    file_df['filename'] = script_root + 'images/' + df.LabelName + '/' + df.photo_name
    files = pd.DataFrame(file_df.filename).sample(frac=1)
    return(files)
    
#train_files = pd.DataFrame(script_root + 'images/' + train_df.LabelName + '/' + train_df.photo_name.unique()).sample(frac=1)
train_files = new_filepaths(train_df)
train_files.to_csv(script_root + 'train_imgs.txt', index=False, header=False)
# val_files = pd.DataFrame(script_root + 'images/' + val_df.LabelName + '/' + val_df.photo_name.unique()).sample(frac=1)
val_files = new_filepaths(val_df)
val_files.to_csv(script_root + 'val_imgs.txt', index=False, header=False)

# make new folders for images by class
label_root = script_root + 'labels/'
for labels in label2target.keys():
    os.mkdir(label_root + labels)

 # write label txt files
def write_label_txt(df):

    df['photo_name'] = df.filename.str.split(pat="/", expand=True)[1]
    file_list = df.photo_name.unique()
    
    for i in range(len(file_list)):
        image_info = df[df['photo_name']==file_list[i]]
        try:
            txt_name = image_info['LabelName'].head(1).astype(str).item() + '/' + image_info['photo_name'].head(1).str.replace(".jpg", ".txt", case = False).item()
            image_info[['label', 'x_center', 'y_center', 'box_w', 'box_h']].to_csv(script_root + 'labels/' + txt_name, sep=" ", index=False, header=False)
        except ValueError:
            print(f'Oops! Unable to print text for {image_info.filename}.') 
write_label_txt(train_df)
write_label_txt(val_df)

# make new folders for images by class
img_root = script_root + 'images/'
for img in label2target.keys():
    os.mkdir(img_root + img)

# copy images to project folder
def copy_images(df):
    name_df = df[['filename', 'photo_name', 'LabelName']].drop_duplicates()
    move_df = pd.DataFrame({'org': IMAGE_ROOT + name_df.filename,
                           'dest': script_root + 'images/' + name_df.LabelName + '/' + name_df.photo_name})
    for i in range(len(move_df)):
        if not os.path.exists(str(move_df.iloc[i]['dest'])):
            try:
                shutil.copy2(str(move_df.iloc[i]['org']), str(move_df.iloc[i]['dest']))
            except:
                print(f'Oops! Unable to move image: {name_df.iloc[i]['filename']}.') 
    return move_df
move_train = copy_images(train_df)
move_val = copy_images(val_df)


script_root = '/project/cameratrapdetector/train_runs/species_20240404/'

# load models
wcmodel = YOLO('/project/90daydata/cameratrapdetector/iwildcam/pretrain_yolo/03282024/weights/best.pt')
dmodel = YOLO('yolov8s.pt')


# PreTrained iwildcam weights ##
resume baseline training of iwildcam pretrained weights
wcresults = wcmodel.train(data= script_root + 'data.yaml', epochs=300, plots=True,
                      imgsz=640, rect=True, optimizer='SGD', save_period=1, project=script_root, name='wbase')
# Validate the model
best_wts = YOLO(script_root + 'wbase/weights/best.pt')

metrics = best_wts.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# add in some data augmentation
results = wcmodel.train(data='/project/cameratrapdetector/train_runs/species_20240404/data.yaml', epochs=300, 
                        imgsz=640, rect=True, optimizer='SGD', save_period=1, plots=True,
                        project=script_root, name='wbase_augs', 
                        degrees=20, shear=20, perspective=0.00001, mixup=0.05, copy_paste=0.05, fliplr=0.3, 
                        auto_augment='randaugment')

# Validate the model
best_wts = YOLO(script_root + 'wbase_augs/weights/best.pt')

metrics = best_wts.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# make hyperparameter grid
optimizers = ['SGD', 'Adam', 'AdamW']
lr0 = [0.005, 0.01, 0.05]
coslr = ['', 'True']
momentum = [0.9, 0.93, 0.95]
weight_decay = [0.0001, 0.0005]

hyp_grid = np.array([(o,l,c,m,w) for o in optimizers for l in lr0 for c in coslr for m in momentum for w in weight_decay])
hyp_df = pd.DataFrame({'optimizer': hyp_grid[:, 0], 'lr0': hyp_grid[:, 1], 'cos_lr': hyp_grid[:, 2],
                      'momentum': hyp_grid[:,3], 'weight_decay': hyp_grid[:,4]})

# test different hyperparameter combinations
for i in range(len(hyp_df)):
    # define name
    name = 'wbase_hypgrid_' + str(i)

    # define hyperparameters outside run
    optz = str(hyp_df['optimizer'].iloc[i])
    lr = float(hyp_df['lr0'].iloc[i])
    cos_lr = bool(hyp_df['cos_lr'].iloc[i])
    mmtm = float(hyp_df['momentum'].iloc[i])
    wtdc = float(hyp_df['weight_decay'].iloc[i])
    
    
    # train model
    results = wcmodel.train(data = script_root + 'data.yaml', epochs=300, plots=True, 
                           imgsz=640, rect=True, save_period=10, project=script_root,
                           name=name, optimizer=optz, lr0=lr , cos_lr= cos_lr,
                           momentum=mmtm, weight_decay=wtdc)
    # validate model
    best_wts = YOLO(script_root + '/' + name + '/weights/best.pt')
    metrics = best_wts.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category


# do it again with data augmentation
for i in range(len(hyp_df)):
    # define name
    name = 'wbase_augs_hypgrid_' + str(i)

    # define hyperparameters outside model train run
    optz = str(hyp_df['optimizer'].iloc[i])
    lr = float(hyp_df['lr0'].iloc[i])
    cos_lr = bool(hyp_df['cos_lr'][i])
    mmtm = float(hyp_df['momentum'][i])
    wtdc = float(hyp_df['weight_decay'].iloc[i])
    
    # train model
    results = wcmodel.train(data = script_root + 'data.yaml', epochs=300, plots=True, 
                           imgsz=640, rect=True, save_period=10, project=script_root,
                           degrees=20, shear=20, perspective=0.00001, mixup=0.05, 
                           copy_paste=0.05, fliplr=0.3, auto_augment='randaugment',
                           name=name, optimizer=optz, lr0=lr , cos_lr= cos_lr,
                           momentum=mmtm, weight_decay=wtdc)
    # validate model
    best_wts = YOLO(script_root + '/' + name + '/weights/best.pt')
    metrics = best_wts.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

## Run inference on out of sample set

# set root dir
oos_img_root = script_root + 'hp_tuning_oos/'

# make list of different model weights
models_to_test = ['wbase/', 'wbase_augs', 'wbase_augs_hypgrid_0/', 'wbase_augs_hypgrid_1/', 'wbase_augs_hypgrid_4/', 'wbase_augs_hypgrid_5/',
                  'wbase_augs_hypgrid_6/', 'wbase_augs_hypgrid_7/', 'wbase_augs_hypgrid_8/', 'wbase_hypgrid_1/', 'wbase_hypgrid_10/', 'wbase_hypgrid_11/', 
                  'wbase_hypgrid_12/', 'wbase_hypgrid_13/', 'wbase_hypgrid_2/', 'wbase_hypgrid_3/', 'wbase_hypgrid_4/', 'wbase_hypgrid_5/', 
                  'wbase_hypgrid_6/', 'wbase_hypgrid_8/', 'wbase_hypgrid_9/']

# define label/target dict
target2label = {0: 'Common_Raccoon', 1: 'Common_Raven', 2: 'Mountain_Lion', 3: 'White-Tailed_Deer', 4: 'Wild_Pig', 5: 'Wild_Turkey'}

# loop through models
for i in range(len(models_to_test)):
    # load weights
    model = YOLO(script_root + models_to_test[i] + 'weights/best.pt')

    # run model on predictions
    results = model.predict(source = oos_paths, visualize = True, agnostic_nms = True, 
                        project = 'species_20240404/' + models_to_test[i], name = 'oos_predictions',
                        save = True, save_txt = True, save_conf = True)

    

# END