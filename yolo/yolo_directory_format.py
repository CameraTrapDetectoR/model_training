# Split data for YOLO training

# Import libraries
import os
import pandas as pd
import numpy as np
from collections import Counter
import splitfolders

############
# -- system setup
############

# determine if job is running locally
if os.name == 'posix':
    local = False
else:
    local = True

# set working directory and image pathways
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    YOLO_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos_yolo'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu/IMAGES"
    YOLO_ROOT = "/scratch/summit/burnsal@colostate.edu/YOLO_IMAGES"
    os.chdir('/projects/burnsal@colostate.edu/CameraTrapDetectoR')

##########
# -- split data into test, val, training sets
##########
splitfolders.ratio(IMAGE_ROOT, output=YOLO_ROOT,
                   seed=1337, ratio=(.7, .2, .1),
                   group_prefix=None, move=False)

##########
# -- format labels
##########

# read in label csv
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-10-19.csv")

# cross check filenames to image files in directory
# for local jobs, need to be connected to VPN here
extant = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_ROOT) for f in fn]
extant = [x.replace(IMAGE_ROOT + '/', '/').strip('/') for x in extant]

# filter df for existing images
df = df[df['filename'].isin(extant)]

# exclude partial images
df = df[df['partial.image'] == False]

# swap y-axis for images where bboxes originate in LL corner (pytorch looks for UL and LR corners)
df['YMin_org'] = df['YMin']
df['YMax_org'] = df['YMax']
df.drop(['YMax', 'YMin'], axis=1, inplace=True)
df['YMax'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMin_org']), df['YMax_org'])
df['YMin'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMax_org']), df['YMin_org'])

# check for bboxes that are too small
df['YDiff'] = df['YMax'] - df['YMin']
df['XDiff'] = df['XMax'] - df['XMin']
df_tooSmall = df[(df['YDiff'] < 0.001) | (df['XDiff'] < 0.001)]
assert df_tooSmall.shape[0] == 0, "Remove image files where bboxes are too small"

# convert bbox coordinates to YOLO format
df['x_center'] = (df['XMin'] + df['XMax']) / 2
df['y_center'] = (df['YMin'] + df['YMax']) / 2
df['w_bbox'] = df['XMax'] - df['XMin']
df['h_bbox'] = df['YMax'] - df['YMin']

# update column names for common name
df['common.name_org'] = df['common.name']
df['common.name'] = df['common.name.general']

# combine squirrels into one group
squirrels = ["Aberts_Squirrel", 'American_Red_Squirrel', 'Douglas_Squirrel', 'Eastern_Fox_Squirrel',
             'Eastern_Gray_Squirrel', 'Fox_Squirrel', 'Golden-Mantled_Ground_Squirrel', 'Gray_Squirrel',
             'Northern_Flying_Squirrel', 'Rock_Squirrel', 'Squirrel', 'Western_Gray_Squirrel']
df.loc[df['common.name'].isin(squirrels), 'common.name'] = 'squirrel_spp'

# combine doves into one group
doves = ['White-Winged_Dove', 'White-Tipped_Dove', 'Rock_Dove', 'Mourning_Dove',
         'Dove', 'Eurasian_Collared_Dove', 'Common_Ground_Dove'] # need to change folder name for common ground dove
df.loc[df['common.name'].isin(doves), 'common.name'] = 'dove_spp'

# combine egrets into one group
egrets = ['Cattle_Egret', 'Great_Egret']
df.loc[df['common.name'].isin(egrets), 'common.name'] = 'Egret'

# combine blackbirds into one group
blackbirds = ['Yellow-Headed_Blackbird', 'Red-Winged_Blackbird']
df.loc[df['common.name'].isin(blackbirds), 'common.name'] = 'blackbird_spp'



# TODO: identify other species folders to combine
Counter(df['common.name'])

# create new filename column
df['filename_org'] = df['filename']
files = []
for index, row in df.iterrows():
    files.append(row['common.name'] + "/" + row['filename_org'].split('/')[1])
df['filename'] = files

# Remove species with fewer images than class minimum
class_min = 300
too_few = list({k for (k, v) in Counter(df['common.name']).items() if v < class_min})
# remove general bird images
too_few.append('Bird')
print(too_few)

#TODO: remove these folders from YOLO_IMAGES directory (all datasets)

# exclude those images from the sample
df = df[~df['common.name'].isin(too_few)]

# Get full paths for all remaining image files
paths = pd.DataFrame([os.path.join(dp, f) for dp, dn, fn in os.walk(YOLO_ROOT) for f in fn], columns=['path'])

# Join paths to df
df['pathway'] = paths[df['filename'].apply(lambda name: paths['path'].str.contains(name)).any(0)]

# TODO: Remove any rows where pathway = NaN

# TODO: Add column for image organization in train/val/test dataset
