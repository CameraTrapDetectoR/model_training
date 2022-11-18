# Comb through Caltech Camera Trap Dataset for images of species of interest

# -- Imports
import os
import json
import pandas as pd
from collections import Counter

# -- Set directories
image_dir = 'C:/Temp/cct_images'
dest_dir = 'G:/!ML_training_datasets/Caltech_Camera_Traps'


# -- Load metadata
path2metadata = dest_dir + "/caltech_bboxes_20200316.json"
m = open(path2metadata)
metadata = json.load(m)
annotations = metadata['annotations']

# -- Create dictionary of species to include
species_dict = {
    1: 'Opossum',
    3: 'Raccoon',
    6: 'Bobcat',
    7: 'Skunk',
    9: 'Coyote',
    16: 'Domestic_Cat', # verify this
    39: 'Wild_Pig',
    40: 'Mountain_Lion'
}
include = [1, 3, 6, 7, 9, 16, 39, 40]

# -- Put annotations into dataframe
df = pd.DataFrame.from_records(annotations)

# TODO: split bbox column into four columns for each number
# corresponding to COCO annotation format: XMin, YMin, Width, Height
# Need to get image dimensions before normalizing coordinates
