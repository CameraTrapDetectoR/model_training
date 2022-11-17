# Comb through Orinoquia Camera Trap Dataset for images of species of interest

# -- Imports
import os
import json
import csv
import pandas as pd

# -- Set Dirs
image_dir = 'C:/Temp/public'
dest_dir = 'G:/!ML_training_datasets/Orinoquia_Camera_Traps'

# -- Read in metadata
path2metadata = dest_dir + "/orinoquia_camera_traps_metadata.json"
m = open(path2metadata)
metadata = json.load(m)
annotations = metadata['annotations']

# -- Create empty dataframe to hold annotations
df = 


