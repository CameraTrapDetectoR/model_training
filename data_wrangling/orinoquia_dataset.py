# Comb through Orinoquia Camera Trap Dataset for images of species of interest

# -- Imports
import os
import json
import csv
import pandas as pd
from collections import Counter

# -- Set Dirs
image_dir = 'C:/Temp/public'
dest_dir = 'G:/!ML_training_datasets/Orinoquia_Camera_Traps'

# -- Read in metadata
path2metadata = dest_dir + "/orinoquia_camera_traps_metadata.json"
m = open(path2metadata)
metadata = json.load(m)
annotations = metadata['annotations']

# -- Create dictionary of species to include
include = [3, 13, 16, 31, 35, 36, 40, 42, 46]
species_dict = {
    3: 'Collared_Peccary',
    13: 'Puma',
    16: 'South_American_Coati',
    31: 'Ocelot',
    35: 'Jaguarundi',
    36: 'White_Lipped_Peccary',
    40: 'Unknown_Peccary',
    42: 'Margay',
    46: 'Jaguar'
}


# -- Create dataframe of images with filenames
df = pd.DataFrame.from_records(annotations)
df = df[['image_id', 'category_id']]
df['common.name'] = df['category_id'].map(species_dict)


# -- Reformat image_id and create filename
df['file_loc'] = df.image_id.str.replace('_', '/')
df['image_id'] = df['image_id'].str[4:]
df['filename'] = df['common.name'] + '/' + df['image_id']

# -- Add columns for database, location, taxonomic classifications
df['database'] = 'Orinoquia_Camera_Traps'


# -- Separate dataframes
df_remove = df[~df['category_id'].isin(include)]
df_keep = df[df['category_id'].isin(include)]