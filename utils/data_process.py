"""
Functions to process dataset before training model
"""
import os
import numpy as np
import pandas as pd
from collections import Counter


# Define class size range based on model type
def class_range(model_type):
    if model_type == 'species':
        max_per_class = 10000
        min_per_class = 300
    if model_type == 'family':
        max_per_class = 10000
        min_per_class = 300
    if model_type == 'general':
        max_per_class = 30000
        min_per_class = 5000
    if model_type == 'pig_only':
        max_per_class = 10000
        min_per_class = 300
    return max_per_class, min_per_class

# match annotations to existing images
def existing_images(df, IMAGE_ROOT):
    """
    walk through IMAGE_ROOT and remove annotations no associated with an image in the directory

    :param df: raw annotations
    :param IMAGE_ROOT: image directory root
    :return: df filtered to full images in the IMAGE_ROOT directory
    """
    # fix special characters in file names
    # 07.22.22 AB: uncomment this line on CURC, comment out when local
    # df['filename'] = df['filename'].str.replace("'","").replace(" ","")

    # cross check filenames to image files in directory
    # for local jobs, need to be connected to VPN here
    extant = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_ROOT) for f in fn]
    extant = [x.replace(IMAGE_ROOT + '/', '/').strip('/') for x in extant]

    # filter df for existing images
    df = df[df['filename'].isin(extant)]

    # exclude partial images
    df = df[df['partial.image'] == False]

    return df

# orient bboxes
def orient_boxes(df):
    """
    function to orient box annotations so (XMin, YMin) corresponds to UL corner of bbox for PyTorch model.
    swap y-axis for images where bboxes originate in LL corner
    :param df: df
    :return: df with updated bbox coordinates
    """
    # rename y-coords
    df['YMin_org'] = df['YMin']
    df['YMax_org'] = df['YMax']
    # drop originals
    df.drop(['YMax', 'YMin'], axis=1, inplace=True)
    # swap y-coods where bbox-origin is lower left corner
    df['YMax'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMin_org']), df['YMax_org'])
    df['YMin'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMax_org']), df['YMin_org'])

    return df

# format variables
def format_vars(df):
    """
    misc formatting tasks
    :param df: unformatted df
    :return: df with formatting
    """
    # update column names for common name
    df['common.name_org'] = df['common.name']
    df['common.name'] = df['common.name.general']

    # combine squirrel species into one group
    squirrels = ["Aberts_Squirrel", 'American_Red_Squirrel', 'Douglas_Squirrel', 'Eastern_Fox_Squirrel',
                 'Eastern_Gray_Squirrel', 'Fox_Squirrel', 'Golden-Mantled_Ground_Squirrel', 'Gray_Squirrel',
                 'Northern_Flying_Squirrel', 'Rock_Squirrel', 'Squirrel', 'Western_Gray_Squirrel']
    df.loc[df['common.name'].isin(squirrels), 'common.name'] = 'squirrel_spp'

    # combine pigeons into one group
    pigeons = ['Band-Tailed_Pigeon', ' White-Crowned_Pigeon']
    df.loc[df['common.name'].isin(pigeons), 'common.name'] = 'pigeon_spp'

    # combine doves into one group
    doves = ['Common_Ground Dove', 'Dove', 'Eurasian_Collared_Dove', 'Rock_Dove',
             'White-Tipped_Dove']
    df.loc[df['common.name'].isin(doves), 'common.name'] = 'dove_spp'

    # change the taxonomic classifications for vehicle
    df.loc[df['common.name'] == 'Vehicle', ['genus', 'species', 'family', 'order', 'class']] = 'vehicle'

    # add category for general model
    conditions = [(df['class'] == 'Mammalia') & (df['common.name'] != "Human") & (df['common.name'] != "Vehicle"),
                  (df['class'] == "Aves"),
                  (df['common.name'] == 'Human'),
                  (df['common.name'] == "Vehicle")]
    choices = ['mammal', 'bird', 'human', 'vehicle']
    df['general_category'] = np.select(conditions, choices, default=np.NAN)

    return df

