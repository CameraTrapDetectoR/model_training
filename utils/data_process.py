"""
Functions to process dataset before training model
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


# Define class size range based on model type
def class_range(model_type):
    if model_type == 'species':
        max_per_class = 10000
        min_per_class = 300
    if model_type == 'family':
        max_per_class = 10000
        min_per_class = 300
    if model_type == 'general':
        max_per_class = 35000
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

# filter out bboxes that touch image edges
def filter_partials(df):
    """
    In an attempt to filter out images with incomplete animal features, remove images with bboxes
    that touch any edge of an image
    :param df:
    :return: df with updated image list
    """
    # round bbox coordinates down to 2 decimal places
    df_round = df.round({'XMin': 2, 'YMin': 2, 'XMax': 2, 'YMax': 2})
    # collect list of detections where any coordinate touches image edge
    df_part = df_round[(df_round['XMin'] == 0.00) |
                       (df_round['XMax'] == 1.00) |
                       (df_round['YMin'] == 0.00) |
                       (df_round['YMax'] == 1.00)]
    # get unique filenames
    filenames = df_part.filename.unique()
    # filter these images out of training data
    df = df[~df['filename'].isin(filenames)]

    return df



# format variables
def format_vars(df):
    """
    misc formatting tasks
    :param df: unformatted df
    :return: df with formatting
    """
    # create copy of original species column
    df['species_org'] = df['species']

    # replace spaces with underscores in species column
    df['species'] = df['species'].replace(' ', '_', regex=True)

    # combine squirrel species into one group
    squirrels = ['Callospermophilus_lateralis', 'Glaucomys_sabrinus', 'Neosciurus_carolinensis',
                 'Otospermophilus_variegatus', 'Sciurus_aberti', 'Sciurus_griseus', 
                 'Sciurus_niger', 'Tamiasciurus_douglasii', 'Tamiasciurus-hudsonicus']
    df.loc[df['species'].isin(squirrels), 'species'] = 'squirrel_spp'

    # combine pigeons into one group
    # pigeons = ['Patagioenas_fasciata', 'Patagioenas_leucocephala']
    df.loc[df['species'].str.contains('Patagioenas', na=False), 'species'] = 'Patagioenas_spp'

    # combine doves into one group
    doves = ['Columba_livia', 'Columbina_passerina', 'Leptotila verreauxi', 
             'Streptopelia_decaocto', 'Zenaida', 'Zenaida_asiatica', 'Zenaida_macroura']
    df.loc[df['species'].isin(doves), 'species'] = 'dove_spp'

    # combine blackbirds
    blackbirds = ['Agelaius_phoeniceus', 'Euphagus_cyanocephalus', 'Xanthocephalus_xanthocephalus']
    df.loc[df['species'].isin(blackbirds), 'species'] = 'blackbird_spp'

    # combine chipmunks
    # chipmunks = ['Tamias_ruficaudus', 'Tamias_striatus']
    df.loc[df['species'].str.contains('Tamias', na=False), 'species'] = 'Tamias_spp'

    # combine cottontail rabbits
    # cottontails = ['Sylvilagus', 'Sylvilagus_audubonii', 'Sylvilagus_floridanus', 
    #                'Sylvilagus_nuttallii']
    df.loc[df['species'].str.contains('Sylvilagus', na=False), 'species'] = 'Sylvilagus_spp'

    # combine cowbirds
    # cowbirds = ['Molothrus_aeneus', 'Molothrus_ater']
    df.loc[df['species'].str.contains('Molothrus', na=False), 'species'] = 'Molothrus_spp'

    # combine egrets
    egrets = ['Ardea_alba', 'Bubulcus_ibis']
    df.loc[df['species'].isin(egrets), 'species'] = 'egret_spp'

    # combine grackles
    df.loc[df['species'].str.contains('Quiscalus', na=False), 'species'] = 'Quiscalus_spp'

    # combine jackrabbits
    jackrabbits = ['Lepus_californicus', 'Lepus_townsendii']
    #TODO confirm naming
    df.loc[df['species'].isin(jackrabbits), 'species'] = 'jackrabbit_spp'

    # combine herons
    herons = ['Ardea_herodias', 'Butorides_virescens', 'Egretta_tricolor']
    df.loc[df['species'].isin(herons), 'species'] = 'heron_spp'
    night_herons = ['Nyctanassa_violacea', 'Nycticorax_nycticorax']
    df.loc[df['species'].isin(night_herons), 'species'] = 'night_heron_spp'

    # combine owls
    owls = ['Asio_flammeus', 'Athene_cunicularia', 'Bubo_virginianus', 'Strix_varia', 
            'Tyto_alba']
    df.loc[df['species'].isin(owls), 'species'] = 'owl_spp'

    # combine prairie dogs
    df.loc[df['species'].str.contains('Cynomys', na=False), 'species'] = 'Cynomys_spp'

    # combine quails
    quails = ['Callipepla_californica', 'Colinus_virginianus', 'Oreortyx_pictus']
    df.loc[df['species'].isin(quails), 'species'] = 'quail_spp'

    # combine red foxes
    df.loc[df['species'].str.contains('Vulpes_vulpes', na=False), 'species'] = 'Vulpes_vulpes'

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

# split df into training / validation sets
def split_df(df, columns2stratify):
    """
    Takes df, columns2stratify output from the wrangle_df function and splits the dataset by the stratified column.
    70% of total data is allocated to training, 20% allocated to validation, and 10% allocated to out-of-sample test
    :param df: sample df
    :param columns2stratify: column to stratify over sampling to preserve representation across split dfs
    """

    df_unique_filename = df.drop_duplicates(subset='filename', keep='first')
    # split 70% of images into training set
    trn_ids, rem_ids = train_test_split(df_unique_filename['filename'], shuffle=True,
                                        stratify=df_unique_filename[columns2stratify],
                                        test_size=0.3, random_state=22)
    train_df = df[df['filename'].isin(trn_ids)].reset_index(drop=True)
    rem_df = df[df['filename'].isin(rem_ids)].reset_index(drop=True)
    rem_unique_filename = rem_df.drop_duplicates(subset='filename', keep='first')
    # split remaining 30% with a 2/1 validation/test split
    val_ids, test_ids = train_test_split(rem_unique_filename['filename'], shuffle=True,
                                         stratify=rem_unique_filename[columns2stratify],
                                         test_size=0.33, random_state=22)
    val_df = rem_df[rem_df['filename'].isin(val_ids)].reset_index(drop=True)
    test_df = rem_df[rem_df['filename'].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df