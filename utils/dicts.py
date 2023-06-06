# functions to write label dictionaries and create representative samples

import pandas as pd
import numpy as np
from collections import Counter

def gen_dict(df):
    """
    dictionary for the general model
    :param df:
    :return: sample df, dictionary, column to use for train/val split
    """

    # keep images only in the general categories
    df = df[df['general_category'] != "nan"].reset_index()

    # save original df to filter through later
    original_df = df

    # create dictionary of category labels
    label2target = {l: t + 1 for t, l in enumerate(df['general_category'].unique())}

    # add background class
    label2target['empty'] = 0

    # take sample of mammals and birds
    animals = df.loc[df['general_category'].isin(['mammal', 'bird'])]
    animal_sample = animals.groupby('general_category').sample(n=max_per_class, replace=False)

    # add back all vehicle images
    df = pd.concat([animal_sample, df[df['general_category'] == 'vehicle'], df[df['general_category'] == 'human']])

    # locate images within df
    df = original_df.loc[original_df['filename'].isin(df['filename'])]

    # rename label column
    df['LabelName'] = df['general_category']

    # split across category and species for train-val split
    columns2stratify = ['general_category']

    return df, label2target, columns2stratify

def fam_dict(df, max_per_class, min_per_class):
    """
    create label dict and representative sample for family model

    :param df: formatted df
    :param max_per_class: max annotations per class
    :param min_per_class: min annotations per class
    :return: sample df, label dict, col to stratify over test/val split
    """

    # list families with fewer images than category min
    too_few = list({k for (k, v) in Counter(df['family']).items() if v < min_per_class})

    # remove general bird images
    too_few.append('Bird')

    # exclude those images from the sample
    df = df[~df['family'].isin(too_few)]

    # remove rows if family is nan
    df = df[df['family'].notna()]

    # save original df to filter through later
    original_df = df

    # list families where all images are included
    fewerMax = list({k for (k, v) in Counter(df['family']).items() if v <= max_per_class})

    # list families with more images than category max
    overMax = list({k for (k, v) in Counter(df['family']).items() if v > max_per_class})

    # shuffle rows
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # setup for representative sampling across database
    balanced_df = pd.DataFrame(columns=list(df.columns))
    thresh = 100
    for label in overMax:
        temp = balanced_df
        # loop through each family
        subset_df = df[df['family'] == label]
        # list db with image numbers above and below threshold
        fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
        greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
        # add all images for db with image counts below threshold
        lt_df = subset_df[subset_df['database'].isin(fewer)]
        # sample equally from remaining db until total images reaches max per category
        size = max_per_class - lt_df.shape[0]
        ht_subset = subset_df[subset_df['database'].isin(greater)]
        ht_df = pd.DataFrame(columns=list(df.columns))
        while ht_df.shape[0] <= size:
            tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
            ht_df = pd.concat([ht_df, tmp], ignore_index=True)
            ht_df = ht_df.drop_duplicates()
            if ht_df.shape[0] > size:
                break
        # add family sample to balanced df
        fam_df = pd.concat([lt_df, ht_df], ignore_index=True)
        balanced_df = pd.concat([temp, fam_df], ignore_index=True)

    # combine representative db sample with families taking all images
    df = pd.concat([balanced_df, df[df['family'].isin(fewerMax)]])

    # locate images within df
    df = original_df.loc[original_df['filename'].isin(df['filename'])]

    # create dictionary of family labels
    label2target = {l: t + 1 for t, l in enumerate(df['family'].unique())}

    # set background class
    label2target['empty'] = 0
    pd.options.mode.chained_assignment = None

    # standardize label name
    df['LabelName'] = df['family']

    # stratify across species and family for train/val split
    columns2stratify = ['family']

    return df, label2target, columns2stratify

def spec_dict(df, max_per_class, min_per_class):
    """
    create label dict and representative sample for species model

    :param df: formatted df
    :param max_per_class: max annotations per class
    :param min_per_class: min annotations per class
    :return: sample df, label dict, col to stratify over test/val split
    """

    # list species with fewer images than category min
    too_few = list({k for (k, v) in Counter(df['species']).items() if v < min_per_class})

    # remove general bird images
    too_few.append('Bird')

    # include these species despite small sample sizes
    always_include = ['White-nosed_Coati', 'Collared_Peccary', 'Jaguarundi', 'Margay', 'Jaguar', 'Ocelot']
    # always include images from CFT databases
    cft_include = ['Cattle Fever Tick Program', 'Texas A&M']

    # filter always_include species out of the too_few list
    too_few = [e for e in too_few if e not in always_include]
    df = df[~df['common.name'].isin(too_few)]

    # save original df to filter through later
    original_df = df

    # list species with fewer than max cat images
    fewerMax = list({k for (k, v) in Counter(df['common.name']).items() if v <= max_per_class})
    # list species with greater images than max per category
    overMax = list({k for (k, v) in Counter(df['common.name']).items() if v > max_per_class})

    # shuffle rows before sampling
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # initiate representative sample df
    balanced_df = pd.DataFrame(columns=list(df.columns))

    # set threshold for db images to include
    thresh = 50
    # loop over species with num images greater than max per category
    for label in overMax:
        temp = balanced_df
        subset_df = df[df['common.name'] == label]
        # list db with image numbers above and below threshold
        fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
        greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
        # add all images for db with image counts below threshold
        lt_df = subset_df[subset_df['database'].isin(fewer)]
        # sample equally from remaining db until total images reaches max per species
        size = max_per_class - lt_df.shape[0]
        ht_subset = subset_df[subset_df['database'].isin(greater)]
        ht_df = pd.DataFrame(columns=list(df.columns))
        while ht_df.shape[0] <= size:
            tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
            ht_df = pd.concat([ht_df, tmp], ignore_index=True)
            ht_df = ht_df.drop_duplicates()
            if ht_df.shape[0] > size:
                break
        # add species sample to balanced df
        spec_df = pd.concat([lt_df, ht_df], ignore_index=True)
        balanced_df = pd.concat([temp, spec_df], ignore_index=True)

    # combine data and drop duplicates
    df = pd.concat([balanced_df, df[df['common.name'].isin(fewerMax)], df[df['database'].isin(cft_include)]])
    df = df.drop_duplicates()

    # locate images within df
    df = original_df.loc[original_df['filename'].isin(df['filename'])]

    # create dictionary of species labels
    label2target = {l: t + 1 for t, l in enumerate(df['common.name'].unique())}
    # set background class
    label2target['empty'] = 0
    pd.options.mode.chained_assignment = None

    # standardize label name
    df['LabelName'] = df['common.name']
    # stratify across species for train/val split
    columns2stratify = ['common.name']

    return df, label2target, columns2stratify

def pig_dict(df, max_per_class, min_per_class):
    """
    create label dict and representative sample for pig-only model

    :param df: formatted df
    :param max_per_class: max annotations per class
    :param min_per_class: min annotations per class
    :return: sample df, label dict, col to stratify over test/val split
    """

     # list species with fewer images than category min
    too_few = list({k for (k, v) in Counter(df['common.name']).items() if v < min_per_class})

    # remove mammals that are too small to be mistaken for pigs
    too_small = ['squirrel_spp', 'American_Badger', 'American_Marten', 'American_Mink',
                 'Jackrabbit', 'Prairie_Dog', 'Chipmunk', 'Cottontail_Rabbit', 'Spotted_Skunk',
                 'Fisher', 'Mouse_Rat', 'Nine-Banded_Armadillo', 'North_American_Beaver',
                 'North_American_Porcupine', 'Polar_Bear', 'Snowshoe_Hare', 'Striped_Skunk', 
                 'Woodchuck', 'Yellow-Bellied_Marmot']

    # include these species despite small sample sizes
    always_include = ['White-nosed_Coati', 'Collared_Peccary', 'Jaguarundi', 'Margay', 'Jaguar', 'Ocelot']
    too_few = [e for e in too_few if e not in always_include]

    # filter exclusions out of df
    df = df[~df['common.name'].isin(too_few)]
    df = df[~df['common.name'].isin(too_small)]

    #Remove birds and reptiles
    df = df[df['class'] !='Reptilia']
    df = df[df['class'] != 'Aves']

    # save original df to filter through later
    original_df = df

    # list species with fewer than max cat images
    fewerMax = list({k for (k, v) in Counter(df['common.name']).items() if v <= max_per_class})
    # list species with greater images than max per category
    overMax = list({k for (k, v) in Counter(df['common.name']).items() if v > max_per_class})

    # shuffle rows before sampling
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # initiate representative sample df
    balanced_df = pd.DataFrame(columns=list(df.columns))

    # set threshold for db images to include
    thresh = 50
    # loop over species with num images greater than max per category
    for label in overMax:
        temp = balanced_df
        subset_df = df[df['common.name'] == label]
        # list db with image numbers above and below threshold
        fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
        greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
        # add all images for db with image counts below threshold
        lt_df = subset_df[subset_df['database'].isin(fewer)]
        # sample equally from remaining db until total images reaches max per species
        size = max_per_class - lt_df.shape[0]
        ht_subset = subset_df[subset_df['database'].isin(greater)]
        ht_df = pd.DataFrame(columns=list(df.columns))
        while ht_df.shape[0] <= size:
            tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
            ht_df = pd.concat([ht_df, tmp], ignore_index=True)
            ht_df = ht_df.drop_duplicates()
            if ht_df.shape[0] > size:
                break
        # add species sample to balanced df
        spec_df = pd.concat([lt_df, ht_df], ignore_index=True)
        balanced_df = pd.concat([temp, spec_df], ignore_index=True)

    # combine data and drop duplicates
    df = pd.concat([balanced_df, df[df['common.name'].isin(fewerMax)]])
    df = df.drop_duplicates()

    # locate images within df
    df = original_df.loc[original_df['filename'].isin(df['filename'])]

    # create dictionary of species labels
    label2target = {'empty':0, 'Wild_Pig':1, 'Not_Pig':2}
    pd.options.mode.chained_assignment = None

    # standardize label name
    df['LabelName'] = np.where(df['common.name'] == 'Wild_Pig', 'Wild_Pig', 'Not_Pig')
    # stratify across species for train/val split
    columns2stratify = ['common.name']

    return df, label2target, columns2stratify


def train_database(df):
    """
    creates database of training data by class, database, etc.
    :param df: train_df
    :return: train_db_full, trian_db_simple
    """

    # reduce to single row per image
    df = df.drop_duplicates(subset='filename', keep='first')

    # rename variables
    df = df.rename(columns={'LabelName': 'Class', 'provider': 'Provider', 'site': 'Site', \
                            'database': 'Database', 'state': 'State', 'filename': 'Images'})

    # group df and count number of images by species, provider, database, state
    db_full = pd.DataFrame(df[['Images', 'Class', 'Provider', 'Database', 'State']]. \
                           groupby(['Class', 'Provider', 'Database', 'State']).agg('count')).reset_index()
    db_full.loc[db_full.duplicated('Class'), 'Class'] = ''

    # group images by species, database
    db_db = pd.DataFrame(df[['Images', 'Class', 'Database']]. \
                         groupby(['Class', 'Database']).agg('count')).reset_index()
    db_db.loc[db_db.duplicated('Class'), 'Class'] = ''

    return db_full, db_db

def encode_labels(label2target, output_path):
    """
    Reorder label2target by numerical value; export for loading into R package
    :param label2target: dictionary of labels and encoders
    :return: reordered label2target
    """
    # sort label2target by value
    sorted_labels = sorted(label2target.items(), key=lambda x:x[1])

    # convert sorted list back to dict
    label_encoder = dict(sorted_labels)

    # remove special characters
    label_encoder = {x.replace("'", ''): v for x, v in label_encoder.items()}

    # write encoder to text file
    with open(output_path + '/label_encoder.txt', 'w') as f:
        for key, value in label_encoder.items():
            f.write('%s:%s\n' % (key, value))

    return label_encoder