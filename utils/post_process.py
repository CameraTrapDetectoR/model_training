## -- Functions for assessing models post-training

import pandas as pd
from collections import Counter



def format_evals(pred_df):
    # # Drop bboxes
    pred_df = pred_df.drop(['bbox'], axis=1)
    #
    # # Rename and remove columns
    pred_df = pred_df.rename(columns={'filename': 'file_path', 'class_name': 'prediction'}).drop(['file_id'], axis=1)

    # # extract image name/structure from file_path
    image_names = pred_df['file_path']
    image_names = image_names.str.replace('D:/2016.05.19/', '')
    pred_df['image_name'] = image_names

    # # get prediction counts for each image
    cts = Counter(pred_df['file_path']).items()
    pred_counts = pd.DataFrame.from_dict(cts)
    pred_counts.columns = ['file_path', 'count']
    pred_df = pred_df.merge(pred_counts, on='file_path', how='left')

    # # separate images with one prediction and images with >1 predictions
    single_preds = pred_df[pred_df['count'] == 1]
    multi_preds = pred_df[pred_df['count'] > 1]

    # # format single preds
    single_preds.loc[single_preds['prediction'] == 'empty', 'count'] = 0
    #
    # # drop counts from multi_preds
    multi_preds = multi_preds.drop(['count'], axis=1)

    # # get new counts based on image + predicted class
    multi_cts = multi_preds.groupby(['file_path', 'prediction'])['prediction'].count().reset_index(name='count')
    #
    # # join multi_preds to new counts
    multi_preds = multi_preds.merge(multi_cts, on=['file_path', 'prediction'], how='left', copy=False)
    #
    # # filter multi_preds to one prediction per image + class group - take highest confidence
    filtr_preds = multi_preds.groupby(['file_path', 'prediction']).apply(
        lambda x: x[x['confidence'] == max(x['confidence'])])

    # join filtered multi_preds to single_preds
    preds = pd.concat([single_preds, filtr_preds], ignore_index=True).sort_values(['file_path'])

    # reorder image_name column
    preds = preds.loc[:, ['file_path', 'image_name', 'prediction', 'confidence', 'count']]

    # add columns for manual review: true_class, true_count, comments
    preds['true_class'] = ""
    preds['true_count'] = ""
    preds['comments'] = ""

    return preds