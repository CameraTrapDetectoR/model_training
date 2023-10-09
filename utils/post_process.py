## -- Functions for assessing models post-training

import pandas as pd
from collections import Counter

import cv2


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


# plot predictions on an image
def plot_image(image, bbs, confs, labels, img_path, IMAGE_PATH, PRED_PATH):
    """
    Plot predicted bounding boxes
    :param image: original image file
    :param bbs: predicted bounding boxes
    :param confs: predicted confidence scores
    :param labels: predicted class labels
    :param img_path: filepath to original image
    :param IMAGE_PATH: user-provided path to image directory
    :param PRED_PATH: directory for placing plotted images
    :return: plotted image
    """

    # get image width and height
    img_h, img_w = image.shape[:2]

    # loop through predictions
    for box in range(len(bbs)):
        # extract box into coordinates
        xmin = int(bbs[box][0] * img_w)
        ymin = int(bbs[box][1] * img_h)
        xmax = int(bbs[box][2] * img_w)
        ymax = int(bbs[box][3] * img_h)

        # plot box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
        # add conf score and class label
        cv2.putText(image, "conf = " + str(round(confs[box], 2)), (xmin + 20, ymin + 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(labels[box]), (xmin + 20, ymin),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # make new image name
    new_name = img_path.replace(IMAGE_PATH + "/", "").replace("/", "_")
    # save plotted image
    cv2.imwrite(PRED_PATH + new_name, image)
