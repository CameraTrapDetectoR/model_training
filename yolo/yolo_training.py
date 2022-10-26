#################################
### CameraTrapDetectoR
### You Only Look Once (YOLO) Model Training

# performed using Yolo v5 available from PyTorch: https://pytorch.org/hub/ultralytics_yolov5/

## System Setup
import os

# determine operating system, for batch vs. local jobs
if os.name == 'posix':
    local = False
else:
    local = True

# set path based on location, local machine or remote batch job
# for IMAGE_ROOT, specify full path to folder where all/only training images are located
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu/IMAGES"
    os.chdir('/home/burnsal@colostate.edu/CameraTrapDetectoR')

# Import packages
exec(open('./yolo/yolo_imports.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Run yolo_directory_format.py to

## Load and format labels
# Load label .csv file - check to confirm most recent version
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-05-22.csv")


# delete files not in the given sample
extra = df[~df['filename'].isin(sample['filename'])]
extras = extra['filename'].tolist()
for root, dirs, files in os.walk(IMAGE_ROOT):
    for f in files:
        if os.path.join(root, f).replace(os.sep, '/') in extras:
            print(os.path.join(IMAGE_ROOT, f).replace(os.sep, '/'))




# TODO: apply splitfolders function to remaining images
# import splitfolders
#
# splitfolders.ratio(input="IMAGES",
#                    output="yolo_images",
#                    seed=22,
#                    ratio=(0.7,0.2,0.1),
#                    group_prefix=None,
#                    move=True
#                    )

# TODO: create dataset.yaml file


# TODO: Create label .txt files

