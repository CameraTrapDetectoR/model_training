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
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu"
    os.chdir('/home/burnsal@colostate.edu/CameraTrapDetectoR')

# Import packages
exec(open('/yolo/yolo_model_imports.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model functions
exec(open('./yolo/yolo_model_functions.py').read())

# Set model type
# options: 'general', 'family', 'species', 'pig_only'
model_type = 'species'

# Set min/max number images per category
max_per_category, min_per_category = class_range(model_type)

## Load and format labels
# Load label .csv file - check to confirm most recent version
df = pd.read_csv("./labels/varified.bounding.boxes_for.training.final.2022-05-22.csv")

# process labels .csv
df = wrangle_df(df, IMAGE_ROOT)

# create balanced training set, class dictionary, split stratifier
df, label2target, target2label, columns2stratify = define_dictionary(df, model_type)
num_classes = max(label2target.values()) + 1

# split the dataset into training and validation sets
train_df, val_df, test_df = split_df(df, columns2stratify)
# len(train_df), len(val_df), len(test_df)

