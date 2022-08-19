#############################
## - CameraTrapDetectoR
## - Model Evaluation
#############################

## DISCLAIMER!!!
# Use this file only for evaluating a trained model
# If a model needs additional training, refer to './fasterRCNN_resume_training.py'

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
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects")
else:
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu"
    os.chdir('/projects/burnsal@colostate.edu/CameraTrapDetectoR')

# Import packages
exec(open('./CameraTrapDetectoR/fasterRCNN/fasterRCNN_imports.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model functions
exec(open('./CameraTrapDetectoR/fasterRCNN/fasterRCNN_model_functions.py').read())

# set pathways
# Note: Manually change output_path and checkpoint_file to the training session being evaluated and its latest checkpoint
output_path = "./CameraTrapDetectoR_data/output/test/startdate_20220810_fasterRCNN_species_4bs_4gradaccumulation_9momentum_0005weight_decay_005lr/"
checkpoint_path = output_path + "checkpoints/"
checkpoint_file = checkpoint_path + "modelstate_" + "18epochs.pth.tar"
eval_path = output_path + "evals/"
if not os.path.exists(eval_path):
    os.makedirs(eval_path)

# load test df
test_df = pd.read_csv(output_path + "test_df.csv")

# define latest checkpoint
checkpoint = torch.load(checkpoint_file)


# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# re-initialize the model?
num_classes = checkpoint['num_classes']
model = get_model(num_classes)

# deploy model on test images
gt_df, pred_df = deploy(test_df, sample=False)

