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
    os.chdir("C:/Users/Amira.Burns/OneDrive - USDA/Projects/CameraTrapDetectoR")
else:
    IMAGE_ROOT = "/scratch/summit/burnsal@colostate.edu"
    os.chdir('/projects/burnsal@colostate.edu/CameraTrapDetectoR')

# Import packages
exec(open('./fasterRCNN/fasterRCNN_imports.py').read())

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model functions
exec(open('./fasterRCNN/fasterRCNN_model_functions.py').read())

# set pathways
# Note: Manually change output_path and checkpoint_file to the training session being evaluated and its latest checkpoint
output_path = "./output/pig_only_2bs_8gradacc_lr005_weightdecay0005/"
checkpoint_file = output_path + "checkpoint_18epochs.pth"
eval_path = output_path + "evals/"
if not os.path.exists(eval_path):
    os.makedirs(eval_path)

# load test df
test_df = pd.read_csv(output_path + "test_df.csv")

# define latest checkpoint
checkpoint = torch.load(checkpoint_file)

# Load model type
model_type = checkpoint['model_type']

# Review loss history
loss_history = checkpoint['loss_history']
plot_losses(model_type, loss_history)

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# re-initialize the model
num_classes = checkpoint['num_classes']
model = get_model(num_classes).to(device)

# load model weights
model.load_state_dict(checkpoint['state_dict'])

# deploy model on test images
gt_df, pred_df = deploy(df=test_df, w=408, h=307)

# if returning to  previously deployed test sample, load dfs here:
output_path = "./output/20220908_fasterRCNN_pig_only_2bs_8gradaccumulation_9momentum_0005weight_decay_005lr/"
eval_path = output_path + "evals/"
gt_df = pd.read_csv(eval_path + "gt_df.csv")
pred_df = pd.read_csv(eval_path + "pred_df.csv")

# calculate mean average precision
checkpoint_file = output_path + "checkpoint_18epochs.pth"
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}


# extract predicted bboxes, confidence scores, and labels
pred_boxes = [box.strip('[').strip(']').strip(',') for box in pred_df['bbox']]
pred_boxes = np.array([np.fromstring(box, sep=', ') for box in pred_boxes])
pred_scores = pred_df['confidence'].values.tolist()
pred_labels = pred_df['class_name'].map(label2target)
# convert preds to dictionary of tensors
preds = [
    dict(
        boxes=torch.tensor(pred_boxes),
        scores=torch.tensor(pred_scores),
        labels=torch.tensor(pred_labels)
    )
]

# extract target bboxes and labels
target_boxes = [box.strip('[').strip(']').strip(', ') for box in gt_df['bbox']]
target_boxes = np.array([np.fromstring(box, sep=', ') for box in target_boxes])
target_labels = gt_df['class_name'].map(label2target)
# convert targets to tensor dictionary
target = [
    dict(
        boxes=torch.tensor(target_boxes),
        labels=torch.tensor(target_labels)
    )
]

# initialize metric
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
metric.update(preds, target)
results = metric.compute()
#TODO: save computed metrics to file

# If evaluation provides evidence of strong performance, save weights for loading into R package
path2weights = output_path + "weights_" + model_type + ".pth"
torch.save(dict(model.to(device='cpu').state_dict()), path2weights.replace('.pth', '_cpu.pth'))

# Model evaluation complete!
