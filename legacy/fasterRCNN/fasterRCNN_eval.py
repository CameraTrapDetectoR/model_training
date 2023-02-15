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
import torch.jit

if os.name == 'posix':
    local = False
else:
    local = True

# set path based on location, local machine or remote batch job
# for IMAGE_ROOT, specify full path to folder where all/only training images are located
if local:
    IMAGE_ROOT = 'G:/!ML_training_datasets/!VarifiedPhotos'
    os.chdir("/")
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
output_path = "./output/20221006_fasterRCNN_species_2bs_8gradaccumulation_9momentum_001weightdecay_01lr/"
checkpoint_file = output_path + "checkpoint_" + "20epochs.pth"
eval_path = output_path + "evals/"
if not os.path.exists(eval_path):
    os.makedirs(eval_path)

# load test df
test_df = pd.read_csv(output_path + "test_df.csv")
image_infos = test_df.filename.unique()

# define latest checkpoint
checkpoint = torch.load(checkpoint_file, map_location=device)

# Load model type
model_type = checkpoint['model_type']

# Review loss history
loss_history = checkpoint['loss_history']
# plot_losses(model_type, loss_history)

# load dictionaries
label2target = checkpoint['label2target']
target2label = {t: l for l, t in label2target.items()}

# make output dictionaries based on model type
if model_type == 'pig_only':
    # save new column to modify
    test_df['class_name'] = test_df['family']
    # map not-pig to all non-Suidae labels
    test_df.loc[test_df['class_name'] != 'Suidae', ['class_name']] = 'Non-Suidae'
    # make new output dict
    out_label2target = {l: t + 1 for t, l in enumerate(test_df['class_name'].unique())}
    out_label2target['empty'] = 0
    out_target2label = {t: l for l, t in out_label2target.items()}
else:
    out_label2target = label2target
    out_target2label = target2label

# re-initialize the model
num_classes = checkpoint['num_classes']
model = get_model(num_classes)

model = fasterrcnn_resnet50_fpn_v2()

# save model architecture for loading into R package
#TODO: troublehsoot; this is crashing RStudio
# with torch.no_grad():
#     s = torch.jit.script(model())
#     s.save(output_path + "/fasterrcnnArch_" + model_type + "_test.pt")

model.eval()
s = torch.jit.script(model.to(device='cpu'))
torch.jit.save(s, output_path + "/fasterrcnnArch_" + model_type + "_test.pt")

# test reloading model architecture - works
model = torch.jit.load(output_path + "/fasterrcnnArch_" + model_type + "_test.pt")

# load model weights
model.load_state_dict(checkpoint['state_dict'])

# deploy model
pred_df, target_df = deploy_model(df=test_df, w=408, h=307, model_type=model_type,
                                  classwise=True, score_thresh=0.2, iou_thresh=0.1)

# save dfs to csv
target_df.to_csv(eval_path + "target_df.csv")
pred_df.to_csv(eval_path + "pred_df.csv")

# plot confidence scores
# plot_scores(pred_df)


# open target and pred dfs if working in a new session
eval_path = output_path + "evals/"
pred_df = pd.read_csv(eval_path + "pred_df.csv")
target_df = pd.read_csv(eval_path + "target_df.csv")

# calculate evaluation metrics
results_df = eval_metrics(pred_df=pred_df, target_df=target_df,
                          out_label2target=out_label2target, out_target2label=out_target2label, format='csv')

# save results to file
# TODO: give unique naming convention so folder can store multiple results dependent on images, thresholds being evaluated
results_df.to_csv(eval_path + "results_df.csv")


# Save model weights for loading into R package
path2weights = output_path + "weights_" + model_type + "_cpu.pth"
torch.save(dict(model.to(device='cpu').state_dict()), f=path2weights)

# Model evaluation complete!
