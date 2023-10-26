# CameraTrapDetectoR Model Training

This repository contains the Python code used to train new versions of the CameraTrapDetectoR models. 

# CameraTrapDetectoR Model Deployment    

   We also provide documentation and assistance for deploying our models via Python script on the command line. The *deploy_model.py* script provides similar output to the `deploy_model` function in the CameraTrapDetectoR package. This approach allows users to take advantage of a GPU-enabled machine or run the models via batch mode on a remote instance, for example on a high-performance computing (HPC) cluster.  

## Installation

### 1. System Prerequisites

You will need [Anaconda](https://docs.conda.io/projects/miniconda/en/latest/) and [Git](https://git-scm.com/download/win) installations to set up your model environment. If your environment can access a [Pytorch-compatible GPU](https://pytorch.org/get-started/locally/), make sure you have an updated [NVIDIA driver](https://www.nvidia.com/download/index.aspx) installed as well.

### 2. Clone Git Repository    

Open your Anaconda prompt and navigate to the directory where you'd like to store your project. Clone this repository:    

```batch
cd /path/to/project/dir
git clone https://github.com/CameraTrapDetectoR/model_training.git
```

### 3. Create Python Environment

Navigate into the model_training directory and locate a file called *deploy_model_env.yml*. This file contains the setup info and required Python packages needed to run the models. You can create your environment using the YAML file:   
```batch
cd model_training
conda env create -n ctd-deploy-model -f deploy_model_env.yml
```
or you can manually create a Python environment and install the listed packages using the package manager of your choosing.    

This step may take a while! Go get a cup of coffee or a yummy snack, and start [downloading the models](https://github.com/CameraTrapDetectoR/model_training/blob/main/README.md#5-download-models) while you wait.  

Check that your environment was successfully created, and add the repository to your Python path:    
For Windows:    
```batch
conda activate ctd-deploy-model
set PYTHONPATH=%PYTHONPATH%;\path\to\project\dir\model_training
```
For Mac/Linux:    
```batch
conda activate ctd-deploy-model
export PYTHONPATH="$PYTHONPATH:$path/to/project/dir/model_training
```    

### 4. Download Models

Download the model architecture, model weights, and auxiliary information needed to run each desired model. Model versions 2 and later are supported by this script:    
   - [General V2 Model](https://data.nal.usda.gov/system/files/general_v2_cl_1.zip)
   - [Family V2 Model](https://data.nal.usda.gov/system/files/family_v2_cl_0.zip)
   - [Species V2 Model](https://data.nal.usda.gov/system/files/species_v2_cl_0.zip)

The [user tutorial] assumes you are putting the model folder in the same project directory as this repo; you can place it anywhere you like, just adjust accordingly. 
Once you download the model folder to its final location, unzip it. **DO NOT** rename, modify, or delete anything inside this folder or your model will not run. Inside each folder are three files specific to running that model, titled *cameratrapdetector_metadata.cfg*, *model_args.txt*, and *model_checkpoint.pth*. Do not change anything! Just unzip it and leave it.    

Now you are ready to run the CameraTrapDetectoR models!    

## Use


### 1. Activate Python Environment   

Activate your Python environment and, if necessary, export the path to the model_training repo to your Python path:    
For Windows:    
```batch
conda activate ctd-deploy-model
set PYTHONPATH=%PYTHONPATH%;\path\to\project\dir\model_training
```
For Mac/Linux:    
```batch
conda activate ctd-deploy-model
export PYTHONPATH="$PYTHONPATH:$path/to/project/dir/model_training
```

### 2. The deploy_model.py script

Inside the main folder of the model_training repo, you will see a file titled *deploy_model.py*; you will pass your user arguments to this script through the command line. This script provides a similar output to the `deploy_model` function in the CameraTrapDetectoR package. Navigate into the repo and run the help command to view descriptions of the inputs:    
```batch
cd /path/to/project/dir/model_training
python deploy_model.py --help
```
![image](https://github.com/CameraTrapDetectoR/model_training/assets/54477812/11b59ab7-661e-4448-92f2-53dfd857d622)

