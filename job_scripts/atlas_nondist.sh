#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos='normal'
#SBATCH --job-name=mini_run1
#SBATCH --account=cameratrapdetector
#SBATCH --mail-user=Amira.Burns@usda.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module purge

module load miniconda
source activate CTDtrain

module unload miniconda
module load python      # check for latest python release
python /project/cameratrapdetector/minitrain/model_train.py
# END