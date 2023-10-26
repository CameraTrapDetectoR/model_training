#!/bin/bash

# evaluate unlabeled images with CameraTrapDetectoR models in a remote instance

#SBATCH --job-name="CTD"
#SBATCH -p=medium #21 day walltime
#SBATCH -N 1    # number of nodes
#SBATCH --mail-user=user.name@domain.me
#SBATCH --mail-type=ALL

date # print start timestamp

module purge
module load miniconda
source activate ctd-deploy-model

export PYTHONPATH="/project/cameratrapdetector/model_training"

# sample call to run deploy_model.py on images from ProjectA through the species v2
# with a specified output dir and checkpoint file to load and resume
python /project/cameratrapdetector/model_training/deploy_model.py /project/cameratrapdetector/output/species_v2_cl \
/project/cameratrapdetector/path/to/ProjectA_Images \
 --output_dir /project/cameratrapdetector/results/ \
 --resume_from_checkpoint /project/cameratrapdetector/results/ProjectA_species_v2_checkpoint.csv

date # print end timestamp

# END