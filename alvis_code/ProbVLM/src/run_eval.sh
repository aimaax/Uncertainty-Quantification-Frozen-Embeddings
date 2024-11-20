#!/bin/bash
#SBATCH --account=naiss2024-5-95
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:30:00

# don't forget to run this on the virtual environment
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# module load pycocotools
module load Fiona/1.9.5-foss-2023a 
module load Fire/2017.1
module load NLTK/3.8.1
module load ftfy # not found but the code runs :/?

python3 uncertainty_estimates.py
