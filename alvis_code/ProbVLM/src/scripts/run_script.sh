#!/bin/bash
#SBATCH --account=naiss2024-5-95
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:30:00

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# torchvision and torch bnn should be included in the PyTorch-bundle
# module load pycocotools
module load Fiona/1.9.5-foss-2023a 
module load Fire/2017.1
module load NLTK/3.8.1
module load ftfy # not found but the code runs :/?

#/mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/ProbVLM/src/main.py
python3 main.py
