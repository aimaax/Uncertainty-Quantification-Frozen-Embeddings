#!/bin/bash
#SBATCH --account=naiss2024-5-95
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --time=1-00:00:00

module purge

module load virtualenv/20.23.1-GCCcore-12.3.0 matplotlib/3.7.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a h5py/3.9.0-foss-2023a JupyterLab/4.0.5-GCCcore-12.3.0

module load Python/3.11.3-GCCcore-12.3.0
source ../ProbVLM/UQ_venv/bin/activate

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Fiona/1.9.5-foss-2023a
module load Fire/2017.1
module load NLTK/3.8.1
module load ftfy 

python3 train_DO.py

mv /mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/edward/slurm-$SLURM_JOB_ID* /mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/edward/output_train

