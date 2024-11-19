#!/bin/bash
#SBATCH --account=C3SE-STAFF
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=01:00:00

module load pytorch
module load munch
module load torchbnn

~/my_ml_pytorch_script.py --epochs=12 --mnist_src /mimer/NOBACKUP/Datasets/MNIST/raw
