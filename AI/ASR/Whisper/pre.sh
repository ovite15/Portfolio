#!/bin/bash
#SBATCH -p gpu                # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                 # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=4           # Specify the number of GPUs
#SBATCH --ntasks-per-node=1        # Specify tasks per node
#SBATCH -t 2:00:00            # Specify maximum time limit (hour: minute: second)
#SBATCH -A "sbalance"           # Specify project name
#SBATCH -J IT        # Specify job name

module reset
module load Mamba
module load cudatoolkit/23.3_12.0

conda deactivate
conda activate myenv

python predict.py
