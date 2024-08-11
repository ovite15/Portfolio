#!/bin/bash
#SBATCH -p gpu                # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                 # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=4            # Specify the number of GPUs
#SBATCH --ntasks-per-node=1        # Specify tasks per node
#SBATCH -t 12:00:00            # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900118           # Specify project name
#SBATCH -J IT        # Specify job name

module reset
module load Mamba
module load cudatoolkit/23.3_12.0

conda deactivate
conda activate /lustrefs/disk/project/lt900118-ai24o6/IT/mytorch


#  m \
yolo task=detect mode=train epochs=50 batch=128 plots=True \
 model='./weights/yolov10x.pt'\
 data='./data1.yaml' device=[0,1,2,3] \

