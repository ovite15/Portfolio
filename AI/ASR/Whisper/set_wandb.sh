module purge
module load Mamba/23.11.0-0
module load cudatoolkit/23.3_12.0
module load gcc/12.2.0

conda deactivate
conda activate myenv

WANDB_CONFIG_DIR=/project/"ckxxxxxx-xxxxxx"/cache wandb login
