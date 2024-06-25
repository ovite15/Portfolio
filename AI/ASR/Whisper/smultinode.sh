#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script


module purge
module load Mamba/23.11.0-0
module load cudatoolkit/23.3_12.0
module load gcc/12.2.0

conda deactivate
conda activate myenv

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0

export LD_LIBRARY_PATH=/project/ckxxxxxx-xxxxxx/home/ckxxxx/myenv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/project/ckxxxxxx-xxxxxx/cache
export HF_DATASETS_CACHE=/project/ckxxxxx-xxxxx/cache

accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision bf16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    /project/ckxxxxx-xxxx/train.py
