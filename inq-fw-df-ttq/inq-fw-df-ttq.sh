#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-wkui
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=70:00:00
#SBATCH --job-name=inq-ttq-offline
#SBATCH --output=./implementation/inq-fw-df-ttq/output-%j.out
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...

module laod python/3.6.3
source ./implementation/VE/bin/activate
# export PYTHONPATH=/home/pasha/implementation
python ./implementation/inq-fw-df-ttq/solution.py


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

