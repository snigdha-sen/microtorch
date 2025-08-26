#!/bin/bash

#SBATCH --mail-user=AhmedR14@cardiff.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=SANDI_WAND_Model_Test
#SBATCH --partition=cubric-rocky8
#SBATCH -o SWC_%j.out
#SBATCH -e SWC_%j.err

echo "Starting job on $(hostname) at $(date)"
python /batch_fit.py ${SLURM_ARRAY_TASK_ID}
