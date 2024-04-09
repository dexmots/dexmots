#!/bin/bash

#SBATCH --partition=juno 
#SBATCH --account=juno 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time="7-0"
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="test-dmanip"
#SBATCH --output=logs/juno-%j.out
#SBATCH --error=logs/juno-%j.err # add this line to redirect stderr to a file
#SBATCH --mail-user="krshna@stanford.edu"
#SBATCH --mail-type=END,FAIL,REQUEUE

# List out some useful information.
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo ""
# echo ${1}
# echo ${2}
# echo ""
source $HOME/.bashrc
conda activate dmanip
cd /juno/u/ksrini/diff_manip/scripts
python train.py alg=shac_new task=humanoid

# Send an email upon completion.
MAIL_SUBJECT="'SLURM Job_id=${SLURM_JOBID} Log'"
MAIL_FILE="$(pwd -P)/logs/juno-${SLURM_JOBID}.out"
MAIL_CMD="mail -s ${MAIL_SUBJECT} krshna@stanford.edu < ${MAIL_FILE}"