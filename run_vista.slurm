#!/bin/bash
#SBATCH -J Falkon_CPU_N=1e8_M=1e6
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# go to the directory where you ran sbatch from (repo root if you submit there)
cd "$SLURM_SUBMIT_DIR"

# Create logs directory BEFORE SBATCH tries to write to it
mkdir -p logs

echo "Starting SLURM job at $(date)"
echo "Working directory: $PWD"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"

module reset
module load gcc/15.1.0 
module load cuda
module load python3/3.11.8   # adjust if Vista uses a different python module name

echo "Modules loaded successfully"
echo "Python version: $(python --version)"

# activate your venv (make sure it was created on Vista)
source .venv/bin/activate
echo "Virtual environment activated"

# Make imports like "import utils" work from tests/
cd tests

echo "Changed to tests directory: $PWD"
echo "Starting Falkon.py at $(date)"

python -u Falkon.py

echo "Falkon.py completed at $(date)"