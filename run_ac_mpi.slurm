#!/bin/bash
#SBATCH --job-name=adapt_cluster       # Job name
#SBATCH --output=sl-logs/cluster_%j.out        # Output file name (%j = job ID)
#SBATCH --error=sl-logs/cluster_%j.err         # Error file name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=30                    # Number of tasks (processes)
#SBATCH --partition=gg                 # Target the Grace-Grace queue
#SBATCH --time=02:00:00                # Time limit (hh:mm:ss)

# Execute the Python script with the specified parameters
source .venv/bin/activate
ibrun python adaptive_clustering_mpi.py -f \
    --dim 2 \
    --tol 1e-3 \
    --n-train 1e7