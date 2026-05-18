#!/bin/bash
# -----------------------------------------------------------------------------
# TACC Cluster Environment Auto-Activation Script
# Usage: source env_tacc.sh
# -----------------------------------------------------------------------------

# Capture hostname details to determine the environment
HOST_FQDN=$(hostname -f 2>/dev/null)
HOST_SHORT=$(hostname)

if [[ "$HOST_FQDN" == *stampede3* || "$HOST_SHORT" =~ ^c[0-9]{3}- || "$HOST_SHORT" =~ ^g[0-9]{3}- ]]; then
    echo "===================================================="
    echo "  [HPC Env] Detected: TACC Stampede3 Node"
    echo "===================================================="
    module purge
    module load gcc/15.1.0 cuda/12.8 impi
    
    # Force Intel MPI to use local shared memory over H100 GPU nodes
    export FI_PROVIDER=shm
    
    echo "--> Modules Loaded: gcc/15.1.0, cuda/12.8, impi"
    echo "--> Environment Set: FI_PROVIDER=shm"

elif [[ "$HOST_FQDN" == *vista* ]]; then
    echo "===================================================="
    echo "  [HPC Env] Detected: TACC Vista Node"
    echo "===================================================="
    module purge
    module load nvidia/25.9 gcc/15.1.0 cuda/12.9
    
    echo "--> Modules Loaded: nvidia/25.9, gcc/15.1.0, cuda/12.9"

else
    echo "===================================================="
    echo "  [HPC Env] Note: Local or Unrecognized System"
    echo "===================================================="
    echo "--> Leaving active environment untouched."
fi

echo "--> Setup complete! You can run your projects via 'uv run'."