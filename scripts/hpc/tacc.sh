#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# TACC runtime environment (Stampede3, Vista)
#
# Usage (interactive session or SLURM job script, *before* running AFINNI):
#     source scripts/hpc/tacc.sh
#
# This is a RUNTIME helper, not an install step. Installing never needs modules:
# mpi4py ships as a wheel and the `gpu` extra pulls the CUDA libraries from pip.
# What this does is load the cluster's MPI so mpi4py binds to it at runtime.
# On the cluster, install with the `mpi-system` extra (e.g. `make install-hpc`),
# never `mpi`: the pip-bundled MPICH would shadow the cluster's MPI and its
# fast interconnect.
#
# Local machines never need this file. To support another cluster, copy it to
# scripts/hpc/<site>.sh and adjust the module lines.
# -----------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "error: this script must be sourced, not executed:  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

HOST_FQDN=$(hostname -f 2>/dev/null)
HOST_SHORT=$(hostname)

# Vista is checked first: its compute nodes are also named cNNN-NNN, which the
# Stampede3 short-name pattern below would otherwise match.
if [[ "${TACC_SYSTEM:-}" == vista || "$HOST_FQDN" == *vista* ]]; then
    echo "[afinni] TACC Vista detected"
    # Keep Vista's default stack (no `module purge`): it already provides the
    # Open MPI that mpi4py binds to at runtime. Loading is idempotent and
    # restores it if a shell rc swapped compilers (gcc <-> nvidia).
    module load gcc/15.1.0 cuda/12.9 openmpi/5.0.5

    echo "[afinni] Loaded: gcc/15.1.0 cuda/12.9 openmpi/5.0.5"

elif [[ "${TACC_SYSTEM:-}" == stampede3 || "$HOST_FQDN" == *stampede3* || "$HOST_SHORT" =~ ^[cg][0-9]{3}- ]]; then
    echo "[afinni] TACC Stampede3 detected"
    module purge
    module load gcc/15.1.0 cuda/12.8 impi

    # Force Intel MPI to use local shared memory over H100 GPU nodes
    export FI_PROVIDER=shm

    echo "[afinni] Loaded: gcc/15.1.0 cuda/12.8 impi | FI_PROVIDER=shm"

else
    echo "[afinni] error: '$HOST_SHORT' is not a recognised TACC system; nothing was changed." >&2
    echo "[afinni] This script is only needed on TACC. On a local machine just use 'uv run'." >&2
    return 1
fi

# Fail early (on the login/compute node) rather than deep inside a job.
if [[ -x .venv/bin/python ]]; then
    .venv/bin/python -c "import src.infra.backend as b; b.get_mpi(); print('[afinni] MPI:', b._resolve_mpi()[2])" \
        || echo "[afinni] warning: AFINNI could not initialise; run 'make install-hpc' first." >&2
fi
echo "[afinni] Ready. Launch MPI jobs with e.g.:  ibrun .venv/bin/python afinni_launcher.py ..."
