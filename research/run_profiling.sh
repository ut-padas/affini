#!/bin/bash
mkdir -p profiles
CASES=("10D_coarse")

for CASE in "${CASES[@]}"; do
    echo "Profiling Case: $CASE"
    nsys profile --trace=cuda,nvtx,osrt --output="profiles/profile_${CASE}" --force-overwrite=true --stats=true uv run python profile_runner.py --case "$CASE"
done

# Tar the profiles directory
tar -czf profiles-tacc-jan-30.tar.gz profiles/