#!/usr/bin/env bash
# Step 4: discrete stability test (needs steps 1 and 2).   bash scripts/04_stability.sh configs/hyperelastic.yaml
source "$(dirname "$0")/common.sh"
CFG="$1"
NP="${STAB_PROCS:-50}"
wait_for_cpus "$NP"
log "stability $CFG with $NP ranks"
mpi "$NP" python -m rbno.stability "$CFG"
