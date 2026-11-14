#!/usr/bin/env bash
# Step 8: reduced minimizers s_r^dagger (Gauss-Newton in the r POD coefficients, warm-started at
# P_r s_h) for the test set and the sweep over r, and their evaluation (needs steps 1 and 2).
#   bash scripts/08_reduced_opt.sh configs/quasilinear.yaml
source "$(dirname "$0")/common.sh"
CFG="$1"
wait_for_cpus "$NPROCS"
log "reduced minimizers $CFG"
mpi "$NPROCS" python -m rbno.reduced_opt "$CFG"
wait_for_cpus "$NPROCS"
mpi "$NPROCS" python -m rbno.evaluate "$CFG" --method reduced_opt
