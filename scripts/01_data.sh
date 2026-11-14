#!/usr/bin/env bash
# Step 1: parameters, snapshot and test solutions, POD basis of one experiment.
#   bash scripts/01_data.sh configs/semilinear_k1000.yaml
source "$(dirname "$0")/common.sh"
CFG="$1"
log "parameters: $CFG"; python -m rbno.generate_parameters "$CFG"
for store in snapshots test; do
  wait_for_cpus "$NPROCS"
  log "solve $store: $CFG"; mpi "$NPROCS" python -m rbno.solve "$CFG" --store "$store"
done
log "POD: $CFG"; python -m rbno.pod "$CFG"
