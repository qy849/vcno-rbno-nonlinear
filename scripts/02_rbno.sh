#!/usr/bin/env bash
# Step 2: train the RBNO on the residual, predict the test set, evaluate, time.
#   bash scripts/02_rbno.sh configs/quasilinear.yaml
# Training uses one vacant GPU among $GPUS and NPROCS residual workers (NPROCS + 1 processes).
source "$(dirname "$0")/common.sh"
CFG="$1"
NAME="$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['name'])")"
wait_for_cpus "$((NPROCS + 1))"
GPU="$(pick_gpu)"
if [[ -f "checkpoints/$NAME/rbno/best_val.pt" ]]; then
  log "checkpoints/$NAME/rbno/best_val.pt exists; skipping training (delete it to retrain)"
else
  log "train RBNO $NAME on GPU $GPU with $NPROCS workers"
  CUDA_VISIBLE_DEVICES="$GPU" mpi "$((NPROCS + 1))" python -m rbno.train "$CFG" --device cuda:0
fi
log "predict / project / evaluate $NAME"
CUDA_VISIBLE_DEVICES="$GPU" python -m rbno.predict "$CFG" --device cuda:0
python -m rbno.project "$CFG"
wait_for_cpus "$NPROCS"
for m in rbno projection; do mpi "$NPROCS" python -m rbno.evaluate "$CFG" --method "$m"; done
wait_for_cpus 4
log "timing $NAME (single process)"
python -m rbno.timing "$CFG"
