#!/usr/bin/env bash
# Hyperparameter search for the RBNO of one experiment (needs step 1 of it): every line of a
# grid file is one run "<id> <key=value> ...", trained on the training rows and ranked by the
# best validation loss (the test set is not used). Two lanes run concurrently, one per GPU of
# $GPUS, each with TUNE_WORKERS residual workers; finished runs are skipped.
#   bash scripts/07_tune.sh configs/semilinear_k1000.yaml configs/tune/semilinear_stage1.txt
# Output: results/<exp>/tune/<id>/{train_summary.json,loss_history.jsonl,config.yaml},
#         results/<exp>/tune/summary.csv
source "$(dirname "$0")/common.sh"
CFG="$1"
GRID="$2"
WORKERS="${TUNE_WORKERS:-30}"
NAME="$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['name'])")"
mkdir -p "logs/tune/$NAME"
read -ra LANES <<< "$GPUS"

gpu_vacant() {
  local used procs
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | wc -l)
  (( used < 1000 && procs == 0 ))
}

lane() {  # lane <index> <gpu>: runs every len(LANES)-th line of the grid
  local idx="$1" gpu="$2" n=0 id sets
  while read -r id sets; do
    [[ -z "$id" || "$id" == \#* ]] && continue
    n=$((n + 1))
    (( (n - 1) % ${#LANES[@]} == idx )) || continue
    [[ -f "results/$NAME/tune/$id/train_summary.json" ]] && { log "tune $NAME/$id done; skipping"; continue; }
    wait_for_cpus "$((WORKERS + 1))"
    until gpu_vacant "$gpu"; do log "tune $NAME/$id: waiting for GPU $gpu"; sleep 120; done
    log "tune $NAME/$id on GPU $gpu: $sets"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" mpi "$((WORKERS + 1))" python -m rbno.train "$CFG" --device cuda:0 \
      --tag "tune/$id" --set $sets < /dev/null > "logs/tune/$NAME/$id.log" 2>&1 \
      || log "tune $NAME/$id FAILED (see logs/tune/$NAME/$id.log)"
    sleep 20   # let the next lane's process count see this one
  done < "$GRID"
}

for i in "${!LANES[@]}"; do
  lane "$i" "${LANES[$i]}" &
  sleep 60
done
wait
python -m rbno.tune_summary "$CFG"
