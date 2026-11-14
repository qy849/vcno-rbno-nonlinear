#!/usr/bin/env bash
# Shared settings for all pipeline scripts. Source it:  source "$(dirname "$0")/common.sh"
#
# Environment knobs (export before calling a script to override):
#   CONDA_ENV   conda environment with DOLFINx 0.10 + PyTorch       (default: rbno)
#   NPROCS      MPI ranks for CPU stages (solves, evaluation)       (default: 60)
#   GPUS        candidate GPU ids for training                      (default: "0 1")
#   MAX_PROCS   cap on concurrent pipeline processes               (default: 64)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-rbno}"
NPROCS="${NPROCS:-60}"
GPUS="${GPUS:-0 1}"
MAX_PROCS="${MAX_PROCS:-64}"

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX}")" != "$CONDA_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/baselines${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# FFCx/DOLFINx JIT cache on local disk: dozens of ranks reading it from a network
# filesystem make start-up take minutes.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/rbno-cache-$(id -un)}"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p logs
# Byte-compile once: many MPI ranks importing from a network filesystem is otherwise slow.
python -m compileall -q src baselines >/dev/null 2>&1 || true

log() { echo "[$(date '+%F %T')] $*"; }

# Processes of this pipeline (counted against MAX_PROCS; other jobs of the user are not).
PIPELINE_PATTERN='python -m (rbno\.|fno|pca_net|fc_pino|rbno_data_loss|fe_nodal|native_grid)'

# Block until $1 more processes fit: the pipeline's python processes stay <= MAX_PROCS and the
# node's 1-minute load average leaves room for them (other users share the node).
wait_for_cpus() {
  local need="${1:-$NPROCS}" cores load mine
  cores=$(nproc --all)
  while :; do
    load=$(awk '{print int($1)}' /proc/loadavg)
    mine=$(ps -u "$(id -un)" -o args= | grep -cE "$PIPELINE_PATTERN" || true)
    if (( mine + need <= MAX_PROCS && load + need <= cores )); then return 0; fi
    log "waiting for CPUs: pipeline processes $mine, load $load, need $need"
    sleep 120
  done
}

# Print the index of the first GPU of $GPUS with no compute process and < 1 GB in use;
# wait otherwise. Jobs are started with CUDA_VISIBLE_DEVICES=<index> and --device cuda:0,
# so no CUDA context can be created on any other GPU.
pick_gpu() {
  local g used procs
  while :; do
    for g in $GPUS; do
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      procs=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | wc -l)
      if (( used < 1000 && procs == 0 )); then echo "$g"; return 0; fi
    done
    log "waiting for a vacant GPU among: $GPUS" >&2
    sleep 120
  done
}

# mpirun wrapper: rbno's Open MPI, no core binding (other users share the node).
mpi() { mpirun --bind-to none -n "$1" "${@:2}"; }
