#!/usr/bin/env bash
# Step 3: baselines of one experiment (needs steps 1 and 2 of it): FNO, PCA-Net, FC-PINO, the
# two RBNO data-loss ablations and the FE-solution-at-grid-nodes control, then the common FE
# evaluation and the native grid errors.
#   bash scripts/03_baselines.sh quasilinear
# Environment: PHASE=train|eval|all (default all); METHODS="fno pca_net fc_pino rbno_l2_data
# rbno_hdiv_h1_data" to train a subset; FORCE=1 to retrain methods that already have results.
source "$(dirname "$0")/common.sh"
EXP="$1"
CFG="configs/$EXP.yaml"
PHASE="${PHASE:-all}"
METHODS="${METHODS:-fno pca_net fc_pino rbno_l2_data rbno_hdiv_h1_data}"

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  cd "$REPO_ROOT/baselines"
  for method in $METHODS; do
    case "$method" in
      rbno_l2_data) cmd="rbno_data_loss --norm l2" ;;
      rbno_hdiv_h1_data) cmd="rbno_data_loss --norm hdiv_h1" ;;
      *) cmd="$method" ;;
    esac
    if [[ -f "$REPO_ROOT/results/$EXP/$method/pred_indices.npy" && -z "${FORCE:-}" ]]; then
      log "baseline $method ($EXP) done; skipping (FORCE=1 to retrain)"; continue
    fi
    wait_for_cpus 1
    GPU="$(pick_gpu)"
    log "baseline $method ($EXP) on GPU $GPU"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" python -m $cmd "$EXP" --device cuda:0 2>&1 | tee "$REPO_ROOT/logs/${EXP}_${method}.log"
  done
  [[ -f "$REPO_ROOT/results/$EXP/fe_nodal/pred_indices.npy" ]] || python -m fe_nodal "$EXP"
  cd "$REPO_ROOT"
fi

if [[ "$PHASE" == eval || "$PHASE" == all ]]; then
  for m in fno pca_net fc_pino rbno_l2_data rbno_hdiv_h1_data fe_nodal; do
    [[ -f "results/$EXP/$m/pred_indices.npy" ]] || { log "no predictions for $m ($EXP); skipped"; continue; }
    wait_for_cpus "$NPROCS"
    log "evaluate $m ($EXP)"
    mpi "$NPROCS" python -m rbno.evaluate "$CFG" --method "$m"
  done
  cd "$REPO_ROOT/baselines"
  for m in fno pca_net fc_pino; do
    [[ -f "$REPO_ROOT/results/$EXP/$m/pred_grid.npy" ]] && python -m native_grid "$EXP" "$m"
  done
fi
