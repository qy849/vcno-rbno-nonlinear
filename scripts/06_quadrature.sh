#!/usr/bin/env bash
# Quadrature-degree study (loss, gradient, solution, cost) for all three benchmarks.
# Needs steps 1 and 2 (test solutions and RBNO predictions).   bash scripts/06_quadrature.sh
source "$(dirname "$0")/common.sh"
run() {  # experiment, ranks, reference degree, degrees...
  local exp="$1" np="$2" ref="$3"; shift 3
  wait_for_cpus "$np"
  log "quadrature study $exp (ref $ref, degrees $*)"
  mpi "$np" python -m rbno.quadrature_study "configs/$exp.yaml" --ref "$ref" --degrees "$@"
}
run semilinear_k1000 16 12 2 3 4 5 6 7 8 10 12
run quasilinear      16 20 4 5 6 7 8 10 12 16
run hyperelastic      8 20 4 5 6 7 8 10 12 16
