#!/usr/bin/env bash
# Reproduce every number and figure of Section 6 from scratch, in dependency order.
#   bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
# Steps can also be run one at a time (see README.md); each skips work already done.
source "$(dirname "$0")/common.sh"
EXPS="semilinear_k0 semilinear_k1000 quasilinear hyperelastic"
for e in $EXPS; do bash scripts/01_data.sh "configs/$e.yaml"; done
python -m rbno.check_kappa0 configs/semilinear_k0.yaml
for e in $EXPS; do bash scripts/02_rbno.sh "configs/$e.yaml"; done
for e in semilinear_k1000 quasilinear hyperelastic; do bash scripts/03_baselines.sh "$e"; done
for e in $EXPS; do bash scripts/08_reduced_opt.sh "configs/$e.yaml"; done
for e in semilinear_k1000 quasilinear hyperelastic; do bash scripts/04_stability.sh "configs/$e.yaml"; done
bash scripts/05_figures_tables.sh
bash scripts/06_quadrature.sh      # optional: quadrature-degree study
# optional: hyperparameter search of the semilinear RBNO (configs/tune/*.txt)
# bash scripts/07_tune.sh configs/semilinear_k1000.yaml configs/tune/semilinear_stage1.txt
log "done"
