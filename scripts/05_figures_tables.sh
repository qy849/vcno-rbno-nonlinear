#!/usr/bin/env bash
# Step 5: all figures of Section 6 / Appendix B and the LaTeX table fragments (docs/tex/).
source "$(dirname "$0")/common.sh"
F=figures
python -m rbno.plots.fields configs/semilinear_k1000.yaml --row 5000 --out $F/benchmarks/semilinear --fmt-p %.1f --fmt-u %.2f --fmt-sigma %.1f
python -m rbno.plots.fields configs/quasilinear.yaml --row 8000 --out $F/benchmarks/quasilinear --fmt-p %.1f --fmt-u %.2f --fmt-sigma %.1f
python -m rbno.plots.fields configs/hyperelastic.yaml --row 5000 --out $F/benchmarks/hyperelastic --fmt-p %.1f --fmt-u %.2f --fmt-sigma %.2f
python -m rbno.plots.ref_pred_diff configs/semilinear_k1000.yaml --row 5000 --out $F/ref_pred_diff/semilinear_k1000 --fmt-u %.2f --fmt-sigma %.1f
python -m rbno.plots.ref_pred_diff configs/quasilinear.yaml --row 8000 --out $F/ref_pred_diff/quasilinear --fmt-u %.2f --fmt-sigma %.1f
python -m rbno.plots.ref_pred_diff configs/hyperelastic.yaml --row 5000 --out $F/ref_pred_diff/hyperelastic --fmt-u %.1f --fmt-sigma %.2f
python -m rbno.plots.ratios semilinear_k0 semilinear_k1000 --labels '$\kappa=0$' '$\kappa=10^3$' --out $F/ratios/semilinear
python -m rbno.plots.ratios quasilinear --out $F/ratios/quasilinear
python -m rbno.plots.ratios hyperelastic --out $F/ratios/hyperelastic
python -m rbno.plots.pod semilinear_k1000 --out $F/pod/semilinear_k1000.png
python -m rbno.plots.pod semilinear_k0 --out $F/pod/semilinear_k0.png
python -m rbno.plots.pod quasilinear --out $F/pod/quasilinear.png
python -m rbno.plots.pod hyperelastic --out $F/pod/hyperelastic.png
for e in semilinear_k0 semilinear_k1000 quasilinear hyperelastic; do
  python -m rbno.plots.loss_history $e --out $F/training/$e.png
done
for e in semilinear_k1000 quasilinear hyperelastic; do
  python -m rbno.plots.baseline_history $e --out $F/training/baselines/$e
done
for e in semilinear_k1000 quasilinear hyperelastic; do
  python -m rbno.plots.stability $e --out $F/stability/$e.png
done
python -m rbno.plots.stability semilinear_k1000 quasilinear hyperelastic --row --out $F/stability/row.png
(cd baselines
 python -m residual_maps semilinear_k1000 --row 5000 --out ../$F/residual/semilinear_k1000
 python -m residual_maps quasilinear --row 8000 --out ../$F/residual/quasilinear
 python -m residual_maps hyperelastic --row 5000 --out ../$F/residual/hyperelastic)
python -m rbno.report
python -m rbno.plots.export
