# vcno-rbno-nonlinear

Code for the paper
*Variationally correct operator learning for nonlinear parametric partial differential equations*.

This work extends variationally correct operator learning to nonlinear partial differential equations written as first-order systems. We show that, near a solution branch, the first-order system least-squares (FOSLS) residual is equivalent to the error in the graph norm of the system whenever the linearized residual is stable and its derivative is Lipschitz. Both conditions are verified for a monotone reaction (semilinear diffusion–reaction), for a diffusion coefficient that depends on the solution (quasilinear diffusion), and for compressible Neo-Hookean elasticity (hyperelasticity). A discrete version relates the loss of a finite element state to its distance from the finite element solution. A Reduced Basis Neural Operator (RBNO), with values in a conforming finite element space, is trained on this loss without nonlinear solves: the finite element library assembles the loss and its symbolically derived gradient, which the network receives through the chain rule.

This repository builds on the linear code of [vcno-rbno](https://github.com/qy849/vcno-rbno).

## Repository layout

- `configs/`: YAML configuration files of the experiments (mesh, parameter distribution, data splits, solver, POD, network and training settings, stability test); `configs/baselines/` for the baselines and `configs/tune/` for the hyperparameter search.
- `src/rbno/problems/`: FOSLS problems: function spaces, lifts, residual, linearization, Gauss–Newton solver, loss and its gradient, norms.
- `src/rbno/`: pipeline modules: parameter sampling, finite element solves, POD basis, training, prediction, evaluation, timing, reduced minimizers, stability test, report.
- `src/rbno/plots/`: figures of the paper.
- `src/hippylibx/`: bundled copy of hIPPYlib-X (Gaussian random fields, randomized eigensolver), see `src/hippylibx/README.md`.
- `baselines/`: FNO, PCA-Net, FC-PINO (`fc_pino/`), RBNO trained on data losses, and the finite element solution sampled at grid nodes.
- `scripts/`: shell entry points of the pipeline (`01`–`08`, `run_all.sh`) and the data download.
- `environment/`: conda environment file.
- `data/`, `results/`, `checkpoints/`, `analysis/`, `figures/`, `docs/`, `logs/`: generated (not committed to Git).

## Environment setup

Create the environment:

```bash
cd /path/to/vcno-rbno-nonlinear
conda env create -f environment/environment.yml
conda activate rbno
```

Nothing has to be installed from this repository: `scripts/common.sh` activates the environment (`CONDA_ENV`, default `rbno`), puts `src/` and `baselines/` on `PYTHONPATH`, and sets `OMP_NUM_THREADS=1` to avoid CPU oversubscription in MPI workflows. When running the Python modules directly, set these yourself:

```bash
export PYTHONPATH=$PWD/src:$PWD/baselines OMP_NUM_THREADS=1
```

Packages and their versions used in the paper runs:
- Python `3.11`
- DOLFINx `0.10.0`
- PyTorch `>= 2.9`
- neuraloperator `2.0.0`

## Problem setups

| Experiment | Problem | Parameter | Train / val / test | $N_{\mathrm{POD}}$ / $r$ |
|---|---|---|---|---|
| `semilinear_k0` | heat conduction ($\kappa = 0$) | conductivity on 16 squares | 3000 / 500 / 500 | 500 / 128 |
| `semilinear_k1000` | $-\nabla\cdot(p\nabla u) + 10^3u^3 = f$ | same samples | 3000 / 500 / 500 | 500 / 128 |
| `quasilinear` | $-\nabla\cdot(p(1+u^2)\nabla u) = f$ | $p = 0.01 + e^m$ | 8000 / 800 / 500 | 1000 / 512 |
| `hyperelastic` | compressible Neo-Hookean, traction | $p = 1 + e^m$ | 5000 / 500 / 500 | 1000 / 256 |

All experiments use $RT_1 \times CG_2$ elements on a structured triangulation ($128\times128$ cells of the unit square, $128\times64$ cells of $(0,2)\times(0,1)$ for hyperelasticity) and Gauss–Newton for the discrete normal equations. `semilinear_k0` is the heat conduction problem of [vcno-rbno](https://github.com/qy849/vcno-rbno) (`poisson_setup1`) with the same parameter samples, and is used as a regression check of the nonlinear code.

## Data

Create the parameter samples, finite element solutions and POD bases of an experiment:

```bash
bash scripts/01_data.sh configs/quasilinear.yaml
```

The parameter samples, the POD bases and the trained RBNO checkpoints of the paper are available on Hugging Face. Download them into `data/`, `checkpoints/` and `results/` (optionally for a subset of `semilinear quasilinear hyperelastic`):

```bash
bash scripts/download_data.sh https://huggingface.co/datasets/<user>/<dataset>/resolve/main
```

Each step skips work that is already done, so with the downloaded files `01_data.sh` only computes the finite element solutions, and `02_rbno.sh` evaluates the downloaded checkpoint instead of training a new one (delete `checkpoints/<exp>/rbno/best_val.pt` to retrain). The POD bases must be downloaded together with the checkpoints: the networks predict coefficients in these bases.

## Reproducing the results

```bash
bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
```

runs all steps below in order. The scripts launch `NPROCS` MPI ranks (default 60) for the CPU stages and train on a vacant GPU among `GPUS` (default `"0 1"`); set both to match your machine, e.g. `NPROCS=16 GPUS=0 bash scripts/run_all.sh`.

| Step | Command | Output |
|---|---|---|
| 1. Parameters, finite element solutions, POD | `bash scripts/01_data.sh configs/<exp>.yaml` | `data/` |
| 2. RBNO: train, predict, evaluate, time | `bash scripts/02_rbno.sh configs/<exp>.yaml` | `checkpoints/<exp>/rbno`, `results/<exp>/rbno` |
| 3. Baselines | `bash scripts/03_baselines.sh <exp>` | `results/<exp>/<method>` |
| 4. Reduced minimizers | `bash scripts/08_reduced_opt.sh configs/<exp>.yaml` | `results/<exp>/reduced_opt` |
| 5. Stability test | `bash scripts/04_stability.sh configs/<exp>.yaml` | `analysis/<exp>/stability` |
| 6. Figures and tables | `bash scripts/05_figures_tables.sh` | `figures/`, `docs/` |
| 7. Quadrature study (optional) | `bash scripts/06_quadrature.sh` | `analysis/<exp>/quadrature` |
| 8. Hyperparameter search (optional) | `bash scripts/07_tune.sh configs/<exp>.yaml configs/tune/<grid>.txt` | `results/<exp>/tune` |

Steps 3–5 need steps 1 and 2 of the same experiment, and step 6 needs all of them. Baselines and the stability test are run for `semilinear_k1000`, `quasilinear` and `hyperelastic`.

The individual modules take the experiment configuration and can also be run directly, e.g.

```bash
mpirun -n 60 python -m rbno.solve configs/quasilinear.yaml --store test
mpirun -n 61 python -m rbno.train configs/quasilinear.yaml --device cuda:0   # 1 trainer + 60 residual workers
python -m rbno.predict configs/quasilinear.yaml --device cuda:0
mpirun -n 60 python -m rbno.evaluate configs/quasilinear.yaml --method rbno
```

## Model training and evaluation

The RBNO predicts the coefficients of a POD basis of the conforming finite element space and is trained on the FOSLS loss only. The loss and its gradient with respect to the finite element state are assembled by DOLFINx on CPU workers (`rbno.train`), and the gradient is passed to the network through the chain rule. The checkpoint with the smallest validation loss is kept.

Models covered:
- Reduced Basis Neural Operator (RBNO), FOSLS loss: `src/rbno/train.py`
- RBNO, data loss in the $L_2$ or $H(\mathrm{div})\times H^1$ norm: `baselines/rbno_data_loss.py`
- Fourier Neural Operator (FNO): `baselines/fno.py`
- Principal Component Analysis Network (PCA-Net): `baselines/pca_net.py`
- Fourier-Continuation Physics-Informed Neural Operator (FC-PINO): `baselines/fc_pino/`

All predictions are evaluated in the same way (`rbno.evaluate`): relative errors in the graph norm, the FOSLS loss, and the ratio of the error to the residual. Grid-based predictions (FNO, PCA-Net, FC-PINO) are first mapped to the finite element space; `baselines/fe_nodal.py` applies the same map to the finite element solution as a control.

### FC-PINO comparison

As in [vcno-rbno](https://github.com/qy849/vcno-rbno), we compare the RBNO with the Fourier-Continuation Physics-Informed Neural Operator (FC-PINO) [**Ganeshram, A., Maust, H., Duruisseaux, V., Li, Z., Wang, Y., Leibovici, D., Bruno, O., Hou, T., & Anandkumar, A.** (2025), *FC-PINO: High Precision Physics-Informed Neural Operators via Fourier Continuation*, arXiv preprint [arXiv:2211.15960](https://doi.org/10.48550/arXiv.2211.15960)], built on the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library. The FC-FNO architecture is in `baselines/fc_pino/fc_fno.py` and the physics-informed residual of the three problems in `baselines/fc_pino/residual.py`.

## Analysis

- Discrete stability test (`rbno.stability`): smallest generalized eigenvalue of the linearized residual at the finite element solutions on three meshes, at the Gauss–Newton iterates and at the RBNO predictions.
- Reduced minimizers (`rbno.reduced_opt`): minimizers of the loss in the reduced space, which separate the network error from the error of the reduced basis.
- Quadrature study (`rbno.quadrature_study`): effect of the quadrature degree on the loss, the gradient, the solution and the cost.
- Report (`rbno.report`): collects all results into `docs/numbers.json` and LaTeX tables in `docs/tex/`.

## License

This repository is distributed under the GNU General Public License version 2 (`LICENSE`), as it includes a copy of hIPPYlib-X, which is distributed under the same license.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{qiu2026nonlinear,
  title={Variationally correct operator learning for nonlinear parametric partial differential equations},
  author={Qiu, Yuan and Dahmen, Wolfgang and Chen, Peng},
  year={2026}
}
```

and the linear counterpart:

```bibtex
@article{qiu2026variationally,
  title={Variationally correct operator learning: Reduced basis neural operator with a posteriori error estimation},
  author={Qiu, Yuan and Dahmen, Wolfgang and Chen, Peng},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={461},
  pages={119228},
  year={2026},
  publisher={Elsevier}
}
```
