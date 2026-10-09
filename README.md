# P-SLQ: experiments

Code for the experiments of *Stochastic Lanczos Quadrature for Computational
Uncertainty in Linear Algebra* (Fifen, Karambal, Hennig).

| script | reproduces | hardware | runtime |
|---|---|---|---|
| `exp_consistency.py` | P-SLQ bias and variance | CPU | ~5 min |
| `exp_reprojection_matched.py` | reprojection at matched targets | CPU | ~10 min |
| `exp_trace.py` | trace estimation at matched budgets | CPU | ~20 min |
| `calibration_experiment.py` | calibration of the belief: figure and table | CPU | ~30 min |
| `fashionmnist_subspace_kl.py` | Laplace approximation, FashionMNIST | GPU | ~40 min |
| `cifar10_cnn_kl.py` | Laplace approximation, CIFAR-10 | GPU (40 GB) | ~2.5 h |

The three CPU scripts need only NumPy, SciPy and Matplotlib.
`calibration_experiment.py` is self-contained: Lanczos, P-SLQ (`pslq_belief`,
Algorithm 1 of the paper), the reference beliefs and the calibration statistic
of Cockayne et al. (2019) are in one file. `exp_consistency.py` and
`exp_reprojection_matched.py` use the helpers in `pslq_core.py`.

## Usage

```bash
pip install -r requirements.txt               # CPU experiments
python exp_consistency.py                     # --plot redraws from saved results
python exp_reprojection_matched.py
python exp_trace.py                           # --plot redraws from exp_trace.csv
python calibration_experiment.py              # --plot redraws from results.pkl
pip install -r requirements-laplace.txt       # Laplace experiments (GPU)
```
