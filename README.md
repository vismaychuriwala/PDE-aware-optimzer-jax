# PDE-aware-optimizer-jax

Physics-Informed Neural Network (PINN) training framework for benchmarking optimizers on canonical PDEs.

This repository supports training PINNs on:
- Burgers' equation
- Allen–Cahn equation
- Korteweg–de Vries (KdV) equation

Each problem is defined through its PDE residual and solved using multiple optimizers including Adam and SOAP variants.

## Usage

Ensure `.mat` files (e.g., `burgers.mat`) are placed in `Optimiser_PINN/data/`.

Run training with:

```bash
python main.py --pde burgers --model basic
python main.py --pde allen_cahn --model basic
python main.py --pde kdv --model basic
```

After training, error heatmaps are saved in the `figures/` directory:

```
figures/<pde>_<optimizer>_error.png
```

## PDE Residuals

Each PINN minimizes a residual of the form:

**Burgers:**  
  ∂ₜ u + u ∂ₓ u − ν ∂ₓₓ u

**Allen–Cahn:**  
  ∂ₜ u − ε ∂ₓₓ u − f(u)

**KdV (Korteweg–de Vries):**  
  ∂ₜ u + u ∂ₓ u + μ ∂ₓₓₓ u
