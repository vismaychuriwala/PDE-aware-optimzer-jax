# main.py
"""
Top‑level runner for PINN benchmarks.
"""
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from dataloader import build_dataset, load_mat_data
from residual import get_residual
from model import get_model_class
from optimiser import get_optim_trainer_factories
# ────────────────────────────────────────────────────────────────────────────────
#  Training loop (optimizer‑agnostic)
# ────────────────────────────────────────────────────────────────────────────────

def train(rng, init_fn, step_fn, dataset, model, epochs=100, batch_size=1024):
    params = model.init(rng, jnp.zeros((1,)), jnp.zeros((1,)))
    state  = init_fn(params)
    (x_f, _), *_ = dataset
    n_batches = x_f.shape[0] // batch_size
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(x_f.shape[0])
        epoch_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            batch = ((dataset[0][0][idx], dataset[0][1][idx]), dataset[1], dataset[2])
            params, state, loss = step_fn(params, state, batch)
            epoch_loss += float(loss)
        if ep == 1 or ep % 200 == 0:
            print(f"Epoch {ep:4d} | Loss {epoch_loss / n_batches:.3e}")
    return params

# ────────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────────

def evaluate(model, params, X, T):
    XT = jnp.stack([X.flatten(), T.flatten()], -1)
    return np.array(model.apply(params, XT[:, 0], XT[:, 1])).reshape(X.shape)


def compare_error_plots(pde: str, model, trained_params: dict):
    x, t, fdm = load_mat_data(pde)
    X, T = np.meshgrid(x, t)
    os.makedirs("figures", exist_ok=True)  # save to figures/ directory

    for name, params in trained_params.items():
        U_pred = evaluate(model, params, X, T)
        err = np.abs(fdm - U_pred)

        fig = plt.figure(figsize=(6, 4))
        plt.imshow(err, extent=[x[0], x[-1], t[0], t[-1]], origin='lower', aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f"|FDM - {name}| error")
        plt.tight_layout()

        filename = f"figures/{pde}_{name}_error.png"
        fig.savefig(filename)
        print(f"[✓] Saved: {filename}")

        # For GUI (VSCode, terminal) — forces window to open if supported
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[!] Could not show figure: {e}")

# ────────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PINN benchmark runner')
    parser.add_argument('--pde',   required=True, choices=['burgers', 'allen_cahn'], help='PDE type')
    parser.add_argument('--model', default='basic', choices=['basic'],              help='Model architecture')
    args = parser.parse_args()

    rng_key = jax.random.PRNGKey(0)

    # Dataset & residual
    dataset      = build_dataset(rng_key, args.pde)
    residual_fn  = get_residual(args.pde)

    # Model instance
    ModelClass = get_model_class(args.model)
    model      = ModelClass()

    # Optimizer trainers
    trainer_factories = get_optim_trainer_factories(model, residual_fn)

    # Train with each optimizer
    trained_params = {}
    for name, (init_fn, step_fn) in trainer_factories.items():
        if init_fn is None or step_fn is None:
            print(f"[!] Skipping {name} (optimizer unavailable)")
            continue

        print(f"\n Training with {name}…")
        params = train(rng_key, init_fn, step_fn, dataset, model)
        trained_params[name] = params

    # Compare predictions
    compare_error_plots(args.pde, model, trained_params)



if __name__ == '__main__':
    main()
