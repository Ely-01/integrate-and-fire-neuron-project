import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import firing_rate_from_fpt, simulate_first_passage_times, valid_first_passage_times
from src.theory import gain_function_F, siegert_firing_rate


def build_parser():
    parser = argparse.ArgumentParser(description="Reproduce core LIF results from Burkitt's review.")
    parser.add_argument("--n-neurons", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def output_dir():
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_fi_curve(args, out):
    mu_grid = np.linspace(0.55, 1.6, 260)
    mu_points = np.linspace(0.7, 1.5, 9)
    sigma_values = [0.0, 0.1, 0.25]
    colors = ["black", "tab:blue", "tab:red"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for sigma, color in zip(sigma_values, colors):
        theory_rates = [siegert_firing_rate(mu=mu, sigma=sigma) for mu in mu_grid]
        ax.plot(mu_grid, theory_rates, color=color, linewidth=2.0, label=f"Siegert sigma={sigma:g}")

        if sigma > 0.0:
            simulated = []
            for idx, mu in enumerate(mu_points):
                fpt = simulate_first_passage_times(
                    mu=mu,
                    sigma=sigma,
                    dt=args.dt,
                    t_max=args.t_max,
                    n_neurons=args.n_neurons,
                    seed=args.seed + idx,
                )
                simulated.append(firing_rate_from_fpt(fpt))
            ax.scatter(mu_points, simulated, s=34, color=color, edgecolor="white", zorder=3)

    ax.axvline(1.0, color="0.45", linestyle="--", linewidth=1.2, label="deterministic threshold")
    ax.set_xlabel(r"free-potential equilibrium $\mu_Q$")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("LIF transfer function: simulation vs Siegert formula")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    path = out / "01_fi_curve_siegert.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_fpt_density(args, out):
    sigma = 0.2
    mu_values = [0.85, 1.0, 1.2]
    labels = ["subthreshold", "near threshold", "suprathreshold"]
    colors = ["tab:orange", "tab:green", "tab:purple"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for idx, (mu, label, color) in enumerate(zip(mu_values, labels, colors)):
        fpt = simulate_first_passage_times(
            mu=mu,
            sigma=sigma,
            dt=args.dt,
            t_max=args.t_max,
            n_neurons=max(args.n_neurons, 4000),
            seed=args.seed + 100 + idx,
        )
        valid = valid_first_passage_times(fpt)
        ax.hist(valid, bins=70, density=True, alpha=0.45, color=color, label=f"{label}, mu={mu:g}")

    ax.set_xlabel("first-passage time (ms)")
    ax.set_ylabel("density")
    ax.set_title(r"ISI/FPT density changes across threshold ($\sigma_Q=0.2$)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = out / "02_fpt_density.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gain_function(out):
    x = np.linspace(-3.0, 3.0, 320)
    ratios = [0.05, 0.1, 0.2]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True)

    for ratio, color in zip(ratios, colors):
        f_values = gain_function_F(x, ratio)
        derivative = np.gradient(f_values, x)
        label = fr"$\sigma_Q/\theta={ratio:g}$"
        axes[0].plot(x, f_values, color=color, linewidth=2.2, label=label)
        axes[1].plot(x, derivative, color=color, linewidth=2.2, label=label)

    for ax in axes:
        ax.axvline(-1.5, color="0.6", linestyle=":", linewidth=1.2)
        ax.axvline(-0.5, color="0.6", linestyle=":", linewidth=1.2)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$x=(\mu_Q - V_{th})/(\sqrt{2}\sigma_Q)$")

    axes[0].set_ylabel("F")
    axes[1].set_ylabel("dF/dx")
    axes[0].set_title("Burkitt Eq. (61)")
    axes[1].set_title("gain transition")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()

    path = out / "03_gain_modulation_fig4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    out = output_dir()

    paths = [
        plot_fi_curve(args, out),
        plot_fpt_density(args, out),
        plot_gain_function(out),
    ]

    print("Generated figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
