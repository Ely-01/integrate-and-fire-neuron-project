import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for plotting. Run: pip install -r requirements.txt") from exc
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import firing_rate_hz, simulate_white_spike_times
from src.theory import gain_function, siegert_firing_rate_hz


def build_parser():
    parser = argparse.ArgumentParser(description="Generate the selected white-noise LIF paper figures.")
    parser.add_argument("--quick", action="store_true", help="Use fewer Monte Carlo samples.")
    parser.add_argument("--n-neurons", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def configure(args):
    if args.quick:
        args.n_neurons = min(args.n_neurons, 500)
        args.t_max = min(args.t_max, 800.0)
        args.dt = max(args.dt, 0.1)

    return {
        "sigma_values": [0.0, 0.1, 0.25],
        "mu_curve": np.linspace(0.55, 1.6, 260),
        "mu_simulation_points": np.linspace(0.7, 1.5, 9),
    }


def output_dir():
    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def plot_fi_curve(args, config, figure_dir):
    colors = ["black", "tab:blue", "tab:red"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for sigma_index, (sigma_q, color) in enumerate(zip(config["sigma_values"], colors)):
        theory_rates = [siegert_firing_rate_hz(mu_q, sigma_q) for mu_q in config["mu_curve"]]
        ax.plot(
            config["mu_curve"],
            theory_rates,
            color=color,
            linewidth=2.0,
            label=fr"Siegert $\sigma_Q={sigma_q:g}$",
        )

        if sigma_q == 0.0:
            continue

        simulation_rates = []
        for mu_index, mu_q in enumerate(config["mu_simulation_points"]):
            spike_times = simulate_white_spike_times(
                mu_q=mu_q,
                sigma_q=sigma_q,
                dt=args.dt,
                t_max=args.t_max,
                n_neurons=args.n_neurons,
                seed=args.seed + 100 * sigma_index + mu_index,
            )
            simulation_rates.append(firing_rate_hz(spike_times))

        ax.scatter(
            config["mu_simulation_points"],
            simulation_rates,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=fr"Monte Carlo $\sigma_Q={sigma_q:g}$",
        )

    ax.axvline(1.0, color="0.45", linestyle="--", linewidth=1.2, label="deterministic threshold")
    ax.set_xlabel(r"free-potential equilibrium $\mu_Q$")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("White-noise LIF transfer curve: Monte Carlo vs Siegert")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    path = figure_dir / "01_fi_curve_siegert.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gain_function(figure_dir):
    x_values = np.linspace(-3.0, 3.0, 320)
    sigma_ratios = [0.05, 0.1, 0.2]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True)
    for sigma_ratio, color in zip(sigma_ratios, colors):
        gain_values = gain_function(x_values, sigma_ratio)
        axes[0].plot(x_values, gain_values, color=color, linewidth=2.2, label=fr"$\sigma_Q/\theta={sigma_ratio:g}$")
        axes[1].plot(x_values, np.gradient(gain_values, x_values), color=color, linewidth=2.2)

    for ax in axes:
        ax.axvline(-1.5, color="0.6", linestyle=":", linewidth=1.2)
        ax.axvline(-0.5, color="0.6", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$x=(\mu_Q - V_{th})/(\sqrt{2}\sigma_Q)$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("F")
    axes[1].set_ylabel("dF/dx")
    axes[0].set_title("Burkitt Eq. (61)")
    axes[1].set_title("gain transition")
    axes[0].legend(frameon=False)
    fig.tight_layout()

    path = figure_dir / "02_gain_modulation_fig4.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    config = configure(args)
    figure_dir = output_dir()

    figure_paths = [
        plot_fi_curve(args, config, figure_dir),
        plot_gain_function(figure_dir),
    ]

    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
