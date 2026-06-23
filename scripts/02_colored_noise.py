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

from src.engine import finite_spike_times, simulate_colored_spike_times, simulate_white_spike_times


def build_parser():
    parser = argparse.ArgumentParser(description="Generate the selected white-vs-colored spike-time density plot.")
    parser.add_argument("--sigma", type=float, default=0.25, help="stationary free-potential sigma_Q")
    parser.add_argument("--n-neurons", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--quick", action="store_true", help="short run for smoke tests")
    return parser


def output_dir():
    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def simulate_spike_times(mu_q, sigma_q, tau_c, args, seed):
    if tau_c == 0.0:
        return simulate_white_spike_times(
            mu_q=mu_q,
            sigma_q=sigma_q,
            dt=args.dt,
            t_max=args.t_max,
            n_neurons=args.n_neurons,
            seed=seed,
        )

    return simulate_colored_spike_times(
        mu_q=mu_q,
        sigma_q=sigma_q,
        tau_c=tau_c,
        dt=args.dt,
        t_max=args.t_max,
        n_neurons=args.n_neurons,
        seed=seed,
    )


def plot_density_comparison(args, figure_dir):
    mu_values = [0.85, 1.0, 1.2]
    regime_labels = ["subthreshold", "near threshold", "suprathreshold"]
    tau_c_values = [0.0, 2.0, 10.0]
    colors = ["black", "tab:blue", "tab:orange"]
    linestyles = ["-", "--", "-."]
    spike_samples = {}

    for mu_index, mu_q in enumerate(mu_values):
        for tau_index, tau_c in enumerate(tau_c_values):
            spike_times = simulate_spike_times(mu_q, args.sigma, tau_c, args, args.seed + 100 * mu_index + tau_index)
            spike_samples[(mu_q, tau_c)] = finite_spike_times(spike_times)

    valid_samples = [sample for sample in spike_samples.values() if sample.size > 0]
    upper_time = min(float(np.percentile(np.concatenate(valid_samples), 98.0)), args.t_max) if valid_samples else args.t_max
    bins = np.linspace(0.0, max(80.0, upper_time), 70)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)
    for ax, mu_q, regime_label in zip(axes, mu_values, regime_labels):
        for tau_c, color, linestyle in zip(tau_c_values, colors, linestyles):
            label = "white" if tau_c == 0.0 else rf"colored $\tau_c={tau_c:g}$ ms"
            ax.hist(
                spike_samples[(mu_q, tau_c)],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                linestyle=linestyle,
                label=label,
            )

        ax.set_title(rf"{regime_label}: $\mu_Q={mu_q:g}$")
        ax.set_xlabel("first-spike time (ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle(rf"Comparable spike-time densities: fixed $\sigma_Q={args.sigma:g}$", y=1.03)
    fig.tight_layout()

    path = figure_dir / "03_white_colored_fpt_density_by_mu.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    if args.quick:
        args.n_neurons = 500
        args.t_max = 800.0
        args.dt = max(args.dt, 0.1)

    path = plot_density_comparison(args, output_dir())
    print(f"Generated figure:\n- {path}")


if __name__ == "__main__":
    main()
