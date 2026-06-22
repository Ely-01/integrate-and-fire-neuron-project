import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (
    cv_from_fpt,
    firing_rate_from_fpt,
    simulate_colored_first_passage_times,
    simulate_first_passage_times,
    valid_first_passage_times,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Compare white and colored-noise LIF regimes.")
    parser.add_argument("--mu", type=float, default=0.95, help="free-potential equilibrium mu_Q")
    parser.add_argument("--sigma", type=float, default=0.25, help="stationary free-potential sigma_Q")
    parser.add_argument("--n-neurons", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--quick", action="store_true", help="short run for smoke tests")
    return parser


def output_dir():
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_condition(mu, sigma, tau_c, args, seed):
    if tau_c == 0.0:
        fpt = simulate_first_passage_times(
            mu=mu,
            sigma=sigma,
            dt=args.dt,
            t_max=args.t_max,
            n_neurons=args.n_neurons,
            seed=seed,
        )
    else:
        fpt = simulate_colored_first_passage_times(
            mu=mu,
            sigma=sigma,
            tau_c=tau_c,
            dt=args.dt,
            t_max=args.t_max,
            n_neurons=args.n_neurons,
            seed=seed,
        )

    valid = valid_first_passage_times(fpt)
    spike_fraction = valid.size / args.n_neurons
    return fpt, firing_rate_from_fpt(fpt), cv_from_fpt(fpt), spike_fraction


def plot_fixed_mu_regime(args, tau_c_values, out):
    rates = []
    cvs = []
    fractions = []

    for idx, tau_c in enumerate(tau_c_values):
        fpt, rate, cv, spike_fraction = run_condition(
            args.mu, args.sigma, tau_c, args, args.seed + idx
        )
        rates.append(rate)
        cvs.append(cv)
        fractions.append(spike_fraction)

    labels = ["white" if tau_c == 0.0 else f"{tau_c:g}" for tau_c in tau_c_values]
    x = np.arange(len(tau_c_values))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))

    axes[0].plot(x, rates, marker="o", color="tab:blue", linewidth=2.0)
    axes[0].set_ylabel("firing rate (Hz)")

    axes[1].plot(x, cvs, marker="o", color="tab:red", linewidth=2.0)
    axes[1].set_ylabel("ISI coefficient of variation")

    axes[2].plot(x, fractions, marker="o", color="tab:green", linewidth=2.0)
    axes[2].set_ylabel("spiking fraction before t_max")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"noise correlation time $\tau_c$ (ms)")
        ax.grid(alpha=0.25)

    fig.suptitle(
        rf"White-to-colored noise regime, $\mu_Q={args.mu:g}$, $\sigma_Q={args.sigma:g}$",
        y=1.02,
    )
    fig.tight_layout()

    path = out / "04_colored_noise_regime.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return path, rates, cvs, fractions


def plot_density_comparison_by_mu(args, tau_c_values, out):
    mu_values = [0.85, 1.0, 1.2]
    mu_labels = ["subthreshold", "near threshold", "suprathreshold"]

    if args.quick:
        compared_tau_c = [tau_c_values[0], tau_c_values[-1]]
    else:
        compared_tau_c = [tau_c_values[0], tau_c_values[len(tau_c_values) // 2], tau_c_values[-1]]

    colors = ["black", "tab:blue", "tab:orange"]
    linestyles = ["-", "--", "-."]
    samples = {}

    density_args = argparse.Namespace(**vars(args))
    density_args.n_neurons = max(args.n_neurons, 3000)

    for mu_idx, mu in enumerate(mu_values):
        for tau_idx, tau_c in enumerate(compared_tau_c):
            fpt, _, _, _ = run_condition(
                mu,
                args.sigma,
                tau_c,
                density_args,
                args.seed + 2000 + 100 * mu_idx + tau_idx,
            )
            samples[(mu, tau_c)] = valid_first_passage_times(fpt)

    all_valid = [data for data in samples.values() if data.size > 0]
    if all_valid:
        upper = np.percentile(np.concatenate(all_valid), 98.0)
        upper = max(upper, 50.0)
    else:
        upper = args.t_max
    bins = np.linspace(0.0, min(upper, args.t_max), 70)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)

    for ax, mu, mu_label in zip(axes, mu_values, mu_labels):
        for tau_c, color, linestyle in zip(compared_tau_c, colors, linestyles):
            data = samples[(mu, tau_c)]
            label = "white" if tau_c == 0.0 else rf"colored $\tau_c={tau_c:g}$ ms"
            ax.hist(
                data,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                linestyle=linestyle,
                label=label,
            )

        ax.set_title(rf"{mu_label}: $\mu_Q={mu:g}$")
        ax.set_xlabel("first-passage time (ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle(
        rf"Comparable FPT densities: same $\mu_Q$ values, fixed $\sigma_Q={args.sigma:g}$",
        y=1.03,
    )
    fig.tight_layout()

    path = out / "05_white_colored_fpt_density_by_mu.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transfer_comparison(args, out):
    if args.quick:
        mu_values = np.linspace(0.75, 1.35, 7)
        tau_c_values = [0.0, 2.0, 10.0]
        n_neurons = max(300, args.n_neurons)
    else:
        mu_values = np.linspace(0.7, 1.45, 10)
        tau_c_values = [0.0, 1.0, 5.0, 10.0]
        n_neurons = args.n_neurons

    colors = ["black", "tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)

    for tau_idx, tau_c in enumerate(tau_c_values):
        rates = []
        cvs = []

        for mu_idx, mu in enumerate(mu_values):
            local_args = argparse.Namespace(**vars(args))
            local_args.n_neurons = n_neurons
            fpt, rate, cv, _ = run_condition(
                mu,
                args.sigma,
                tau_c,
                local_args,
                args.seed + 1000 + 100 * tau_idx + mu_idx,
            )
            rates.append(rate)
            cvs.append(cv)

        label = "white" if tau_c == 0.0 else rf"colored $\tau_c={tau_c:g}$ ms"
        axes[0].plot(
            mu_values,
            rates,
            marker=markers[tau_idx],
            color=colors[tau_idx],
            linewidth=2.0,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            mu_values,
            cvs,
            marker=markers[tau_idx],
            color=colors[tau_idx],
            linewidth=2.0,
            markersize=5,
            label=label,
        )

    for ax in axes:
        ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.1)
        ax.set_xlabel(r"free-potential equilibrium $\mu_Q$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("transfer curve")
    axes[1].set_ylabel("ISI coefficient of variation")
    axes[1].set_title("irregularity curve")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle(rf"White vs colored noise at fixed $\sigma_Q={args.sigma:g}$", y=1.02)
    fig.tight_layout()

    path = out / "06_white_colored_transfer_cv.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    tau_c_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    if args.quick:
        args.n_neurons = 500
        args.t_max = 800.0
        args.dt = 0.1
        tau_c_values = [0.0, 1.0, 5.0]

    out = output_dir()
    path, rates, cvs, fractions = plot_fixed_mu_regime(args, tau_c_values, out)
    density_path = plot_density_comparison_by_mu(args, tau_c_values, out)
    transfer_path = plot_transfer_comparison(args, out)

    print("tau_c_ms,rate_hz,cv,spike_fraction")
    for tau_c, rate, cv, fraction in zip(tau_c_values, rates, cvs, fractions):
        print(f"{tau_c:g},{rate:.4f},{cv:.4f},{fraction:.4f}")
    print(f"Generated figures:\n- {path}\n- {density_path}\n- {transfer_path}")


if __name__ == "__main__":
    main()
