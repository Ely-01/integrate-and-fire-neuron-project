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

from src.engine import coefficient_of_variation, finite_spike_times, firing_rate_hz, simulate_colored_spike_times
from src.theory import siegert_firing_rate_hz


def build_parser():
    parser = argparse.ArgumentParser(description="Generate selected colored-noise comparison figures.")
    parser.add_argument("--preset", choices=["quick", "standard", "publication"], default="standard")
    parser.add_argument("--sigma", type=float, default=0.25, help="white-noise free-potential sigma_Q")
    parser.add_argument("--seed", type=int, default=31415)
    return parser


def configure(args):
    if args.preset == "quick":
        return {
            "mu_values": np.array([0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35]),
            "tau_c_values": np.array([0.0, 1.0, 5.0, 10.0]),
            "dt": 0.1,
            "t_max": 1200.0,
            "n_neurons": 2500,
            "free_dt": 0.05,
            "free_n_steps": 120_000,
            "free_burn_in_steps": 15_000,
        }

    if args.preset == "publication":
        return {
            "mu_values": np.array([0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5]),
            "tau_c_values": np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
            "dt": 0.05,
            "t_max": 2000.0,
            "n_neurons": 8000,
            "free_dt": 0.025,
            "free_n_steps": 260_000,
            "free_burn_in_steps": 30_000,
        }

    return {
        "mu_values": np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5]),
        "tau_c_values": np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0]),
        "dt": 0.075,
        "t_max": 1600.0,
        "n_neurons": 3500,
        "free_dt": 0.05,
        "free_n_steps": 150_000,
        "free_burn_in_steps": 20_000,
    }


def output_dir():
    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def normal_pdf(x, mean, standard_deviation):
    return np.exp(-0.5 * ((x - mean) / standard_deviation) ** 2) / (
        standard_deviation * np.sqrt(2.0 * np.pi)
    )


def colored_free_variance(sigma_q, tau_c, tau_q=20.0):
    if tau_c <= 0.0:
        return sigma_q * sigma_q
    return sigma_q * sigma_q * tau_q / (tau_q + tau_c)


def variance_matched_sigma(sigma_q, tau_c, tau_q=20.0):
    if tau_c <= 0.0:
        return sigma_q
    return sigma_q * np.sqrt((tau_q + tau_c) / tau_q)


def simulate_free_voltage(mu_q, sigma_q, tau_c, dt, step_count, rng, tau_q=20.0):
    voltage = np.empty(int(step_count), dtype=float)
    voltage[0] = mu_q

    if tau_c <= 0.0:
        decay = np.exp(-dt / tau_q)
        step_std = sigma_q * np.sqrt(1.0 - decay * decay)
        for index in range(1, int(step_count)):
            voltage[index] = mu_q + decay * (voltage[index - 1] - mu_q) + step_std * rng.standard_normal()
        return voltage

    colored_noise = np.sqrt(1.0 / (2.0 * tau_c)) * rng.standard_normal()
    noise_decay = np.exp(-dt / tau_c)
    noise_step_std = np.sqrt((1.0 - noise_decay * noise_decay) / (2.0 * tau_c))
    voltage_noise_scale = sigma_q * np.sqrt(2.0 / tau_q)

    for index in range(1, int(step_count)):
        colored_noise = noise_decay * colored_noise + noise_step_std * rng.standard_normal()
        drift = (mu_q - voltage[index - 1]) / tau_q
        voltage[index] = voltage[index - 1] + (drift + voltage_noise_scale * colored_noise) * dt

    return voltage


def summarize_spike_times(spike_times, n_neurons):
    valid_times = finite_spike_times(spike_times)
    summary = {
        "rate_hz": firing_rate_hz(spike_times),
        "cv": coefficient_of_variation(spike_times),
        "spike_fraction": float(valid_times.size / n_neurons),
        "median_ms": np.nan,
    }
    if valid_times.size > 0:
        summary["median_ms"] = float(np.median(valid_times))
    return summary


def run_spike_grid(config, sigma_q, seed, match_free_variance=False):
    rows = []
    spike_samples = {}

    for tau_index, tau_c in enumerate(config["tau_c_values"]):
        input_sigma = variance_matched_sigma(sigma_q, tau_c) if match_free_variance else sigma_q
        for mu_index, mu_q in enumerate(config["mu_values"]):
            spike_times = simulate_colored_spike_times(
                mu_q=mu_q,
                sigma_q=input_sigma,
                tau_c=tau_c,
                dt=config["dt"],
                t_max=config["t_max"],
                n_neurons=config["n_neurons"],
                seed=seed + 1000 * tau_index + mu_index,
            )
            row = summarize_spike_times(spike_times, config["n_neurons"])
            row.update(
                {
                    "mu_q": float(mu_q),
                    "tau_c_ms": float(tau_c),
                    "sigma_input_q": float(input_sigma),
                }
            )
            rows.append(row)
            if not match_free_variance:
                spike_samples[(float(mu_q), float(tau_c))] = finite_spike_times(spike_times)

    return rows, spike_samples


def tau_label(tau_c):
    return "white" if tau_c == 0.0 else rf"$\tau_c={tau_c:g}$ ms"


def selected_existing(values, targets):
    selected = [float(values[np.argmin(np.abs(values - target))]) for target in targets]
    return list(dict.fromkeys(selected))


def plot_free_distribution_comparison(config, sigma_q, seed, figure_dir):
    mu_q = 0.95
    tau_values = selected_existing(config["tau_c_values"], np.array([0.0, 1.0, 5.0, 10.0, 20.0]))
    rng = np.random.default_rng(seed + 10_000)
    stationary_by_tau = {}

    for tau_c in tau_values:
        voltage = simulate_free_voltage(
            mu_q=mu_q,
            sigma_q=sigma_q,
            tau_c=tau_c,
            dt=config["free_dt"],
            step_count=config["free_n_steps"],
            rng=rng,
        )
        stationary_by_tau[tau_c] = voltage[config["free_burn_in_steps"] :]

    all_values = np.concatenate(list(stationary_by_tau.values()))
    x_min = min(np.percentile(all_values, 0.2), mu_q - 4.0 * sigma_q)
    x_max = max(np.percentile(all_values, 99.8), mu_q + 4.0 * sigma_q)
    x = np.linspace(x_min, x_max, 600)
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for tau_index, tau_c in enumerate(tau_values):
        effective_std = np.sqrt(colored_free_variance(sigma_q, tau_c))
        color = colors[tau_index % len(colors)]
        axes[0].plot(x, normal_pdf(x, mu_q, effective_std), color=color, linewidth=2.0, label=tau_label(tau_c))
        if tau_index in (0, len(tau_values) - 1):
            axes[0].hist(
                stationary_by_tau[tau_c],
                bins=75,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.12,
                edgecolor=color,
                linewidth=1.1,
            )

    axes[0].axvline(1.0, color="0.35", linestyle=":", linewidth=1.5, label=r"$V_{th}$")
    axes[0].set_xlabel("free membrane potential")
    axes[0].set_ylabel("density")
    axes[0].set_title("stationary free-potential equilibrium")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    tau_grid = np.linspace(0.0, max(config["tau_c_values"][-1], 20.0), 300)
    theory_ratio = np.array([colored_free_variance(sigma_q, tau_c) / (sigma_q * sigma_q) for tau_c in tau_grid])
    empirical_ratio = np.array([np.var(stationary_by_tau[tau_c]) / (sigma_q * sigma_q) for tau_c in tau_values])

    axes[1].plot(tau_grid, theory_ratio, color="black", linewidth=2.2, label=r"$\tau_Q / (\tau_Q+\tau_c)$")
    axes[1].scatter(tau_values, empirical_ratio, color="tab:blue", s=42, zorder=3, label="simulation")
    axes[1].set_xlabel(r"noise correlation time $\tau_c$ (ms)")
    axes[1].set_ylabel("free-potential variance / white variance")
    axes[1].set_title("colored noise narrows the free equilibrium")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(rf"White vs colored free dynamics, $\mu_Q={mu_q:g}$, $\sigma_Q={sigma_q:g}$", y=1.02)
    fig.tight_layout()

    path = figure_dir / "10_free_potential_colored_vs_white_stationary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spike_density_by_regime(spike_samples, mu_values, tau_c_values, sigma_q, figure_dir):
    selected_mus = selected_existing(mu_values, np.array([0.85, 1.0, 1.2]))
    selected_taus = [0.0, float(tau_c_values[tau_c_values > 0.0][-1])]
    regime_labels = ["subthreshold", "near threshold", "suprathreshold"]
    colors = ["black", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)
    for ax, mu_q, regime_label in zip(axes, selected_mus, regime_labels):
        samples = [spike_samples[(float(mu_q), float(tau_c))] for tau_c in selected_taus]
        nonempty_samples = [sample for sample in samples if sample.size > 0]
        if nonempty_samples:
            upper_time = max(80.0, min(float(np.percentile(np.concatenate(nonempty_samples), 98.0)), 350.0))
        else:
            upper_time = 200.0
        bins = np.linspace(0.0, upper_time, 55)

        for tau_c, sample, color in zip(selected_taus, samples, colors):
            ax.hist(
                sample,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=2.0,
                label=tau_label(tau_c),
            )
        ax.set_title(rf"{regime_label}: $\mu_Q={mu_q:g}$")
        ax.set_xlabel("first-spike time (ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle(rf"Spike-time distributions: white vs strongest colored noise, $\sigma_Q={sigma_q:g}$", y=1.03)
    fig.tight_layout()

    path = figure_dir / "11_fpt_density_white_vs_colored_regimes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_variance_matched_free_equilibrium(config, sigma_q, seed, figure_dir):
    mu_q = 0.95
    colored_tau = float(config["tau_c_values"][config["tau_c_values"] > 0.0][-1])
    matched_sigma = variance_matched_sigma(sigma_q, colored_tau)
    rng = np.random.default_rng(seed + 80_000)
    conditions = [
        ("white", 0.0, sigma_q, "black"),
        (rf"colored same $\sigma_Q$, $\tau_c={colored_tau:g}$ ms", colored_tau, sigma_q, "tab:red"),
        (rf"colored matched variance, $\tau_c={colored_tau:g}$ ms", colored_tau, matched_sigma, "tab:blue"),
    ]

    stationary_by_label = {}
    for label, tau_c, input_sigma, _color in conditions:
        voltage = simulate_free_voltage(
            mu_q=mu_q,
            sigma_q=input_sigma,
            tau_c=tau_c,
            dt=config["free_dt"],
            step_count=config["free_n_steps"],
            rng=rng,
        )
        stationary_by_label[label] = voltage[config["free_burn_in_steps"] :]

    all_values = np.concatenate(list(stationary_by_label.values()))
    x_min = min(np.percentile(all_values, 0.2), mu_q - 4.0 * sigma_q)
    x_max = max(np.percentile(all_values, 99.8), mu_q + 4.0 * matched_sigma)
    x = np.linspace(x_min, x_max, 600)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for label, tau_c, input_sigma, color in conditions:
        effective_std = np.sqrt(colored_free_variance(input_sigma, tau_c))
        ax.hist(stationary_by_label[label], bins=75, density=True, histtype="step", color=color, linewidth=1.35)
        ax.plot(x, normal_pdf(x, mu_q, effective_std), color=color, linewidth=2.0, label=label)

    ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.5, label=r"$V_{th}$")
    ax.set_xlabel("free membrane potential")
    ax.set_ylabel("density")
    ax.set_title("Variance-matched free equilibrium")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = figure_dir / "12_variance_matched_free_equilibrium.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_variance_matched_transfer_curves(same_sigma_rows, matched_rows, mu_values, tau_c_values, sigma_q, figure_dir):
    colored_tau = float(tau_c_values[tau_c_values > 0.0][-1])
    same_sigma_lookup = {(row["mu_q"], row["tau_c_ms"]): row for row in same_sigma_rows}
    matched_lookup = {(row["mu_q"], row["tau_c_ms"]): row for row in matched_rows}
    series = [
        ("white", same_sigma_lookup, 0.0, "black", "o"),
        (rf"colored same $\sigma_Q$, $\tau_c={colored_tau:g}$ ms", same_sigma_lookup, colored_tau, "tab:red", "s"),
        (rf"colored matched variance, $\tau_c={colored_tau:g}$ ms", matched_lookup, colored_tau, "tab:blue", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharex=True)
    for label, row_lookup, tau_c, color, marker in series:
        rates = [row_lookup[(float(mu_q), float(tau_c))]["rate_hz"] for mu_q in mu_values]
        cvs = [row_lookup[(float(mu_q), float(tau_c))]["cv"] for mu_q in mu_values]
        axes[0].plot(mu_values, rates, color=color, marker=marker, linewidth=2.0, markersize=5, label=label)
        axes[1].plot(mu_values, cvs, color=color, marker=marker, linewidth=2.0, markersize=5, label=label)

    theory_rates = [siegert_firing_rate_hz(mu_q, sigma_q) for mu_q in mu_values]
    axes[0].plot(mu_values, theory_rates, color="0.35", linestyle="--", linewidth=1.5, label="white Siegert")

    for ax in axes:
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=1.1)
        ax.set_xlabel(r"$\mu_Q$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("transfer curve")
    axes[1].set_ylabel("ISI coefficient of variation")
    axes[1].set_title("irregularity")
    axes[0].legend(frameon=False)
    fig.suptitle(rf"Fair colored-noise comparison, target free $\sigma_Q={sigma_q:g}$", y=1.02)
    fig.tight_layout()

    path = figure_dir / "13_variance_matched_transfer_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    config = configure(args)
    figure_dir = output_dir()

    same_sigma_rows, spike_samples = run_spike_grid(config, args.sigma, args.seed, match_free_variance=False)
    matched_rows, _ = run_spike_grid(config, args.sigma, args.seed + 50_000, match_free_variance=True)

    figure_paths = [
        plot_free_distribution_comparison(config, args.sigma, args.seed, figure_dir),
        plot_spike_density_by_regime(spike_samples, config["mu_values"], config["tau_c_values"], args.sigma, figure_dir),
        plot_variance_matched_free_equilibrium(config, args.sigma, args.seed, figure_dir),
        plot_variance_matched_transfer_curves(
            same_sigma_rows, matched_rows, config["mu_values"], config["tau_c_values"], args.sigma, figure_dir
        ),
    ]

    print(f"Preset: {args.preset}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
