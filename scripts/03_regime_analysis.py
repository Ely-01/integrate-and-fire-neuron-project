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

from src.engine import (
    coefficient_of_variation,
    finite_spike_times,
    firing_rate_hz,
    simulate_colored_spike_times,
    simulate_voltage_traces,
    simulate_white_spike_times,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate selected white-vs-colored-noise regime figures.")
    parser.add_argument("--preset", choices=["quick", "standard", "publication"], default="standard")
    parser.add_argument("--sigma", type=float, default=0.25, help="stationary free-potential sigma_Q")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--n-neurons", type=int, default=None)
    parser.add_argument("--seed", type=int, default=900)
    return parser


def configure(args):
    if args.preset == "quick":
        mu_values = np.array([0.8, 0.95, 1.0, 1.1, 1.25])
        tau_c_values = np.array([0.0, 2.0, 10.0])
        defaults = {"dt": 0.1, "t_max": 800.0, "n_neurons": 300}
    elif args.preset == "publication":
        mu_values = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5])
        tau_c_values = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        defaults = {"dt": 0.05, "t_max": 2000.0, "n_neurons": 3000}
    else:
        mu_values = np.array([0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5])
        tau_c_values = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        defaults = {"dt": 0.1, "t_max": 1200.0, "n_neurons": 800}

    args.dt = defaults["dt"] if args.dt is None else args.dt
    args.t_max = defaults["t_max"] if args.t_max is None else args.t_max
    args.n_neurons = defaults["n_neurons"] if args.n_neurons is None else args.n_neurons
    return mu_values, tau_c_values


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


def summarize_spike_times(spike_times, n_neurons):
    valid_times = finite_spike_times(spike_times)
    summary = {
        "n_spiked": int(valid_times.size),
        "spike_fraction": float(valid_times.size / n_neurons),
        "rate_hz": firing_rate_hz(spike_times),
        "cv": coefficient_of_variation(spike_times),
        "mean_ms": np.nan,
        "median_ms": np.nan,
        "q90_ms": np.nan,
        "tail100_all": 0.0,
    }

    if valid_times.size == 0:
        return summary

    summary.update(
        {
            "mean_ms": float(np.mean(valid_times)),
            "median_ms": float(np.median(valid_times)),
            "q90_ms": float(np.percentile(valid_times, 90.0)),
            "tail100_all": float(np.sum(valid_times > 100.0) / n_neurons),
        }
    )
    return summary


def safe_ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return np.nan
    return float(value / reference)


def run_grid(mu_values, tau_c_values, args):
    rows = []
    spike_samples = {}

    for tau_index, tau_c in enumerate(tau_c_values):
        for mu_index, mu_q in enumerate(mu_values):
            seed = args.seed + 100 * tau_index + mu_index
            spike_times = simulate_spike_times(mu_q, args.sigma, tau_c, args, seed)
            row = summarize_spike_times(spike_times, args.n_neurons)
            row.update({"mu_q": float(mu_q), "sigma_q": float(args.sigma), "tau_c_ms": float(tau_c)})
            rows.append(row)
            spike_samples[(float(mu_q), float(tau_c))] = finite_spike_times(spike_times)

    white_rate_by_mu = {row["mu_q"]: row["rate_hz"] for row in rows if row["tau_c_ms"] == 0.0}
    for row in rows:
        row["rate_ratio_vs_white"] = safe_ratio(row["rate_hz"], white_rate_by_mu[row["mu_q"]])

    return rows, spike_samples


def tau_label(tau_c):
    return "white" if tau_c == 0.0 else rf"$\tau_c={tau_c:g}$ ms"


def nearest_values(values, targets):
    return [float(values[np.argmin(np.abs(values - target))]) for target in targets]


def plot_metric_lines(rows, mu_values, tau_c_values, figure_dir):
    row_by_condition = {(row["mu_q"], row["tau_c_ms"]): row for row in rows}
    panels = [
        ("rate_hz", "firing rate (Hz)", "firing rate"),
        ("rate_ratio_vs_white", "colored / white", "rate ratio"),
        ("cv", "CV", "ISI irregularity"),
        ("q90_ms", "ms", "90th percentile spike time"),
        ("tail100_all", "fraction", r"$P(T>100 ms)$"),
        ("spike_fraction", "fraction", "spiking fraction"),
    ]
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    axes = axes.ravel()

    for ax, (metric_name, ylabel, title) in zip(axes, panels):
        for tau_index, tau_c in enumerate(tau_c_values):
            values = [row_by_condition[(float(mu_q), float(tau_c))][metric_name] for mu_q in mu_values]
            ax.plot(
                mu_values,
                values,
                marker=markers[tau_index % len(markers)],
                color=colors[tau_index % len(colors)],
                linewidth=2.0,
                markersize=5,
                label=tau_label(tau_c),
            )
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    for ax in axes[-3:]:
        ax.set_xlabel(r"$\mu_Q$")

    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(rf"White vs colored noise at fixed $\sigma_Q={rows[0]['sigma_q']:g}$", y=1.01)
    fig.tight_layout()

    path = figure_dir / "04_metric_lines_same_scale.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def survival_probability(spike_times, time_grid):
    if spike_times.size == 0:
        return np.full_like(time_grid, np.nan, dtype=float)
    return np.array([np.mean(spike_times > time) for time in time_grid], dtype=float)


def plot_survival_curves(spike_samples, mu_values, tau_c_values, args, figure_dir):
    selected_mus = nearest_values(mu_values, np.array([0.85, 1.0, 1.2]))
    selected_taus = list(dict.fromkeys(nearest_values(tau_c_values, np.array([0.0, 2.0, 10.0]))))
    colors = ["black", "tab:blue", "tab:orange"]
    linestyles = ["-", "--", "-."]

    selected_samples = [spike_samples[(mu_q, tau_c)] for mu_q in selected_mus for tau_c in selected_taus]
    nonempty_samples = [sample for sample in selected_samples if sample.size > 0]
    if nonempty_samples:
        max_time = min(args.t_max, max(80.0, float(np.percentile(np.concatenate(nonempty_samples), 98.0))))
    else:
        max_time = args.t_max
    time_grid = np.linspace(0.0, max_time, 240)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharex=True, sharey=True)
    for ax, mu_q in zip(axes, selected_mus):
        for tau_index, tau_c in enumerate(selected_taus):
            sample = spike_samples[(mu_q, tau_c)]
            ax.plot(
                time_grid,
                survival_probability(sample, time_grid),
                color=colors[tau_index % len(colors)],
                linestyle=linestyles[tau_index % len(linestyles)],
                linewidth=2.0,
                label=tau_label(tau_c),
            )
        ax.set_title(rf"$\mu_Q={mu_q:g}$")
        ax.set_xlabel("time (ms)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"survival $S(t)=P(T>t)$")
    axes[0].legend(frameon=False)
    fig.suptitle("First-spike survival curves on shared axes", y=1.03)
    fig.tight_layout()

    path = figure_dir / "05_survival_curves_same_scale.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_voltage_traces(args, figure_dir):
    mu_values = [0.85, 1.0, 1.2]
    tau_values = [0.0, 10.0]
    colors = ["black", "tab:orange"]

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    for row_index, mu_q in enumerate(mu_values):
        for col_index, tau_c in enumerate(tau_values):
            ax = axes[row_index, col_index]
            times, traces = simulate_voltage_traces(
                mu_q=mu_q,
                sigma_q=args.sigma,
                tau_c=tau_c,
                dt=args.dt,
                t_max=min(args.t_max, 250.0),
                trace_count=8,
                seed=args.seed + 4000 + 10 * row_index + col_index,
            )
            for trace in traces:
                ax.plot(times, trace, color=colors[col_index], linewidth=1.0, alpha=0.65)
            ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.1)
            ax.set_title(rf"$\mu_Q={mu_q:g}$, {tau_label(tau_c)}")
            ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("voltage")

    fig.suptitle(rf"Example voltage trajectories, $\sigma_Q={args.sigma:g}$", y=1.01)
    fig.tight_layout()

    path = figure_dir / "06_sample_voltage_trajectories.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_empirical_gain(rows, mu_values, tau_c_values, figure_dir):
    selected_taus = list(dict.fromkeys(nearest_values(tau_c_values, np.array([0.0, 2.0, 10.0]))))
    row_by_condition = {(row["mu_q"], row["tau_c_ms"]): row for row in rows}
    colors = ["black", "tab:blue", "tab:orange"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)
    for tau_index, tau_c in enumerate(selected_taus):
        rates = np.array([row_by_condition[(float(mu_q), float(tau_c))]["rate_hz"] for mu_q in mu_values], dtype=float)
        gain = np.gradient(rates, mu_values)
        axes[0].plot(
            mu_values,
            rates,
            marker=markers[tau_index % len(markers)],
            color=colors[tau_index % len(colors)],
            linewidth=2.0,
            markersize=5,
            label=tau_label(tau_c),
        )
        axes[1].plot(
            mu_values,
            gain,
            marker=markers[tau_index % len(markers)],
            color=colors[tau_index % len(colors)],
            linewidth=2.0,
            markersize=5,
            label=tau_label(tau_c),
        )

    for ax in axes:
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=1.1)
        ax.set_xlabel(r"$\mu_Q$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("empirical transfer curve")
    axes[1].set_ylabel(r"empirical gain $d\lambda_{out}/d\mu_Q$")
    axes[1].set_title("colored-noise gain")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Empirical gain from simulations: white vs colored noise", y=1.02)
    fig.tight_layout()

    path = figure_dir / "07_empirical_gain_white_colored.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    mu_values, tau_c_values = configure(args)
    figure_dir = output_dir()

    rows, spike_samples = run_grid(mu_values, tau_c_values, args)
    figure_paths = [
        plot_metric_lines(rows, mu_values, tau_c_values, figure_dir),
        plot_survival_curves(spike_samples, mu_values, tau_c_values, args, figure_dir),
        plot_voltage_traces(args, figure_dir),
        plot_empirical_gain(rows, mu_values, tau_c_values, figure_dir),
    ]

    print(f"Preset: {args.preset}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
