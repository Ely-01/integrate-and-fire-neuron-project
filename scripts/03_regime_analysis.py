import argparse
import csv
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
    cv_from_fpt,
    firing_rate_from_fpt,
    simulate_colored_first_passage_times,
    simulate_first_passage_times,
    simulate_voltage_traces,
    valid_first_passage_times,
)
from src.theory import siegert_firing_rate


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extended white-vs-colored-noise analysis with fair, same-scale comparisons."
    )
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
        defaults = {"dt": 0.1, "t_max": 800.0, "n_neurons": 300, "repeats": 2}
    elif args.preset == "publication":
        mu_values = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5])
        tau_c_values = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        defaults = {"dt": 0.05, "t_max": 2000.0, "n_neurons": 3000, "repeats": 8}
    else:
        mu_values = np.array([0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5])
        tau_c_values = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        defaults = {"dt": 0.1, "t_max": 1200.0, "n_neurons": 800, "repeats": 4}

    args.dt = defaults["dt"] if args.dt is None else args.dt
    args.t_max = defaults["t_max"] if args.t_max is None else args.t_max
    args.n_neurons = defaults["n_neurons"] if args.n_neurons is None else args.n_neurons
    args.validation_repeats = defaults["repeats"]
    return mu_values, tau_c_values


def output_dirs():
    fig_dir = ROOT / "outputs" / "figures"
    data_dir = ROOT / "outputs" / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, data_dir


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
    return fpt


def summarize_fpt(fpt, n_neurons):
    valid = valid_first_passage_times(fpt)
    row = {
        "n_spiked": int(valid.size),
        "spike_fraction": valid.size / n_neurons,
        "rate_hz": firing_rate_from_fpt(fpt),
        "cv": cv_from_fpt(fpt),
        "mean_ms": np.nan,
        "median_ms": np.nan,
        "q10_ms": np.nan,
        "q90_ms": np.nan,
        "q95_ms": np.nan,
        "iqr_ms": np.nan,
        "tail100_given_spike": np.nan,
        "tail150_given_spike": np.nan,
        "tail100_all": 0.0,
        "tail150_all": 0.0,
    }

    if valid.size == 0:
        return row

    q10, q25, q50, q75, q90, q95 = np.percentile(valid, [10, 25, 50, 75, 90, 95])
    row.update(
        {
            "mean_ms": float(np.mean(valid)),
            "median_ms": float(q50),
            "q10_ms": float(q10),
            "q90_ms": float(q90),
            "q95_ms": float(q95),
            "iqr_ms": float(q75 - q25),
            "tail100_given_spike": float(np.mean(valid > 100.0)),
            "tail150_given_spike": float(np.mean(valid > 150.0)),
            "tail100_all": float(np.sum(valid > 100.0) / n_neurons),
            "tail150_all": float(np.sum(valid > 150.0) / n_neurons),
        }
    )
    return row


def run_grid(mu_values, tau_c_values, args):
    rows = []
    samples = {}

    for tau_idx, tau_c in enumerate(tau_c_values):
        for mu_idx, mu in enumerate(mu_values):
            seed = args.seed + 100 * tau_idx + mu_idx
            fpt = run_condition(mu, args.sigma, tau_c, args, seed)
            metrics = summarize_fpt(fpt, args.n_neurons)
            metrics.update(
                {
                    "mu_q": float(mu),
                    "sigma_q": float(args.sigma),
                    "tau_c_ms": float(tau_c),
                    "dt_ms": float(args.dt),
                    "t_max_ms": float(args.t_max),
                    "n_neurons": int(args.n_neurons),
                    "seed": int(seed),
                }
            )
            rows.append(metrics)
            samples[(float(mu), float(tau_c))] = valid_first_passage_times(fpt)

    white_by_mu = {row["mu_q"]: row for row in rows if row["tau_c_ms"] == 0.0}
    for row in rows:
        white = white_by_mu[row["mu_q"]]
        row["rate_ratio_vs_white"] = safe_ratio(row["rate_hz"], white["rate_hz"])
        row["cv_delta_vs_white"] = safe_diff(row["cv"], white["cv"])
        row["q90_delta_vs_white_ms"] = safe_diff(row["q90_ms"], white["q90_ms"])
        row["tail100_delta_vs_white"] = safe_diff(row["tail100_all"], white["tail100_all"])

    return rows, samples


def safe_ratio(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return np.nan
    return float(value / reference)


def safe_diff(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    return float(value - reference)


def save_metrics(rows, data_dir):
    path = data_dir / "03_extended_regime_metrics.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def matrix_from_rows(rows, mu_values, tau_c_values, key):
    matrix = np.full((len(tau_c_values), len(mu_values)), np.nan, dtype=float)
    lookup = {(row["mu_q"], row["tau_c_ms"]): row for row in rows}
    for tau_idx, tau_c in enumerate(tau_c_values):
        for mu_idx, mu in enumerate(mu_values):
            matrix[tau_idx, mu_idx] = lookup[(float(mu), float(tau_c))][key]
    return matrix


def tau_label(tau_c):
    return "white" if tau_c == 0.0 else f"{tau_c:g}"


def add_heatmap(ax, matrix, mu_values, tau_c_values, title, cmap, cbar_label, vmin=None, vmax=None):
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r"$\mu_Q$")
    ax.set_ylabel(r"$\tau_c$ (ms)")
    ax.set_xticks(np.arange(len(mu_values)))
    ax.set_xticklabels([f"{mu:g}" for mu in mu_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tau_c_values)))
    ax.set_yticklabels([tau_label(tau) for tau in tau_c_values])
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_regime_heatmaps(rows, mu_values, tau_c_values, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    axes = axes.ravel()

    add_heatmap(
        axes[0],
        matrix_from_rows(rows, mu_values, tau_c_values, "rate_hz"),
        mu_values,
        tau_c_values,
        "firing rate",
        "viridis",
        "Hz",
    )
    add_heatmap(
        axes[1],
        matrix_from_rows(rows, mu_values, tau_c_values, "rate_ratio_vs_white"),
        mu_values,
        tau_c_values,
        "rate ratio vs white",
        "coolwarm",
        "colored / white",
        vmin=0.0,
        vmax=2.0,
    )
    add_heatmap(
        axes[2],
        matrix_from_rows(rows, mu_values, tau_c_values, "cv"),
        mu_values,
        tau_c_values,
        "ISI irregularity",
        "magma",
        "CV",
    )
    add_heatmap(
        axes[3],
        matrix_from_rows(rows, mu_values, tau_c_values, "cv_delta_vs_white"),
        mu_values,
        tau_c_values,
        "CV change vs white",
        "coolwarm",
        "delta CV",
        vmin=-0.5,
        vmax=0.5,
    )

    fig.suptitle(rf"Regime maps at fixed $\sigma_Q={rows[0]['sigma_q']:g}$", y=1.01)
    fig.tight_layout()
    path = fig_dir / "07_regime_maps_rate_cv.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tail_heatmaps(rows, mu_values, tau_c_values, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    axes = axes.ravel()

    add_heatmap(
        axes[0],
        matrix_from_rows(rows, mu_values, tau_c_values, "median_ms"),
        mu_values,
        tau_c_values,
        "median first-passage time",
        "cividis",
        "ms",
    )
    add_heatmap(
        axes[1],
        matrix_from_rows(rows, mu_values, tau_c_values, "q90_ms"),
        mu_values,
        tau_c_values,
        "90th percentile first-passage time",
        "cividis",
        "ms",
    )
    add_heatmap(
        axes[2],
        matrix_from_rows(rows, mu_values, tau_c_values, "tail100_all"),
        mu_values,
        tau_c_values,
        r"late-spike probability $P(T>100 ms)$",
        "plasma",
        "fraction of neurons",
    )
    add_heatmap(
        axes[3],
        matrix_from_rows(rows, mu_values, tau_c_values, "q90_delta_vs_white_ms"),
        mu_values,
        tau_c_values,
        "q90 change vs white",
        "coolwarm",
        "ms",
    )

    fig.suptitle("Tail and delay maps", y=1.01)
    fig.tight_layout()
    path = fig_dir / "08_regime_maps_tail_delays.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_lines(rows, mu_values, tau_c_values, fig_dir):
    metrics = [
        ("rate_hz", "firing rate (Hz)", "firing rate"),
        ("rate_ratio_vs_white", "colored / white", "rate ratio"),
        ("cv", "CV", "ISI irregularity"),
        ("q90_ms", "ms", "90th percentile FPT"),
        ("tail100_all", "fraction", r"$P(T>100 ms)$"),
        ("spike_fraction", "fraction", "spiking fraction"),
    ]
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    lookup = {(row["mu_q"], row["tau_c_ms"]): row for row in rows}

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    axes = axes.ravel()

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for idx, tau_c in enumerate(tau_c_values):
            values = [lookup[(float(mu), float(tau_c))][key] for mu in mu_values]
            ax.plot(
                mu_values,
                values,
                marker=markers[idx],
                color=colors[idx],
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
    fig.suptitle("Same-scale metric curves across input regimes", y=1.01)
    fig.tight_layout()
    path = fig_dir / "09_metric_lines_same_scale.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_empirical_gain(rows, mu_values, tau_c_values, fig_dir):
    selected_taus = nearest_values(tau_c_values, np.array([0.0, 2.0, 10.0]))
    colors = ["black", "tab:blue", "tab:orange"]
    markers = ["o", "s", "^"]
    lookup = {(row["mu_q"], row["tau_c_ms"]): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)

    for tau_c, color, marker in zip(selected_taus, colors, markers):
        rates = np.array([lookup[(float(mu), float(tau_c))]["rate_hz"] for mu in mu_values], dtype=float)
        gain = np.gradient(rates, mu_values)
        label = tau_label(tau_c)

        axes[0].plot(
            mu_values,
            rates,
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            mu_values,
            gain,
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=5,
            label=label,
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

    path = fig_dir / "14_empirical_gain_white_colored.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def empirical_survival(data, grid):
    if data.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.array([np.mean(data > t) for t in grid], dtype=float)


def nearest_values(values, targets):
    selected = []
    for target in targets:
        selected.append(float(values[np.argmin(np.abs(values - target))]))
    return selected


def plot_survival_and_density(samples, mu_values, tau_c_values, args, fig_dir):
    selected_mus = nearest_values(mu_values, np.array([0.85, 1.0, 1.2]))
    selected_taus = nearest_values(tau_c_values, np.array([0.0, 2.0, 10.0]))
    colors = ["black", "tab:blue", "tab:orange"]
    linestyles = ["-", "--", "-."]

    chosen = [samples[(mu, tau)] for mu in selected_mus for tau in selected_taus]
    nonempty = [data for data in chosen if data.size > 0]
    x_max = min(args.t_max, max(80.0, float(np.percentile(np.concatenate(nonempty), 98.0)))) if nonempty else args.t_max
    grid = np.linspace(0.0, x_max, 240)
    bins = np.linspace(0.0, x_max, 70)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharex=True, sharey=True)
    for ax, mu in zip(axes, selected_mus):
        for tau_c, color, linestyle in zip(selected_taus, colors, linestyles):
            data = samples[(mu, tau_c)]
            ax.plot(
                grid,
                empirical_survival(data, grid),
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=tau_label(tau_c),
            )
        ax.set_title(rf"$\mu_Q={mu:g}$")
        ax.set_xlabel("time (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"survival $S(t)=P(T>t)$")
    axes[0].legend(frameon=False)
    fig.suptitle("Survival curves on shared axes", y=1.03)
    fig.tight_layout()
    survival_path = fig_dir / "10_survival_curves_same_scale.png"
    fig.savefig(survival_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharex=True, sharey=True)
    for ax, mu in zip(axes, selected_mus):
        for tau_c, color, linestyle in zip(selected_taus, colors, linestyles):
            data = samples[(mu, tau_c)]
            ax.hist(
                data,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=tau_label(tau_c),
            )
        ax.set_title(rf"$\mu_Q={mu:g}$")
        ax.set_xlabel("first-passage time (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle("FPT densities on shared axes", y=1.03)
    fig.tight_layout()
    density_path = fig_dir / "11_fpt_densities_same_scale.png"
    fig.savefig(density_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return survival_path, density_path


def plot_voltage_traces(args, fig_dir):
    mu_values = [0.85, 1.0, 1.2]
    tau_values = [0.0, 10.0]
    colors = ["black", "tab:orange"]

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    for row_idx, mu in enumerate(mu_values):
        for col_idx, tau_c in enumerate(tau_values):
            ax = axes[row_idx, col_idx]
            t, traces = simulate_voltage_traces(
                mu=mu,
                sigma=args.sigma,
                tau_c=tau_c,
                dt=args.dt,
                t_max=min(args.t_max, 250.0),
                n_traces=8,
                seed=args.seed + 4000 + 10 * row_idx + col_idx,
            )
            for trace in traces:
                ax.plot(t, trace, color=colors[col_idx], linewidth=1.0, alpha=0.65)
            ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.1)
            ax.set_title(rf"$\mu_Q={mu:g}$, {tau_label(tau_c)}")
            ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("voltage")

    fig.suptitle(rf"Example trajectories before first crossing, $\sigma_Q={args.sigma:g}$", y=1.01)
    fig.tight_layout()
    path = fig_dir / "12_sample_voltage_trajectories.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_white_validation(mu_values, args, fig_dir):
    rates = []
    errors = []
    theory = [siegert_firing_rate(mu=mu, sigma=args.sigma) for mu in mu_values]

    validation_args = argparse.Namespace(**vars(args))
    validation_args.n_neurons = max(300, args.n_neurons // 2)

    for mu_idx, mu in enumerate(mu_values):
        replicate_rates = []
        for rep in range(args.validation_repeats):
            fpt = simulate_first_passage_times(
                mu=mu,
                sigma=args.sigma,
                dt=args.dt,
                t_max=args.t_max,
                n_neurons=validation_args.n_neurons,
                seed=args.seed + 5000 + 100 * mu_idx + rep,
            )
            replicate_rates.append(firing_rate_from_fpt(fpt))
        replicate_rates = np.asarray(replicate_rates, dtype=float)
        rates.append(float(np.mean(replicate_rates)))
        errors.append(float(np.std(replicate_rates, ddof=1)) if replicate_rates.size > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(mu_values, theory, color="black", linewidth=2.2, label="Siegert Eq. (43)")
    ax.errorbar(
        mu_values,
        rates,
        yerr=errors,
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
        capsize=3,
        label=f"Monte Carlo mean +/- SD, {args.validation_repeats} runs",
    )
    ax.axvline(1.0, color="0.55", linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"$\mu_Q$")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(rf"White-noise validation at fixed $\sigma_Q={args.sigma:g}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "13_white_siegert_validation_errorbars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    mu_values, tau_c_values = configure(args)
    fig_dir, data_dir = output_dirs()

    rows, samples = run_grid(mu_values, tau_c_values, args)
    csv_path = save_metrics(rows, data_dir)

    figure_paths = [
        plot_regime_heatmaps(rows, mu_values, tau_c_values, fig_dir),
        plot_tail_heatmaps(rows, mu_values, tau_c_values, fig_dir),
        plot_metric_lines(rows, mu_values, tau_c_values, fig_dir),
        plot_empirical_gain(rows, mu_values, tau_c_values, fig_dir),
    ]
    survival_path, density_path = plot_survival_and_density(samples, mu_values, tau_c_values, args, fig_dir)
    figure_paths.extend([survival_path, density_path])
    figure_paths.append(plot_voltage_traces(args, fig_dir))
    figure_paths.append(plot_white_validation(mu_values, args, fig_dir))

    print(f"Preset: {args.preset}")
    print(f"Metrics CSV: {csv_path}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
