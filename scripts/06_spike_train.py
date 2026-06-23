import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for plotting. Run: pip install -r requirements.txt") from exc
import numpy as np
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import simulate_spike_trains


def build_parser():
    parser = argparse.ArgumentParser(
        description="Show the temporal signature of colored noise at matched variance and firing rate."
    )
    parser.add_argument("--preset", choices=["quick", "standard", "publication"], default="standard")
    parser.add_argument("--mu", type=float, default=0.9, help="white-noise reference mu_Q")
    parser.add_argument("--sigma", type=float, default=0.25, help="white-noise reference sigma_Q")
    parser.add_argument("--seed", type=int, default=27182)
    return parser


def configure(preset):
    if preset == "quick":
        return {
            "tau_values": np.array([0.0, 5.0, 20.0]),
            "dt": 0.2,
            "duration_ms": 30000.0,
            "trial_count": 12,
            "burn_in_ms": 1000.0,
            "calibration_duration_ms": 5000.0,
            "calibration_trials": 10,
            "calibration_steps": 7,
        }
    if preset == "publication":
        return {
            "tau_values": np.array([0.0, 2.0, 5.0, 10.0, 20.0]),
            "dt": 0.05,
            "duration_ms": 180000.0,
            "trial_count": 40,
            "burn_in_ms": 3000.0,
            "calibration_duration_ms": 20000.0,
            "calibration_trials": 32,
            "calibration_steps": 10,
        }
    return {
        "tau_values": np.array([0.0, 2.0, 5.0, 10.0, 20.0]),
        "dt": 0.1,
        "duration_ms": 90000.0,
        "trial_count": 24,
        "burn_in_ms": 2000.0,
        "calibration_duration_ms": 10000.0,
        "calibration_trials": 18,
        "calibration_steps": 8,
    }


def output_dir():
    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def variance_matched_sigma(reference_sigma, tau_c, tau_q=20.0):
    return reference_sigma * np.sqrt((tau_q + tau_c) / tau_q)


def firing_rate_hz(spike_trains, duration_ms):
    spike_count = sum(train.size for train in spike_trains)
    return 1000.0 * spike_count / (len(spike_trains) * duration_ms)


def match_firing_rate(target_rate, sigma_q, tau_c, config, seed):
    lower_mu = 0.45
    upper_mu = 1.45

    for iteration in range(config["calibration_steps"]):
        test_mu = 0.5 * (lower_mu + upper_mu)
        trains = simulate_spike_trains(
            mu_q=test_mu,
            sigma_q=sigma_q,
            tau_c=tau_c,
            dt=config["dt"],
            duration_ms=config["calibration_duration_ms"],
            trial_count=config["calibration_trials"],
            burn_in_ms=config["burn_in_ms"],
            seed=seed + iteration,
        )
        rate = firing_rate_hz(trains, config["calibration_duration_ms"])
        if rate < target_rate:
            lower_mu = test_mu
        else:
            upper_mu = test_mu

    return 0.5 * (lower_mu + upper_mu)


def coefficient_of_variation(spike_trains):
    intervals = [np.diff(train) for train in spike_trains if train.size >= 3]
    if not intervals:
        return np.nan
    all_intervals = np.concatenate(intervals)
    return float(np.std(all_intervals, ddof=1) / np.mean(all_intervals))


def fano_factor(spike_trains, duration_ms, window_ms):
    counts = []
    edges = np.arange(0.0, duration_ms + window_ms, window_ms)
    for train in spike_trains:
        trial_counts, _ = np.histogram(train, bins=edges)
        counts.extend(trial_counts)
    counts = np.asarray(counts, dtype=float)
    if counts.size < 2 or np.mean(counts) == 0.0:
        return np.nan
    return float(np.var(counts, ddof=1) / np.mean(counts))


def spike_train_spectrum(spike_trains, duration_ms, bin_ms=1.0):
    sample_rate_hz = 1000.0 / bin_ms
    bin_count = int(duration_ms / bin_ms)
    spectra = []

    for train in spike_trains:
        indices = (train / bin_ms).astype(int)
        indices = indices[(indices >= 0) & (indices < bin_count)]
        counts = np.bincount(indices, minlength=bin_count).astype(float)
        rate_signal = counts * sample_rate_hz
        frequencies, power = welch(
            rate_signal,
            fs=sample_rate_hz,
            nperseg=min(8192, bin_count),
            noverlap=min(4096, bin_count // 2),
            detrend="constant",
            scaling="density",
        )
        mean_rate = 1000.0 * train.size / duration_ms
        if mean_rate > 0.0:
            # Welch returns a one-sided spectrum, whose positive-frequency
            # Poisson baseline is 2 * rate.
            spectra.append(power / (2.0 * mean_rate))

    return frequencies, np.asarray(spectra)


def prepare_conditions(reference_mu, reference_sigma, config, seed):
    white_trains = simulate_spike_trains(
        mu_q=reference_mu,
        sigma_q=reference_sigma,
        tau_c=0.0,
        dt=config["dt"],
        duration_ms=config["duration_ms"],
        trial_count=config["trial_count"],
        burn_in_ms=config["burn_in_ms"],
        seed=seed,
    )
    target_rate = firing_rate_hz(white_trains, config["duration_ms"])
    conditions = [
        {
            "tau_c": 0.0,
            "mu_q": reference_mu,
            "sigma_q": reference_sigma,
            "trains": white_trains,
        }
    ]

    for tau_index, tau_c in enumerate(config["tau_values"][1:], start=1):
        matched_sigma = variance_matched_sigma(reference_sigma, tau_c)
        matched_mu = match_firing_rate(
            target_rate,
            matched_sigma,
            tau_c,
            config,
            seed + 10000 * tau_index,
        )
        trains = simulate_spike_trains(
            mu_q=matched_mu,
            sigma_q=matched_sigma,
            tau_c=tau_c,
            dt=config["dt"],
            duration_ms=config["duration_ms"],
            trial_count=config["trial_count"],
            burn_in_ms=config["burn_in_ms"],
            seed=seed + 20000 * tau_index,
        )
        conditions.append(
            {
                "tau_c": float(tau_c),
                "mu_q": matched_mu,
                "sigma_q": matched_sigma,
                "trains": trains,
            }
        )

    return conditions, target_rate


def condition_label(condition):
    if condition["tau_c"] == 0.0:
        return "white"
    return rf"$\tau_c={condition['tau_c']:g}$ ms"


def plot_fano_scaling(conditions, duration_ms, figure_dir):
    window_values = np.array([20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for condition, color, marker in zip(conditions, colors, markers):
        fano_values = [
            fano_factor(condition["trains"], duration_ms, window_ms)
            for window_ms in window_values
            if window_ms <= duration_ms / 4.0
        ]
        valid_windows = window_values[: len(fano_values)]
        axes[0].plot(
            valid_windows,
            fano_values,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=condition_label(condition),
        )

    tau_values = [condition["tau_c"] for condition in conditions]
    cv_values = [coefficient_of_variation(condition["trains"]) for condition in conditions]
    rates = [firing_rate_hz(condition["trains"], duration_ms) for condition in conditions]
    rate_ratios = np.asarray(rates) / rates[0]
    long_window = min(2000.0, duration_ms / 4.0)
    long_fano = [
        fano_factor(condition["trains"], duration_ms, long_window)
        for condition in conditions
    ]
    axes[1].plot(tau_values, cv_values, color="tab:blue", marker="o", linewidth=2.0, label="ISI CV")
    axes[1].plot(
        tau_values,
        long_fano,
        color="tab:red",
        marker="s",
        linewidth=2.0,
        label=rf"Fano, $T={long_window:g}$ ms",
    )
    axes[1].plot(
        tau_values,
        rate_ratios,
        color="0.35",
        marker="^",
        linestyle="--",
        linewidth=1.8,
        label="rate / white rate",
    )

    axes[0].axhline(1.0, color="0.55", linestyle=":", linewidth=1.2, label="Poisson")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("counting window (ms)")
    axes[0].set_ylabel("Fano factor")
    axes[0].set_title("variability depends on observation scale")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_xlabel(r"correlation time $\tau_c$ (ms)")
    axes[1].set_ylabel("variability index")
    axes[1].set_title("rate stays fixed while variability grows")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle("Hidden temporal variability at matched firing rate", y=1.02)
    fig.tight_layout()

    path = figure_dir / "14_fano_factor_memory_scaling.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spike_spectrum(conditions, duration_ms, figure_dir):
    selected_indices = sorted(set([0, len(conditions) // 2, len(conditions) - 1]))
    colors = ["black", "tab:orange", "tab:red"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for condition_index, color in zip(selected_indices, colors):
        condition = conditions[condition_index]
        frequencies, spectra = spike_train_spectrum(condition["trains"], duration_ms)
        mean_spectrum = np.mean(spectra, axis=0)
        positive = (frequencies >= 0.25) & (frequencies <= 100.0)
        ax.semilogx(
            frequencies[positive],
            mean_spectrum[positive],
            color=color,
            linewidth=2.0,
            label=condition_label(condition),
        )

    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=1.2, label="Poisson baseline")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power / mean firing rate")
    ax.set_title("Normalized spike-train power spectrum")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()

    path = figure_dir / "15_spike_train_power_spectrum.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    config = configure(args.preset)
    figure_dir = output_dir()
    conditions, target_rate = prepare_conditions(args.mu, args.sigma, config, args.seed)

    paths = [
        plot_fano_scaling(conditions, config["duration_ms"], figure_dir),
        plot_spike_spectrum(conditions, config["duration_ms"], figure_dir),
    ]

    print(f"Preset: {args.preset}")
    print(f"White reference rate: {target_rate:.3f} Hz")
    print("Matched conditions:")
    for condition in conditions:
        rate = firing_rate_hz(condition["trains"], config["duration_ms"])
        cv = coefficient_of_variation(condition["trains"])
        fano = fano_factor(condition["trains"], config["duration_ms"], 2000.0)
        print(
            f"- tau_c={condition['tau_c']:g} ms, mu_Q={condition['mu_q']:.4f}, "
            f"sigma_Q={condition['sigma_q']:.4f}, rate={rate:.3f} Hz, CV={cv:.3f}, Fano_2s={fano:.3f}"
        )
    print("Generated figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
