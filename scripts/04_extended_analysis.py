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
from src.theory import siegert_firing_rate_hz


def build_parser():
    parser = argparse.ArgumentParser(description="Generate selected course-aligned figures.")
    parser.add_argument("--preset", choices=["quick", "standard", "publication"], default="standard")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def configure(args):
    if args.preset == "quick":
        return {
            "acf_dt": 0.05,
            "acf_n_steps": 50_000,
            "acf_max_lag_ms": 45.0,
            "tau_c_values": np.array([0.5, 2.0, 10.0]),
            "convergence_dt_values": np.array([1.0, 0.5, 0.25, 0.1]),
            "convergence_repeats": 2,
            "convergence_n_neurons": 1500,
            "convergence_t_max": 900.0,
        }

    if args.preset == "publication":
        return {
            "acf_dt": 0.025,
            "acf_n_steps": 220_000,
            "acf_max_lag_ms": 60.0,
            "tau_c_values": np.array([0.5, 1.0, 2.0, 5.0, 10.0]),
            "convergence_dt_values": np.array([1.0, 0.5, 0.25, 0.1, 0.05]),
            "convergence_repeats": 8,
            "convergence_n_neurons": 20_000,
            "convergence_t_max": 1800.0,
        }

    return {
        "acf_dt": 0.05,
        "acf_n_steps": 120_000,
        "acf_max_lag_ms": 55.0,
        "tau_c_values": np.array([0.5, 1.0, 2.0, 5.0, 10.0]),
        "convergence_dt_values": np.array([1.0, 0.5, 0.25, 0.1, 0.05]),
        "convergence_repeats": 5,
        "convergence_n_neurons": 10_000,
        "convergence_t_max": 1500.0,
    }


def output_dir():
    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def simulate_colored_noise(tau_c, dt, n_steps, rng):
    decay = np.exp(-dt / tau_c)
    step_std = np.sqrt((1.0 - decay * decay) / (2.0 * tau_c))
    noise = np.empty(int(n_steps), dtype=float)
    noise[0] = np.sqrt(1.0 / (2.0 * tau_c)) * rng.standard_normal()

    for index in range(1, int(n_steps)):
        noise[index] = decay * noise[index - 1] + step_std * rng.standard_normal()

    return noise


def empirical_autocorrelation(values, max_lag_steps):
    centered_values = np.asarray(values, dtype=float) - np.mean(values)
    sample_count = centered_values.size
    fft_size = 1 << (2 * sample_count - 1).bit_length()
    spectrum = np.fft.rfft(centered_values, n=fft_size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: max_lag_steps + 1]
    covariance /= np.arange(sample_count, sample_count - max_lag_steps - 1, -1)
    return covariance / covariance[0]


def plot_colored_noise_autocorrelation(config, seed, figure_dir):
    rng = np.random.default_rng(seed)
    dt = config["acf_dt"]
    max_lag_steps = int(round(config["acf_max_lag_ms"] / dt))
    lags = np.arange(max_lag_steps + 1) * dt
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for tau_index, tau_c in enumerate(config["tau_c_values"]):
        colored_noise = simulate_colored_noise(tau_c, dt, config["acf_n_steps"], rng)
        empirical_curve = empirical_autocorrelation(colored_noise, max_lag_steps)
        theoretical_curve = np.exp(-lags / tau_c)
        color = colors[tau_index % len(colors)]

        ax.plot(lags, empirical_curve, color=color, linewidth=1.8, label=rf"simulation $\tau_c={tau_c:g}$ ms")
        ax.plot(
            lags,
            theoretical_curve,
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.8,
            label=r"theory $e^{-lag/\tau_c}$" if tau_index == 0 else None,
        )

    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("normalized autocorrelation")
    ax.set_title("OU colored-noise autocorrelation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    path = figure_dir / "08_colored_noise_autocorrelation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_euler_maruyama_convergence(config, seed, figure_dir):
    mu_q = 1.2
    sigma_q = 0.25
    theory_rate = siegert_firing_rate_hz(mu_q, sigma_q)
    rows = []

    for dt_index, dt in enumerate(config["convergence_dt_values"]):
        rates = []
        for repeat in range(config["convergence_repeats"]):
            spike_times = simulate_white_spike_times(
                mu_q=mu_q,
                sigma_q=sigma_q,
                dt=float(dt),
                t_max=config["convergence_t_max"],
                n_neurons=config["convergence_n_neurons"],
                seed=seed + 1000 + 100 * dt_index + repeat,
            )
            rates.append(firing_rate_hz(spike_times))

        rates = np.asarray(rates, dtype=float)
        mean_rate = float(np.mean(rates))
        rows.append(
            {
                "dt_ms": float(dt),
                "mean_rate_hz": mean_rate,
                "sd_rate_hz": float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0,
                "relative_error": float(abs(mean_rate - theory_rate) / theory_rate),
            }
        )

    dt_values = np.array([row["dt_ms"] for row in rows], dtype=float)
    mean_rates = np.array([row["mean_rate_hz"] for row in rows], dtype=float)
    sd_rates = np.array([row["sd_rate_hz"] for row in rows], dtype=float)
    relative_errors = np.array([row["relative_error"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].errorbar(dt_values, mean_rates, yerr=sd_rates, fmt="o-", color="tab:blue", capsize=3)
    axes[0].axhline(theory_rate, color="black", linestyle="--", linewidth=1.6, label="Siegert Eq. (43)")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"Euler-Maruyama step $\Delta t$ (ms)")
    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("rate estimate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(dt_values, relative_errors, marker="o", color="tab:red", linewidth=2.0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Euler-Maruyama step $\Delta t$ (ms)")
    axes[1].set_ylabel("absolute relative error")
    axes[1].set_title("convergence to Eq. (43)")
    axes[1].grid(alpha=0.25, which="both")

    fig.suptitle(rf"Euler-Maruyama convergence, $\mu_Q={mu_q:g}$, $\sigma_Q={sigma_q:g}$", y=1.02)
    fig.tight_layout()

    path = figure_dir / "09_euler_maruyama_convergence.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    config = configure(args)
    figure_dir = output_dir()

    figure_paths = [
        plot_colored_noise_autocorrelation(config, args.seed, figure_dir),
        plot_euler_maruyama_convergence(config, args.seed, figure_dir),
    ]

    print(f"Preset: {args.preset}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
