import numpy as np

try:
    from scipy.integrate import quad
    from scipy.special import erfcx
except ModuleNotFoundError:
    import math

    def erfcx(x):
        if x > 25.0:
            inverse_x = 1.0 / x
            return inverse_x / math.sqrt(math.pi) * (1.0 - 0.5 * inverse_x * inverse_x)
        return math.exp(x * x) * math.erfc(x)

    def quad(function, lower, upper, limit=200):
        if lower == upper:
            return 0.0, 0.0

        interval_count = max(512, int(abs(upper - lower) * 256))
        if interval_count % 2 == 1:
            interval_count += 1

        points = np.linspace(lower, upper, interval_count + 1)
        values = np.array([function(float(point)) for point in points], dtype=float)
        step = (upper - lower) / interval_count
        integral = step / 3.0 * (
            values[0] + values[-1] + 4.0 * np.sum(values[1:-1:2]) + 2.0 * np.sum(values[2:-2:2])
        )
        return float(integral), 0.0


def deterministic_isi_ms(mu_q, tau_q=20.0, threshold=1.0, reset=0.0, refractory_ms=0.0):
    """Noiseless LIF interspike interval in ms."""
    if mu_q <= threshold:
        return np.inf

    numerator = mu_q - reset
    denominator = mu_q - threshold
    if numerator <= 0.0 or denominator <= 0.0:
        return np.inf

    return refractory_ms + tau_q * np.log(numerator / denominator)


def siegert_mean_isi_ms(
    mu_q,
    sigma_q,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    refractory_ms=0.0,
):
    """Mean first-passage time from Burkitt Eq. (43), in ms."""
    if sigma_q <= 0.0:
        return deterministic_isi_ms(mu_q, tau_q, threshold, reset, refractory_ms)

    lower = (reset - mu_q) / (sigma_q * np.sqrt(2.0))
    upper = (threshold - mu_q) / (sigma_q * np.sqrt(2.0))
    integral, _ = quad(lambda x: erfcx(-x), lower, upper, limit=200)
    return refractory_ms + tau_q * np.sqrt(np.pi) * integral


def siegert_firing_rate_hz(
    mu_q,
    sigma_q,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    refractory_ms=0.0,
):
    """Output firing rate from Burkitt Eq. (43), in Hz."""
    mean_isi = siegert_mean_isi_ms(mu_q, sigma_q, tau_q, threshold, reset, refractory_ms)
    if not np.isfinite(mean_isi) or mean_isi <= 0.0:
        return 0.0
    return 1000.0 / mean_isi


def gain_function(x, sigma_over_threshold):
    """Dimensionless gain function F from Burkitt Eq. (61)."""
    x_values = np.atleast_1d(np.asarray(x, dtype=float))
    ratio = float(sigma_over_threshold)
    if ratio <= 0.0:
        raise ValueError("sigma_over_threshold must be positive")

    width = 1.0 / (np.sqrt(2.0) * ratio)
    values = np.empty_like(x_values)

    for index, x_value in enumerate(x_values):
        lower = -x_value - width
        upper = -x_value
        integral, _ = quad(lambda u: erfcx(-u), lower, upper, limit=200)
        values[index] = 1.0 / (np.sqrt(np.pi) * integral)

    if np.ndim(x) == 0:
        return float(values[0])
    return values
