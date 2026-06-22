import numpy as np

try:
    from scipy.integrate import quad as _quad
    from scipy.special import erfcx as _erfcx
except ModuleNotFoundError:
    import math

    def _erfcx(x):
        if x > 25.0:
            inv = 1.0 / x
            return inv / math.sqrt(math.pi) * (1.0 - 0.5 * inv * inv)
        return math.exp(x * x) * math.erfc(x)

    def _quad(func, lower, upper, limit=200):
        if lower == upper:
            return 0.0, 0.0

        width = abs(upper - lower)
        n_intervals = max(512, int(width * 256))
        if n_intervals % 2 == 1:
            n_intervals += 1

        xs = np.linspace(lower, upper, n_intervals + 1)
        ys = np.array([func(float(x)) for x in xs], dtype=float)
        h = (upper - lower) / n_intervals
        value = h / 3.0 * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-2:2]))
        return float(value), 0.0


def _coalesce(value, alias, name):
    if value is None and alias is None:
        raise ValueError(f"{name} must be provided")
    if value is not None and alias is not None:
        raise ValueError(f"Provide either {name} or its alias, not both")
    return alias if alias is not None else value


def deterministic_isi(mu=None, tau=20.0, v_th=1.0, v_reset=0.0, refractory_period=0.0, *, mu_q=None, tau_q=None):
    """Interspike interval for the noiseless leaky integrate-and-fire model.

    Constant-input result. A spike is possible only when mu_q > v_th.
    """
    mu_q = float(_coalesce(mu, mu_q, "mu"))
    tau_q = float(tau if tau_q is None else tau_q)

    if mu_q <= v_th:
        return np.inf

    numerator = mu_q - v_reset
    denominator = mu_q - v_th
    if numerator <= 0.0 or denominator <= 0.0:
        return np.inf

    return refractory_period + tau_q * np.log(numerator / denominator)


def deterministic_firing_rate(
    mu=None,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    refractory_period=0.0,
    time_scale=1000.0,
    *,
    mu_q=None,
    tau_q=None,
):
    isi = deterministic_isi(
        mu=mu,
        tau=tau,
        v_th=v_th,
        v_reset=v_reset,
        refractory_period=refractory_period,
        mu_q=mu_q,
        tau_q=tau_q,
    )
    if not np.isfinite(isi) or isi <= 0.0:
        return 0.0
    return time_scale / isi


def siegert_mean_isi(
    mu=None,
    sigma=None,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    refractory_period=0.0,
    *,
    mu_q=None,
    sigma_q=None,
    tau_q=None,
):
    """Mean first-passage time for the LIF model"""
    mu_q = float(_coalesce(mu, mu_q, "mu"))
    sigma_q = float(_coalesce(sigma, sigma_q, "sigma"))
    tau_q = float(tau if tau_q is None else tau_q)

    if sigma_q <= 0.0:
        return deterministic_isi(
            mu_q=mu_q,
            tau_q=tau_q,
            v_th=v_th,
            v_reset=v_reset,
            refractory_period=refractory_period,
        )

    lower = (v_reset - mu_q) / (sigma_q * np.sqrt(2.0))
    upper = (v_th - mu_q) / (sigma_q * np.sqrt(2.0))

    integral_value, _ = _quad(lambda x: _erfcx(-x), lower, upper, limit=200)
    return refractory_period + tau_q * np.sqrt(np.pi) * integral_value


def siegert_firing_rate(
    mu=None,
    sigma=None,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    refractory_period=0.0,
    time_scale=1000.0,
    *,
    mu_q=None,
    sigma_q=None,
    tau_q=None,
):
    """Output firing rate from the Siegert formula, in Hz for millisecond units."""
    mean_isi = siegert_mean_isi(
        mu=mu,
        sigma=sigma,
        tau=tau,
        v_th=v_th,
        v_reset=v_reset,
        refractory_period=refractory_period,
        mu_q=mu_q,
        sigma_q=sigma_q,
        tau_q=tau_q,
    )
    if not np.isfinite(mean_isi) or mean_isi <= 0.0:
        return 0.0
    return time_scale / mean_isi


def gain_function_F(x, sigma_over_theta):
    """Dimensionless gain function F.
    
    x = (mu_q - v_th) / (sqrt(2) sigma_q)
    sigma_over_theta = sigma_q / (v_th - v_reset)
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    r = float(sigma_over_theta)
    if r <= 0.0:
        raise ValueError("sigma_over_theta must be positive")

    width = 1.0 / (np.sqrt(2.0) * r)
    values = np.empty_like(x_arr)

    for idx, xi in enumerate(x_arr):
        lower = -xi - width
        upper = -xi
        integral_value, _ = _quad(lambda u: _erfcx(-u), lower, upper, limit=200)
        values[idx] = 1.0 / (np.sqrt(np.pi) * integral_value)

    if np.ndim(x) == 0:
        return float(values[0])
    return values
