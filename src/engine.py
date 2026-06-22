import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func


@njit
def _simulate_white_fpt(mu_q, sigma_q, tau_q, v_th, v_reset, dt, t_max, n_neurons, refractory_period, seed):
    if seed >= 0:
        np.random.seed(seed)

    n_steps = int(np.ceil(t_max / dt))
    sqrt_dt = np.sqrt(dt)
    noise_coeff = sigma_q * np.sqrt(2.0 / tau_q)

    v = np.full(n_neurons, v_reset, dtype=np.float64)
    fpt = np.full(n_neurons, np.nan, dtype=np.float64)
    has_spiked = np.zeros(n_neurons, dtype=np.bool_)
    spiked_count = 0

    for step in range(n_steps):
        if spiked_count == n_neurons:
            break

        t_left = step * dt

        for i in range(n_neurons):
            if has_spiked[i]:
                continue

            v_old = v[i]
            drift = (mu_q - v_old) / tau_q
            noise = 0.0
            if sigma_q > 0.0:
                noise = noise_coeff * sqrt_dt * np.random.randn()

            v_new = v_old + drift * dt + noise
            v[i] = v_new

            if v_new >= v_th:
                frac = 1.0
                if v_new != v_old:
                    frac = (v_th - v_old) / (v_new - v_old)
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0

                fpt[i] = refractory_period + t_left + frac * dt
                has_spiked[i] = True
                spiked_count += 1

    return fpt


@njit
def _simulate_colored_fpt(
    mu_q,
    sigma_q,
    tau_c,
    tau_q,
    v_th,
    v_reset,
    dt,
    t_max,
    n_neurons,
    refractory_period,
    seed,
):
    if tau_c <= 0.0:
        return _simulate_white_fpt(
            mu_q, sigma_q, tau_q, v_th, v_reset, dt, t_max, n_neurons, refractory_period, seed
        )

    if seed >= 0:
        np.random.seed(seed)

    n_steps = int(np.ceil(t_max / dt))
    colored_to_voltage = sigma_q * np.sqrt(2.0 / tau_q)

    # OU colored noise eta has autocovariance exp(-|t| / tau_c) / (2 tau_c).
    # This normalization converges to white noise as tau_c -> 0.
    decay = np.exp(-dt / tau_c)
    eta_step_std = np.sqrt((1.0 - decay * decay) / (2.0 * tau_c))
    eta_stationary_std = np.sqrt(1.0 / (2.0 * tau_c))

    v = np.full(n_neurons, v_reset, dtype=np.float64)
    eta = np.empty(n_neurons, dtype=np.float64)
    for i in range(n_neurons):
        eta[i] = eta_stationary_std * np.random.randn()

    fpt = np.full(n_neurons, np.nan, dtype=np.float64)
    has_spiked = np.zeros(n_neurons, dtype=np.bool_)
    spiked_count = 0

    for step in range(n_steps):
        if spiked_count == n_neurons:
            break

        t_left = step * dt

        for i in range(n_neurons):
            if has_spiked[i]:
                continue

            eta[i] = decay * eta[i] + eta_step_std * np.random.randn()

            v_old = v[i]
            drift = (mu_q - v_old) / tau_q
            v_new = v_old + (drift + colored_to_voltage * eta[i]) * dt
            v[i] = v_new

            if v_new >= v_th:
                frac = 1.0
                if v_new != v_old:
                    frac = (v_th - v_old) / (v_new - v_old)
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0

                fpt[i] = refractory_period + t_left + frac * dt
                has_spiked[i] = True
                spiked_count += 1

    return fpt


def _coalesce(value, alias, name):
    if value is None and alias is None:
        raise ValueError(f"{name} must be provided")
    if value is not None and alias is not None:
        raise ValueError(f"Provide either {name} or its alias, not both")
    return alias if alias is not None else value


def simulate_first_passage_times(
    mu=None,
    sigma=None,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    dt=0.05,
    t_max=1000.0,
    n_neurons=10000,
    refractory_period=0.0,
    seed=-1,
    *,
    mu_q=None,
    sigma_q=None,
    tau_q=None,
):
    """First-passage times for the white-noise LIF model

    Parameters follow the paper convention:
    - mu: equilibrium mean of the free membrane potential
    - sigma: stationary standard deviation of the free potential
    - tau: membrane time constant

    simulated SDE is dV = (mu_q - V) / tau_q dt
    + sigma_q * sqrt(2 / tau_q) dW. Threshold crossings are linearly
    interpolated within the Euler-Maruyama step
    """
    mu_q = float(_coalesce(mu, mu_q, "mu"))
    sigma_q = float(_coalesce(sigma, sigma_q, "sigma"))
    tau_q = float(tau if tau_q is None else tau_q)

    return _simulate_white_fpt(
        mu_q,
        sigma_q,
        tau_q,
        float(v_th),
        float(v_reset),
        float(dt),
        float(t_max),
        int(n_neurons),
        float(refractory_period),
        int(seed),
    )


def simulate_colored_first_passage_times(
    mu=None,
    sigma=None,
    tau_c=1.0,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    dt=0.01,
    t_max=1000.0,
    n_neurons=10000,
    refractory_period=0.0,
    seed=-1,
    *,
    mu_q=None,
    sigma_q=None,
    tau_q=None,
):
    """First-passage times when the white noise is replaced by OU noise.

    The auxiliary colored noise eta has autocovariance
    exp(-|t| / tau_c) / (2 tau_c). With this normalization, eta approaches
    Gaussian white noise as tau_c approaches zero.
    """
    mu_q = float(_coalesce(mu, mu_q, "mu"))
    sigma_q = float(_coalesce(sigma, sigma_q, "sigma"))
    tau_q = float(tau if tau_q is None else tau_q)

    return _simulate_colored_fpt(
        mu_q,
        sigma_q,
        float(tau_c),
        tau_q,
        float(v_th),
        float(v_reset),
        float(dt),
        float(t_max),
        int(n_neurons),
        float(refractory_period),
        int(seed),
    )


def valid_first_passage_times(fpt):
    """Drop neurons that did not spike before t_max."""
    return np.asarray(fpt)[np.isfinite(fpt)]


def firing_rate_from_fpt(fpt, time_scale=1000.0):
    """Estimate firing rate from first-passage times in Hz when time is in ms."""
    valid = valid_first_passage_times(fpt)
    if valid.size == 0:
        return 0.0
    return time_scale / np.mean(valid)


def cv_from_fpt(fpt):
    """Coefficient of variation of valid first-passage times."""
    valid = valid_first_passage_times(fpt)
    if valid.size < 2:
        return np.nan
    return np.std(valid) / np.mean(valid)


def simulate_voltage_traces(
    mu=None,
    sigma=None,
    tau_c=0.0,
    tau=20.0,
    v_th=1.0,
    v_reset=0.0,
    dt=0.05,
    t_max=250.0,
    n_traces=8,
    seed=-1,
    *,
    mu_q=None,
    sigma_q=None,
    tau_q=None,
):
    """Simulate illustrative voltage traces until their first threshold crossing.
    """
    mu_q = float(_coalesce(mu, mu_q, "mu"))
    sigma_q = float(_coalesce(sigma, sigma_q, "sigma"))
    tau_q = float(tau if tau_q is None else tau_q)

    rng = np.random.default_rng(None if seed < 0 else seed)
    n_steps = int(np.ceil(t_max / dt))
    times = np.arange(n_steps + 1, dtype=float) * dt
    traces = np.full((int(n_traces), n_steps + 1), np.nan, dtype=float)
    traces[:, 0] = v_reset

    active = np.ones(int(n_traces), dtype=bool)
    v = np.full(int(n_traces), v_reset, dtype=float)

    use_colored = tau_c > 0.0
    eta = np.zeros(int(n_traces), dtype=float)
    if use_colored:
        decay = np.exp(-dt / tau_c)
        eta_step_std = np.sqrt((1.0 - decay * decay) / (2.0 * tau_c))
        eta_stationary_std = np.sqrt(1.0 / (2.0 * tau_c))
        eta = eta_stationary_std * rng.standard_normal(int(n_traces))
    else:
        noise_coeff = sigma_q * np.sqrt(2.0 / tau_q)

    for step in range(n_steps):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        v_old = v[idx].copy()
        drift = (mu_q - v_old) / tau_q

        if use_colored:
            eta[idx] = decay * eta[idx] + eta_step_std * rng.standard_normal(idx.size)
            v_new = v_old + (drift + sigma_q * np.sqrt(2.0 / tau_q) * eta[idx]) * dt
        else:
            v_new = v_old + drift * dt + noise_coeff * np.sqrt(dt) * rng.standard_normal(idx.size)

        v[idx] = v_new
        traces[idx, step + 1] = v_new

        crossed = idx[v_new >= v_th]
        if crossed.size > 0:
            traces[crossed, step + 1] = v_th
            active[crossed] = False

    return times, traces
