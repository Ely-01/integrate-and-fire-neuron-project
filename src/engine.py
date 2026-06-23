import numpy as np


def simulate_white_spike_times(
    mu_q,
    sigma_q,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    dt=0.05,
    t_max=1000.0,
    n_neurons=10000,
    refractory_ms=0.0,
    seed=-1,
):
    """Spike times for Burkitt's white-noise LIF model.

    The simulated equation is
    dV = (mu_Q - V) / tau_Q dt + sigma_Q sqrt(2 / tau_Q) dW.
    """
    rng = np.random.default_rng(None if seed < 0 else seed)
    step_count = int(np.ceil(t_max / dt))
    sqrt_dt = np.sqrt(dt)
    noise_scale = sigma_q * np.sqrt(2.0 / tau_q)

    voltage = np.full(int(n_neurons), reset, dtype=float)
    spike_times = np.full(int(n_neurons), np.nan, dtype=float)
    active = np.arange(int(n_neurons))

    for step in range(step_count):
        if active.size == 0:
            break

        step_start_time = step * dt
        old_voltage = voltage[active]
        new_voltage = old_voltage + (mu_q - old_voltage) / tau_q * dt
        if sigma_q > 0.0:
            new_voltage = new_voltage + noise_scale * sqrt_dt * rng.standard_normal(active.size)
        voltage[active] = new_voltage

        crossed = new_voltage >= threshold
        if np.any(crossed):
            crossed_neurons = active[crossed]
            crossing_fraction = threshold_crossing_fraction(old_voltage[crossed], new_voltage[crossed], threshold)
            spike_times[crossed_neurons] = refractory_ms + step_start_time + crossing_fraction * dt
            active = active[~crossed]

    return spike_times


def simulate_colored_spike_times(
    mu_q,
    sigma_q,
    tau_c,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    dt=0.05,
    t_max=1000.0,
    n_neurons=10000,
    refractory_ms=0.0,
    seed=-1,
):
    """Spike times when the white-noise term is replaced by OU colored noise."""
    if tau_c <= 0.0:
        return simulate_white_spike_times(
            mu_q=mu_q,
            sigma_q=sigma_q,
            tau_q=tau_q,
            threshold=threshold,
            reset=reset,
            dt=dt,
            t_max=t_max,
            n_neurons=n_neurons,
            refractory_ms=refractory_ms,
            seed=seed,
        )

    rng = np.random.default_rng(None if seed < 0 else seed)
    step_count = int(np.ceil(t_max / dt))
    voltage_noise_scale = sigma_q * np.sqrt(2.0 / tau_q)

    noise_decay = np.exp(-dt / tau_c)
    noise_step_std = np.sqrt((1.0 - noise_decay * noise_decay) / (2.0 * tau_c))
    noise_stationary_std = np.sqrt(1.0 / (2.0 * tau_c))

    voltage = np.full(int(n_neurons), reset, dtype=float)
    colored_noise = noise_stationary_std * rng.standard_normal(int(n_neurons))
    spike_times = np.full(int(n_neurons), np.nan, dtype=float)
    active = np.arange(int(n_neurons))

    for step in range(step_count):
        if active.size == 0:
            break

        step_start_time = step * dt
        colored_noise[active] = noise_decay * colored_noise[active] + noise_step_std * rng.standard_normal(active.size)

        old_voltage = voltage[active]
        drift = (mu_q - old_voltage) / tau_q
        new_voltage = old_voltage + (drift + voltage_noise_scale * colored_noise[active]) * dt
        voltage[active] = new_voltage

        crossed = new_voltage >= threshold
        if np.any(crossed):
            crossed_neurons = active[crossed]
            crossing_fraction = threshold_crossing_fraction(old_voltage[crossed], new_voltage[crossed], threshold)
            spike_times[crossed_neurons] = refractory_ms + step_start_time + crossing_fraction * dt
            active = active[~crossed]

    return spike_times


def threshold_crossing_fraction(old_voltage, new_voltage, threshold):
    """Linear interpolation of the threshold crossing inside one Euler step."""
    voltage_change = new_voltage - old_voltage
    fraction = np.ones_like(new_voltage, dtype=float)
    nonzero = voltage_change != 0.0
    fraction[nonzero] = (threshold - old_voltage[nonzero]) / voltage_change[nonzero]
    return np.clip(fraction, 0.0, 1.0)


def finite_spike_times(spike_times):
    """Return only neurons that crossed threshold before t_max."""
    return np.asarray(spike_times)[np.isfinite(spike_times)]


def firing_rate_hz(spike_times):
    """Estimate firing rate in Hz from spike times measured in ms."""
    valid_times = finite_spike_times(spike_times)
    if valid_times.size == 0:
        return 0.0
    return 1000.0 / np.mean(valid_times)


def coefficient_of_variation(spike_times):
    """Coefficient of variation of valid spike times."""
    valid_times = finite_spike_times(spike_times)
    if valid_times.size < 2:
        return np.nan
    return np.std(valid_times) / np.mean(valid_times)


def simulate_voltage_traces(
    mu_q,
    sigma_q,
    tau_c=0.0,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    dt=0.05,
    t_max=250.0,
    trace_count=8,
    seed=-1,
):
    """Example voltage traces, stopped after the first threshold crossing."""
    rng = np.random.default_rng(None if seed < 0 else seed)
    step_count = int(np.ceil(t_max / dt))
    times = np.arange(step_count + 1, dtype=float) * dt
    traces = np.full((int(trace_count), step_count + 1), np.nan, dtype=float)
    traces[:, 0] = reset

    voltage = np.full(int(trace_count), reset, dtype=float)
    active = np.ones(int(trace_count), dtype=bool)

    if tau_c > 0.0:
        noise_decay = np.exp(-dt / tau_c)
        noise_step_std = np.sqrt((1.0 - noise_decay * noise_decay) / (2.0 * tau_c))
        colored_noise = np.sqrt(1.0 / (2.0 * tau_c)) * rng.standard_normal(int(trace_count))
    else:
        white_noise_step = sigma_q * np.sqrt(2.0 / tau_q) * np.sqrt(dt)

    for step in range(step_count):
        if not np.any(active):
            break

        active_indices = np.where(active)[0]
        old_voltage = voltage[active_indices].copy()
        drift = (mu_q - old_voltage) / tau_q

        if tau_c > 0.0:
            colored_noise[active_indices] = (
                noise_decay * colored_noise[active_indices] + noise_step_std * rng.standard_normal(active_indices.size)
            )
            new_voltage = old_voltage + (drift + sigma_q * np.sqrt(2.0 / tau_q) * colored_noise[active_indices]) * dt
        else:
            new_voltage = old_voltage + drift * dt + white_noise_step * rng.standard_normal(active_indices.size)

        voltage[active_indices] = new_voltage
        traces[active_indices, step + 1] = new_voltage

        crossed = new_voltage >= threshold
        if np.any(crossed):
            crossed_indices = active_indices[crossed]
            traces[crossed_indices, step + 1] = threshold
            active[crossed_indices] = False

    return times, traces


def simulate_spike_trains(
    mu_q,
    sigma_q,
    tau_c=0.0,
    tau_q=20.0,
    threshold=1.0,
    reset=0.0,
    dt=0.1,
    duration_ms=60000.0,
    trial_count=20,
    burn_in_ms=1000.0,
    refractory_ms=0.0,
    seed=-1,
):
    """Continuous spike trains with voltage reset and persistent input noise.

    For colored noise, only the voltage is reset after a spike. The OU input
    continues evolving, so temporal memory can pass from one ISI to the next.
    """
    rng = np.random.default_rng(None if seed < 0 else seed)
    trial_count = int(trial_count)
    total_steps = int(np.ceil((burn_in_ms + duration_ms) / dt))
    burn_in_steps = int(np.ceil(burn_in_ms / dt))

    voltage = np.full(trial_count, reset, dtype=float)
    refractory_steps = np.zeros(trial_count, dtype=int)
    spike_trains = [[] for _ in range(trial_count)]
    voltage_noise_scale = sigma_q * np.sqrt(2.0 / tau_q)

    if tau_c > 0.0:
        noise_decay = np.exp(-dt / tau_c)
        noise_step_std = np.sqrt((1.0 - noise_decay * noise_decay) / (2.0 * tau_c))
        colored_noise = np.sqrt(1.0 / (2.0 * tau_c)) * rng.standard_normal(trial_count)
    else:
        white_noise_step = voltage_noise_scale * np.sqrt(dt)

    refractory_step_count = int(np.ceil(refractory_ms / dt))

    for step in range(total_steps):
        if tau_c > 0.0:
            colored_noise = noise_decay * colored_noise + noise_step_std * rng.standard_normal(trial_count)

        refractory = refractory_steps > 0
        refractory_steps[refractory] -= 1
        voltage[refractory] = reset
        active = ~refractory
        if not np.any(active):
            continue

        active_indices = np.flatnonzero(active)
        old_voltage = voltage[active_indices].copy()
        drift = (mu_q - old_voltage) / tau_q

        if tau_c > 0.0:
            new_voltage = old_voltage + (
                drift + voltage_noise_scale * colored_noise[active_indices]
            ) * dt
        else:
            new_voltage = old_voltage + drift * dt
            new_voltage += white_noise_step * rng.standard_normal(active_indices.size)

        voltage[active_indices] = new_voltage
        crossed = new_voltage >= threshold
        if not np.any(crossed):
            continue

        crossed_indices = active_indices[crossed]
        crossing_fraction = threshold_crossing_fraction(
            old_voltage[crossed], new_voltage[crossed], threshold
        )
        crossing_times = (step + crossing_fraction) * dt - burn_in_ms

        for neuron, crossing_time in zip(crossed_indices, crossing_times):
            if step >= burn_in_steps:
                spike_trains[neuron].append(float(crossing_time))

        voltage[crossed_indices] = reset
        refractory_steps[crossed_indices] = refractory_step_count

    return [np.asarray(train, dtype=float) for train in spike_trains]
