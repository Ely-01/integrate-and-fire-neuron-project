import numpy as np
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import finite_spike_times, firing_rate_hz, simulate_white_spike_times

print("Compilazione del kernel e avvio simulazione...")
start_time = time.time()

mu_q = 1.1
sigma_q = 0.2
n_neurons = 20000


spike_times = simulate_white_spike_times(
    mu_q=mu_q,
    sigma_q=sigma_q,
    n_neurons=n_neurons,
    seed=42,
)

valid_times = finite_spike_times(spike_times)

print(f"Simulation completed in {time.time() - start_time:.3f} seconds.")
print(f"Neurons that spiked: {len(valid_times)} out of {n_neurons}")

if len(valid_times) > 0:
    mean_spike_time = np.mean(valid_times)
    rate_hz = firing_rate_hz(spike_times)

    print(f"\nResults:")
    print(f"- Mean spike time: {mean_spike_time:.2f} ms")
    print(f"- Estimated firing rate: {rate_hz:.2f} Hz")
