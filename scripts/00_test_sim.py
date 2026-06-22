import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine import simulate_first_passage_times

print("Starting simulation")
start_time = time.time()

# Noisy regime
mu_test = 1.1
sigma_test = 0.2
n_pop = 20000


fpt_results = simulate_first_passage_times(
    mu=mu_test, sigma=sigma_test, n_neurons=n_pop, seed=42
)

# filtering out neurons that never spiked (NaN values)
fpt_valid = fpt_results[~np.isnan(fpt_results)]

print(f"Simulation completed in {time.time() - start_time:.3f} seconds.")
print(f"Neurons that spiked: {len(fpt_valid)} out of {n_pop}")

if len(fpt_valid) > 0:
    # The firing rate is the inverse of the mean first-passage time
    mean_fpt = np.mean(fpt_valid)
    firing_rate = 1000.0 / mean_fpt  # Convert to Hz (1/s) from ms
    
    print(f"\nResults:")
    print(f"- Mean spike time: {mean_fpt:.2f} ms")
    print(f"- Estimated Firing Rate: {firing_rate:.2f} Hz")

    # distribution plot (Probability Density of First-Passage Time)
    plt.figure(figsize=(8, 5))
    plt.hist(fpt_valid, bins=80, density=True, color='teal', edgecolor='black', alpha=0.7)
    plt.title(rf"Distribution of Spike Times ($\mu_Q$={mu_test}, $\sigma_Q$={sigma_test})", fontsize=14)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "00_test_sim_fpt_density.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"- Figure saved to: {out_path}")