import numpy as np
from numba import njit

@njit
def simulate_first_passage_times(mu, sigma, tau=20.0, v_th=1.0, v_reset=0.0, dt=0.1, t_max=1000.0, n_neurons=10000):
    """
    Simula una popolazione di neuroni LIF e restituisce il tempo del primo spike per ciascuno.
    Questo equivale alla distribuzione degli Inter-Spike Intervals (ISI).
    """
    n_steps = int(t_max / dt)
    sqrt_dt = np.sqrt(dt)
    
    # potential inizialization (start from v_reset)
    v = np.full(n_neurons, v_reset, dtype=np.float64)
    
    # spike time initialization
    fpt = np.full(n_neurons, np.nan, dtype=np.float64)
    
    # track of which neurons have already spiked to avoid overwriting the time
    has_spiked = np.zeros(n_neurons, dtype=np.bool_)
    spiked_count = 0
    
    for step in range(n_steps):
        # if all neurons have spiked, we can stop the simulation early
        if spiked_count == n_neurons:
            break
            
        # white noise term for each neuron
        Z = np.random.randn(n_neurons)
        
        # Ito equation integrated with Euler-Maruyama method
        v = v + (mu - v / tau) * dt + sigma * sqrt_dt * Z
        
        # threshold check (v_th)
        current_time = step * dt
        for i in range(n_neurons):
            if not has_spiked[i] and v[i] >= v_th:
                fpt[i] = current_time
                has_spiked[i] = True
                spiked_count += 1
                
    return fpt