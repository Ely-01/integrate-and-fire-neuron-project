import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

def siegert_firing_rate(mu, sigma, tau=20.0, v_th=1.0, v_reset=0.0):
    """
    Theoretical firing rate for a stochastic LIF neuron (Siegert formula) 
    """
    if sigma == 0.0:
        if mu > v_th / tau:
            mean_fpt = -tau * np.log(1.0 - v_th / (mu * tau))
            return 1000.0 / mean_fpt
        else:
            return 0.0

    noise_scale = sigma * np.sqrt(tau)
    
    y_th = (v_th - mu * tau) / noise_scale
    y_res = (v_reset - mu * tau) / noise_scale

    # use of erfcx(-x) to avoid overflow (inf * 0)
    def integrand(x):
        return erfcx(-x)

    integral_value, _ = quad(integrand, y_res, y_th, limit=100)

    mean_fpt = tau * np.sqrt(np.pi) * integral_value
    return 1000.0 / mean_fpt