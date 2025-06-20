import numpy as np
from qutip import mesolve, sigmax, sigmaz, basis


def simulate_mw_pulse_with_noise(Ω0, phase_noise, amp_noise, duration, dt, nu, T1, T2):
    """Simulate a microwave pulse with amplitude and phase noise."""
    rho0 = basis(2, 0) * basis(2, 0).dag()

    def H_t(t, args):
        Ω = Ω0 * (1 + amp_noise(t))
        φ = phase_noise(t)
        return Ω * np.cos(2 * np.pi * nu * t + φ) * sigmax()

    L = [np.sqrt(1 / T2) * sigmaz(), np.sqrt(1 / T1) * sigmax()]

    times = np.arange(0, duration, dt)
    result = mesolve(H_t, rho0, times, L, [])
    return result
