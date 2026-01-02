# ==============================================================================
#  QUSIM - Quantum Simulator for NV Centers
#  Leon Kaiser, MSQC Goethe University, Frankfurt, Germany
#  https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#
#  This software is provided for scientific and educational purposes.
#  Free to use, modify, and distribute with attribution.
# ==============================================================================
"""
Arbitrary Waveform Generator (AWG) interface.

The AWG generates time-dependent microwave waveforms for spin control.
It stores amplitude Ω(t), phase φ(t), and detuning Δ(t) as arrays
and provides interpolation for arbitrary time points.

Waveform Structure
------------------
The MW field in the rotating frame is characterized by:
- Amplitude (Rabi frequency) Ω(t) in MHz
- Phase φ(t) in radians
- Detuning Δ(t) in MHz (optional)

The resulting Hamiltonian is:
    H_MW = (Ω/2)(S₊ e^(-iφ) + S₋ e^(iφ)) + Δ Sz

Pulse Shapes
------------
Supported envelope shapes:
- rect: Rectangular (constant amplitude)
- gauss: Gaussian envelope
- sinc: Sinc function (sin(x)/x)
- blackman: Blackman window
- hermite: Hermite-Gaussian
- drag: DRAG pulse for leakage reduction
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from ..core.constants import MHZ


@dataclass
class AWGInterface:
    """
    Arbitrary Waveform Generator for microwave pulse control.

    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz. Default: 1e9 (1 GS/s)

    Attributes
    ----------
    sample_rate : float
        AWG sample rate
    dt : float
        Time step between samples
    duration : float
        Total waveform duration

    Examples
    --------
    >>> awg = AWGInterface()
    >>> awg.add_pulse("gauss", amplitude=10, duration=100e-9, phase=0)
    >>> awg.add_delay(500e-9)
    >>> awg.add_pulse("gauss", amplitude=10, duration=100e-9, phase=np.pi)
    >>>
    >>> # Get Rabi frequency at t=50 ns
    >>> omega = awg.get_omega(50e-9)
    """

    sample_rate: float = 1e9  # 1 GS/s
    _amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    _phase: np.ndarray = field(default_factory=lambda: np.array([]))
    _detuning: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.dt = 1.0 / self.sample_rate
        self.duration = 0.0

    def clear(self):
        """Clear all waveform data."""
        self._amplitude = np.array([])
        self._phase = np.array([])
        self._detuning = np.array([])
        self.duration = 0.0

    def add_pulse(
        self,
        shape: str,
        amplitude: float,      # MHz (Rabi frequency)
        duration: float,       # seconds
        phase: float = 0.0,    # radians
        detuning: float = 0.0, # MHz
        **kwargs
    ):
        """
        Add a pulse to the waveform.

        Parameters
        ----------
        shape : str
            Pulse shape: 'rect', 'gauss', 'sinc', 'blackman', 'hermite', 'drag'
        amplitude : float
            Peak Rabi frequency in MHz
        duration : float
            Pulse duration in seconds
        phase : float
            Pulse phase in radians. Default: 0
        detuning : float
            Frequency detuning in MHz. Default: 0
        **kwargs
            Shape-specific parameters:
            - sigma: Gaussian width (default: duration/6)
            - beta: DRAG coefficient (default: 0.5)
            - order: Hermite order (default: 0)
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt

        # Generate envelope
        envelope = self._generate_envelope(shape, t, duration, **kwargs)

        # Scale by amplitude
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        detuning_array = np.full(n_samples, detuning)

        # Append to waveform
        self._amplitude = np.concatenate([self._amplitude, amp_array])
        self._phase = np.concatenate([self._phase, phase_array])
        self._detuning = np.concatenate([self._detuning, detuning_array])
        self.duration += duration

    def add_delay(self, duration: float):
        """
        Add a delay (zero amplitude) segment.

        Parameters
        ----------
        duration : float
            Delay duration in seconds
        """
        n_samples = int(duration * self.sample_rate)

        self._amplitude = np.concatenate([
            self._amplitude, np.zeros(n_samples)
        ])
        self._phase = np.concatenate([
            self._phase, np.zeros(n_samples)
        ])
        self._detuning = np.concatenate([
            self._detuning, np.zeros(n_samples)
        ])
        self.duration += duration

    def _generate_envelope(
        self,
        shape: str,
        t: np.ndarray,
        duration: float,
        **kwargs
    ) -> np.ndarray:
        """Generate pulse envelope."""
        if shape == "rect":
            return np.ones_like(t)

        elif shape == "gauss":
            sigma = kwargs.get("sigma", duration / 6)
            t0 = duration / 2
            return np.exp(-((t - t0) / sigma) ** 2)

        elif shape == "sinc":
            zero_crossings = kwargs.get("zero_crossings", 3)
            t0 = duration / 2
            x = zero_crossings * np.pi * (t - t0) / (duration / 2)
            env = np.sinc(x / np.pi)  # np.sinc includes the pi
            return env

        elif shape == "blackman":
            n = len(t)
            return 0.42 - 0.5 * np.cos(2 * np.pi * t / duration) + \
                   0.08 * np.cos(4 * np.pi * t / duration)

        elif shape == "hermite":
            order = kwargs.get("order", 0)
            sigma = kwargs.get("sigma", duration / 6)
            t0 = duration / 2
            x = (t - t0) / sigma

            if order == 0:
                return np.exp(-x**2 / 2)
            elif order == 1:
                return x * np.exp(-x**2 / 2)
            elif order == 2:
                return (x**2 - 1) * np.exp(-x**2 / 2)
            else:
                # General Hermite polynomial
                from scipy.special import hermite
                H = hermite(order)
                return H(x) * np.exp(-x**2 / 2)

        elif shape == "drag":
            sigma = kwargs.get("sigma", duration / 6)
            beta = kwargs.get("beta", 0.5)
            t0 = duration / 2
            x = (t - t0) / sigma

            # DRAG: I channel is Gaussian, Q channel is derivative
            # Here we return amplitude; phase handles the quadrature
            gauss = np.exp(-x**2 / 2)
            derivative = -x * gauss / sigma
            return gauss  # Phase modulation would be added separately

        else:
            raise ValueError(f"Unknown pulse shape: {shape}")

    def get_omega(self, t: float) -> float:
        """
        Get Rabi frequency at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Rabi frequency in MHz
        """
        if len(self._amplitude) == 0:
            return 0.0

        idx = int(t * self.sample_rate)
        if idx < 0 or idx >= len(self._amplitude):
            return 0.0

        return self._amplitude[idx]

    def get_phase(self, t: float) -> float:
        """
        Get phase at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Phase in radians
        """
        if len(self._phase) == 0:
            return 0.0

        idx = int(t * self.sample_rate)
        if idx < 0 or idx >= len(self._phase):
            return 0.0

        return self._phase[idx]

    def get_detuning(self, t: float) -> float:
        """
        Get detuning at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Detuning in MHz
        """
        if len(self._detuning) == 0:
            return 0.0

        idx = int(t * self.sample_rate)
        if idx < 0 or idx >= len(self._detuning):
            return 0.0

        return self._detuning[idx]

    def to_mw_drive(self):
        """
        Create a MicrowaveDrive term from this AWG waveform.

        Returns
        -------
        MicrowaveDrive
            Hamiltonian term with this waveform
        """
        from ..hamiltonian.terms import MicrowaveDrive

        return MicrowaveDrive(
            omega=self.get_omega,
            phase=self.get_phase,
            detuning=0  # Detuning handled separately
        )

    def plot(self, ax=None):
        """
        Plot the waveform.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates new figure.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        t = np.arange(len(self._amplitude)) * self.dt * 1e9  # in ns

        ax.plot(t, self._amplitude, label='Amplitude (MHz)')
        ax.plot(t, self._phase, label='Phase (rad)')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title('AWG Waveform')

        return ax

    def __repr__(self) -> str:
        n_samples = len(self._amplitude)
        return f"AWGInterface(duration={self.duration*1e9:.1f} ns, samples={n_samples})"
