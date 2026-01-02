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
Photon counter interface for fluorescence detection.

Models realistic single-photon detection including:
- Detection efficiency
- Dark counts
- Dead time
- Timing jitter
- Afterpulsing

Detector Characteristics
------------------------
Typical APD (Avalanche Photodiode) parameters:
- Detection efficiency: 5-20%
- Dark count rate: 10-1000 Hz
- Dead time: 20-100 ns
- Timing jitter: 0.3-1 ns
- Afterpulsing probability: 0.1-5%

SNSPD (Superconducting Nanowire):
- Detection efficiency: 70-95%
- Dark count rate: <1 Hz
- Dead time: 10-50 ns
- Timing jitter: <100 ps
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class PhotonCounter:
    """
    Realistic photon counter for fluorescence detection.

    Parameters
    ----------
    efficiency : float
        Detection efficiency (0-1). Default: 0.1 (10%)
    dark_count_rate : float
        Dark count rate in Hz. Default: 100
    dead_time : float
        Dead time in seconds. Default: 50e-9 (50 ns)
    timing_jitter : float
        Timing jitter (std dev) in seconds. Default: 0.5e-9
    afterpulsing_prob : float
        Afterpulsing probability. Default: 0.01 (1%)

    Attributes
    ----------
    total_counts : int
        Total detected photons
    detection_times : list
        List of photon arrival times

    Examples
    --------
    >>> counter = PhotonCounter(efficiency=0.1)
    >>>
    >>> # Simulate detection from excited state population
    >>> for rho in rho_trajectory:
    ...     result = counter.count(rho, dt=1e-9, gamma=8e7)
    >>>
    >>> print(f"Total photons: {counter.total_counts}")
    """

    efficiency: float = 0.1
    dark_count_rate: float = 100.0  # Hz
    dead_time: float = 50e-9  # seconds
    timing_jitter: float = 0.5e-9  # seconds
    afterpulsing_prob: float = 0.01
    afterpulsing_delay: float = 100e-9  # seconds

    # Internal state
    _total_counts: int = field(default=0, repr=False)
    _detection_times: List[float] = field(default_factory=list, repr=False)
    _last_detection: float = field(default=-1.0, repr=False)
    _current_time: float = field(default=0.0, repr=False)

    @property
    def total_counts(self) -> int:
        """Total number of detected photons."""
        return self._total_counts

    @property
    def detection_times(self) -> List[float]:
        """List of photon detection times."""
        return self._detection_times.copy()

    def reset(self):
        """Reset all counters."""
        self._total_counts = 0
        self._detection_times.clear()
        self._last_detection = -1.0
        self._current_time = 0.0

    def count(
        self,
        rho: np.ndarray,
        dt: float,
        gamma: float,
        current_time: Optional[float] = None
    ) -> Dict:
        """
        Simulate photon detection from the current state.

        Parameters
        ----------
        rho : np.ndarray
            18×18 density matrix
        dt : float
            Time step in seconds
        gamma : float
            Spontaneous emission rate in Hz (~8×10⁷ for NV)
        current_time : float, optional
            Current time. If None, auto-incremented.

        Returns
        -------
        dict
            Detection result with keys:
            - 'detected': Number of photons detected in this step
            - 'expected': Expected number of photons
            - 'pop_excited': Excited state population
        """
        if current_time is not None:
            self._current_time = current_time

        # Calculate excited state population
        pop_excited = np.real(np.trace(rho[9:, 9:]))

        # Expected photon rate: R = γ × pop_e
        emission_rate = gamma * pop_excited

        # Expected photons in this time step
        expected = emission_rate * dt

        # Simulate Poisson process for emission
        n_emitted = np.random.poisson(expected)

        # Apply detection efficiency
        n_detected_signal = np.random.binomial(n_emitted, self.efficiency)

        # Add dark counts
        n_dark = np.random.poisson(self.dark_count_rate * dt)

        # Total potential detections
        n_potential = n_detected_signal + n_dark

        # Apply dead time
        n_detected = 0
        for _ in range(n_potential):
            if self._current_time - self._last_detection > self.dead_time:
                # Detection accepted
                t_detect = self._current_time

                # Add timing jitter
                if self.timing_jitter > 0:
                    t_detect += np.random.normal(0, self.timing_jitter)

                self._detection_times.append(t_detect)
                self._last_detection = self._current_time
                n_detected += 1

                # Possible afterpulse
                if np.random.random() < self.afterpulsing_prob:
                    t_after = t_detect + np.random.exponential(self.afterpulsing_delay)
                    self._detection_times.append(t_after)
                    n_detected += 1

        self._total_counts += n_detected
        self._current_time += dt

        return {
            'detected': n_detected,
            'expected': expected,
            'pop_excited': pop_excited,
        }

    def count_rate(self, window: float = 1e-6) -> float:
        """
        Calculate count rate over a time window.

        Parameters
        ----------
        window : float
            Time window in seconds

        Returns
        -------
        float
            Count rate in Hz
        """
        if len(self._detection_times) == 0:
            return 0.0

        # Count detections in the last window
        t_end = max(self._detection_times)
        t_start = t_end - window

        n_counts = sum(1 for t in self._detection_times if t >= t_start)
        return n_counts / window

    def histogram(
        self,
        bin_width: float = 1e-9,
        t_range: Optional[tuple] = None
    ) -> tuple:
        """
        Create histogram of detection times.

        Parameters
        ----------
        bin_width : float
            Bin width in seconds
        t_range : tuple, optional
            (t_min, t_max) time range

        Returns
        -------
        tuple
            (counts, bin_edges)
        """
        if len(self._detection_times) == 0:
            return np.array([]), np.array([])

        times = np.array(self._detection_times)

        if t_range is None:
            t_range = (times.min(), times.max())

        n_bins = int((t_range[1] - t_range[0]) / bin_width)
        counts, edges = np.histogram(times, bins=n_bins, range=t_range)

        return counts, edges

    def calculate_g2(
        self,
        tau_max: float = 1e-6,
        n_bins: int = 100
    ) -> tuple:
        """
        Calculate g²(τ) autocorrelation function.

        Parameters
        ----------
        tau_max : float
            Maximum delay time in seconds
        n_bins : int
            Number of delay bins

        Returns
        -------
        tuple
            (tau, g2) arrays

        Notes
        -----
        g²(0) < 1 indicates antibunching (single photon source)
        g²(0) > 1 indicates bunching (thermal light)
        g²(0) = 1 for coherent light
        """
        if len(self._detection_times) < 2:
            return np.array([]), np.array([])

        times = np.array(sorted(self._detection_times))

        # Calculate coincidences
        tau_bins = np.linspace(-tau_max, tau_max, n_bins)
        coincidences = np.zeros(n_bins - 1)

        for i, t1 in enumerate(times[:-1]):
            delays = times[i+1:] - t1
            delays = delays[delays <= tau_max]
            hist, _ = np.histogram(delays, bins=tau_bins[n_bins//2:])
            coincidences[n_bins//2:-1] += hist

        # Normalize
        bin_width = tau_bins[1] - tau_bins[0]
        T = times[-1] - times[0]
        n = len(times)

        if T > 0 and n > 1:
            rate = n / T
            norm = rate ** 2 * bin_width * T
            g2 = coincidences / (norm + 1e-10)
        else:
            g2 = coincidences

        tau = (tau_bins[:-1] + tau_bins[1:]) / 2

        return tau, g2

    def fano_factor(self, window: float = 1e-6) -> float:
        """
        Calculate Fano factor F = Var(n)/⟨n⟩.

        Parameters
        ----------
        window : float
            Counting window in seconds

        Returns
        -------
        float
            Fano factor (F<1: sub-Poissonian, F=1: Poissonian, F>1: super-Poissonian)
        """
        if len(self._detection_times) < 10:
            return 1.0

        times = np.array(sorted(self._detection_times))
        t_max = times[-1]

        # Count in each window
        n_windows = int(t_max / window)
        counts = []

        for i in range(n_windows):
            t_start = i * window
            t_end = (i + 1) * window
            n = np.sum((times >= t_start) & (times < t_end))
            counts.append(n)

        counts = np.array(counts)
        mean = np.mean(counts)
        var = np.var(counts)

        if mean > 0:
            return var / mean
        return 1.0

    def __repr__(self) -> str:
        return f"PhotonCounter(counts={self._total_counts}, efficiency={self.efficiency})"
