# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
from typing import Dict, List, Tuple

class PhotonCounter:
    def __init__(self, detection_efficiency: float = 0.1, dark_count_rate: float = 100.0, 
                 dead_time: float = 50e-9, time_resolution: float = 0.5e-9,
                 afterpulsing_probability: float = 0.01, afterpulsing_delay: float = 100e-9):
        """
        Realistischer Photonenzähler für NV-Zentren
        
        Args:
            detection_efficiency: Detektionseffizienz (0-1)
            dark_count_rate: Dunkelzählrate in Hz
            dead_time: Totzeit in Sekunden (typisch 50ns für APD)
            time_resolution: Zeitauflösung/Jitter in Sekunden
            afterpulsing_probability: Wahrscheinlichkeit für Afterpulse (0-1)
            afterpulsing_delay: Mittlere Verzögerung für Afterpulse in Sekunden
        """
        self.detection_efficiency = detection_efficiency
        self.dark_count_rate = dark_count_rate
        self.dead_time = dead_time
        self.time_resolution = time_resolution
        self.afterpulsing_probability = afterpulsing_probability
        self.afterpulsing_delay = afterpulsing_delay
        
        # Statistiken
        self.total_photons = 0
        self.emission_events = []
        self.count_history = []
        self.photon_arrival_times = []  # Präzise Ankunftszeiten für g²(τ)
        
        # Totzeit-Tracking
        self.last_detection_time = -np.inf
        self.dead_time_losses = 0
        self.afterpulse_buffer = []  # [(time, is_afterpulse), ...]
        
    def count_photons(self, rho: np.ndarray, dt: float, gamma_emission: float, 
                     current_time: float = None) -> Dict:
        """
        Zählt emittierte Photonen mit realistischen Detektoreffekten
        
        Die Photonenrate ergibt sich aus:
        R = γ * Tr(L†L * ρ) = γ * ⟨e|ρ|e⟩
        
        Args:
            rho: 18×18 Dichtematrix
            dt: Zeitschritt in Sekunden
            gamma_emission: Spontane Emissionsrate in Hz
            current_time: Aktuelle Simulationszeit
            
        Returns:
            Dict mit Zählergebnissen
        """
        if current_time is None:
            current_time = len(self.emission_events) * dt
            
        # Berechne Population im angeregten Zustand
        # |e⟩ Zustände sind Indizes 9-17 (zweiter 9×9 Block)
        pop_excited = np.real(np.trace(rho[9:, 9:]))
        
        # Photonenrate = γ * Population im angeregten Zustand
        emission_rate = gamma_emission * pop_excited
        
        # Erwartete Photonen in diesem Zeitschritt
        expected_photons = emission_rate * dt
        
        # Erzeuge rohe Photonenereignisse (ohne Effizienz)
        raw_photons = np.random.poisson(expected_photons)
        
        # Anwende Detektionseffizienz
        if raw_photons > 0:
            detected_signal = np.random.binomial(raw_photons, self.detection_efficiency)
        else:
            detected_signal = 0
            
        # Dunkelzählungen
        dark_counts = np.random.poisson(self.dark_count_rate * dt)
        
        # Generiere individuelle Photonen-Ankunftszeiten
        total_events = detected_signal + dark_counts
        photon_times = []
        
        if total_events > 0:
            # Zufällige Ankunftszeiten innerhalb des Zeitschritts
            arrival_times = current_time + np.random.uniform(0, dt, total_events)
            
            # Sortiere chronologisch
            arrival_times = np.sort(arrival_times)
            
            # Anwende Totzeit-Filter
            filtered_times = self._apply_dead_time_filter(arrival_times, current_time)
            
            # Anwende Timing-Jitter
            jittered_times = self._apply_timing_jitter(filtered_times)
            
            # Generiere Afterpulse
            afterpulse_times = self._generate_afterpulses(filtered_times, current_time)
            
            # Kombiniere alle Events
            photon_times = list(jittered_times) + list(afterpulse_times)
            photon_times.sort()
            
        # Aktualisiere Statistiken
        final_count = len(photon_times)
        self.total_photons += final_count
        self.photon_arrival_times.extend(photon_times)
        
        self.emission_events.append({
            'time': current_time,
            'rate': emission_rate,
            'detected': final_count,
            'lost_to_dead_time': total_events - len(photon_times) + len([t for t in photon_times if t >= current_time + dt]),
            'afterpulses': len([t for t in photon_times if hasattr(t, '_is_afterpulse')])
        })
        
        return {
            'emission_rate': emission_rate,
            'expected_photons': expected_photons,
            'raw_photons': raw_photons,
            'detected_photons': final_count,
            'dark_counts': dark_counts,
            'dead_time_losses': self.dead_time_losses,
            'total_count': self.total_photons,
            'photon_times': photon_times
        }
    
    def _apply_dead_time_filter(self, arrival_times: np.ndarray, current_time: float) -> np.ndarray:
        """
        Filtert Photonen basierend auf Totzeit des Detektors
        
        Args:
            arrival_times: Array von Ankunftszeiten
            current_time: Aktuelle Zeit
            
        Returns:
            Gefilterte Ankunftszeiten
        """
        if len(arrival_times) == 0:
            return arrival_times
            
        filtered_times = []
        
        for t in arrival_times:
            # Prüfe ob genug Zeit seit letzter Detektion vergangen
            if t - self.last_detection_time >= self.dead_time:
                filtered_times.append(t)
                self.last_detection_time = t
            else:
                self.dead_time_losses += 1
                
        return np.array(filtered_times)
    
    def _apply_timing_jitter(self, times: np.ndarray) -> np.ndarray:
        """
        Fügt Timing-Jitter zu Photonen-Ankunftszeiten hinzu
        
        Args:
            times: Array von exakten Ankunftszeiten
            
        Returns:
            Zeiten mit Jitter
        """
        if len(times) == 0 or self.time_resolution <= 0:
            return times
            
        # Gauss'scher Jitter
        jitter = np.random.normal(0, self.time_resolution, len(times))
        jittered_times = times + jitter
        
        # Verhindere negative Zeiten
        jittered_times = np.maximum(jittered_times, 0)
        
        return jittered_times
    
    def _generate_afterpulses(self, trigger_times: np.ndarray, current_time: float) -> np.ndarray:
        """
        Generiert Afterpulse basierend auf Hauptpulsen
        
        Args:
            trigger_times: Zeiten der Hauptpulse
            current_time: Aktuelle Zeit
            
        Returns:
            Afterpuls-Zeiten
        """
        afterpulse_times = []
        
        for t in trigger_times:
            # Entscheide ob Afterpuls auftritt
            if np.random.random() < self.afterpulsing_probability:
                # Exponentiell verteilte Verzögerung
                delay = np.random.exponential(self.afterpulsing_delay)
                afterpulse_time = t + delay
                
                # Afterpuls muss auch Totzeit respektieren
                if afterpulse_time - self.last_detection_time >= self.dead_time:
                    afterpulse_times.append(afterpulse_time)
                    # Markiere als Afterpuls für Statistiken
                    setattr(afterpulse_time, '_is_afterpulse', True)
                    
        return np.array(afterpulse_times)
    
    def get_correlation_g2(self, tau: float, window: float = 1e-9, 
                          max_pairs: int = 10000) -> float:
        """
        Berechnet g²(τ) Korrelationsfunktion basierend auf echten Photonen-Ankunftszeiten
        
        g²(τ) = ⟨I(t)I(t+τ)⟩ / (⟨I(t)⟩²)
        
        Args:
            tau: Verzögerungszeit in Sekunden
            window: Toleranzfenster um τ 
            max_pairs: Maximale Anzahl Photon-Paare für Berechnung
            
        Returns:
            g²(τ) Wert
        """
        if len(self.photon_arrival_times) < 2:
            return 0.0 if tau == 0 else 1.0
            
        times = np.array(self.photon_arrival_times)
        
        # Zähle Photonenpaare mit Verzögerung τ ± window/2
        pairs_in_window = 0
        total_pairs = 0
        
        for i, t1 in enumerate(times):
            if total_pairs >= max_pairs:
                break
                
            for j, t2 in enumerate(times[i+1:], i+1):
                time_diff = t2 - t1
                
                # Nur vorwärts in der Zeit
                if abs(time_diff - abs(tau)) <= window/2:
                    pairs_in_window += 1
                    
                total_pairs += 1
                if total_pairs >= max_pairs:
                    break
        
        if total_pairs == 0:
            return 0.0 if tau == 0 else 1.0
            
        # Normalisierung für g²(τ)
        observation_time = times[-1] - times[0] if len(times) > 1 else 1.0
        count_rate = len(times) / observation_time
        
        # Erwartete Paare für unkorrelierten Strahl
        expected_pairs = count_rate**2 * observation_time * window
        
        if expected_pairs == 0:
            return 0.0 if tau == 0 else 1.0
            
        g2_value = pairs_in_window / expected_pairs
        
        return max(0.0, g2_value)  # Verhindere negative Werte
    
    def calculate_g2_curve(self, tau_max: float = 1e-6, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet komplette g²(τ) Kurve
        
        Args:
            tau_max: Maximale Verzögerungszeit
            n_points: Anzahl Punkte
            
        Returns:
            (tau_array, g2_array)
        """
        tau_array = np.linspace(0, tau_max, n_points)
        g2_array = np.array([self.get_correlation_g2(tau) for tau in tau_array])
        
        return tau_array, g2_array
    
    def get_photon_statistics(self) -> Dict:
        """
        Berechnet erweiterte Photonen-Statistiken
        
        Returns:
            Dict mit statistischen Kenngrößen
        """
        if not self.photon_arrival_times:
            return {}
            
        times = np.array(self.photon_arrival_times)
        
        # Zeitaufgelöste Zählrate
        if len(times) > 1:
            observation_time = times[-1] - times[0]
            count_rate = len(times) / observation_time
            
            # Inter-Photon-Zeiten
            inter_times = np.diff(times)
            mean_inter_time = np.mean(inter_times)
            std_inter_time = np.std(inter_times)
            
        else:
            count_rate = 0.0
            mean_inter_time = np.inf
            std_inter_time = 0.0
            
        # Antibunching-Parameter (g²(0))
        g2_zero = self.get_correlation_g2(0.0, window=self.time_resolution)
        
        return {
            'count_rate_hz': count_rate,
            'total_photons': len(times),
            'mean_inter_photon_time': mean_inter_time,
            'std_inter_photon_time': std_inter_time,
            'g2_zero': g2_zero,
            'antibunching_quality': 1.0 - g2_zero,  # 1 = perfektes Antibunching
            'observation_time': observation_time if len(times) > 1 else 0.0
        }
    
    def get_statistics(self) -> Dict:
        """Gibt erweiterte Zählstatistiken zurück"""
        if not self.emission_events:
            return {
                'total_photons': 0,
                'average_rate': 0.0,
                'fano_factor': 0.0,
                'dead_time_losses': 0,
                'detection_efficiency': self.detection_efficiency
            }
        
        rates = [e['rate'] for e in self.emission_events]
        counts = [e['detected'] for e in self.emission_events]
        
        avg_rate = np.mean(rates)
        avg_count = np.mean(counts)
        var_count = np.var(counts)
        
        # Fano-Faktor F = σ²/μ (sollte ~1 für Poisson sein)
        fano_factor = var_count / avg_count if avg_count > 0 else 0.0
        
        # Totzeit-Statistiken
        total_lost = sum(e.get('lost_to_dead_time', 0) for e in self.emission_events)
        total_afterpulses = sum(e.get('afterpulses', 0) for e in self.emission_events)
        
        # Zeitaufgelöste Statistiken
        observation_time = self.emission_events[-1]['time'] - self.emission_events[0]['time'] if len(self.emission_events) > 1 else 1.0
        
        base_stats = {
            'total_photons': self.total_photons,
            'average_rate': avg_rate,
            'average_count_rate': avg_count / observation_time,
            'fano_factor': fano_factor,
            'detection_efficiency': self.detection_efficiency,
            'dead_time_ns': self.dead_time * 1e9,
            'dead_time_losses': total_lost,
            'afterpulses': total_afterpulses,
            'time_resolution_ps': self.time_resolution * 1e12,
            'observation_time_us': observation_time * 1e6
        }
        
        # Füge Photonen-Statistiken hinzu
        photon_stats = self.get_photon_statistics()
        base_stats.update(photon_stats)
        
        return base_stats
    
    def get_time_trace(self, bin_width: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Erstellt zeitaufgelösten Photonen-Trace
        
        Args:
            bin_width: Bin-Breite für Histogramm in Sekunden
            
        Returns:
            (time_bins, counts_per_bin)
        """
        if not self.photon_arrival_times:
            return np.array([]), np.array([])
            
        times = np.array(self.photon_arrival_times)
        
        # Erstelle Zeit-Bins
        t_start = times[0]
        t_end = times[-1]
        n_bins = int((t_end - t_start) / bin_width) + 1
        
        time_bins = np.linspace(t_start, t_start + n_bins * bin_width, n_bins + 1)
        
        # Histogramm der Photonen-Counts
        counts, _ = np.histogram(times, bins=time_bins)
        
        # Bin-Zentren für Plot
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        return bin_centers, counts
    
    def reset(self):
        """Setzt Zähler zurück"""
        self.total_photons = 0
        self.emission_events = []
        self.count_history = []
        self.photon_arrival_times = []
        self.last_detection_time = -np.inf
        self.dead_time_losses = 0
        self.afterpulse_buffer = []