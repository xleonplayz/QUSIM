import numpy as np
from typing import Dict, List, Tuple

class PhotonCounter:
    def __init__(self, detection_efficiency: float = 0.1, dark_count_rate: float = 100.0):
        """
        Photonenzähler für NV-Zentren
        
        Args:
            detection_efficiency: Detektionseffizienz (0-1)
            dark_count_rate: Dunkelzählrate in Hz
        """
        self.detection_efficiency = detection_efficiency
        self.dark_count_rate = dark_count_rate
        
        # Statistiken
        self.total_photons = 0
        self.emission_events = []
        self.count_history = []
        
    def count_photons(self, rho: np.ndarray, dt: float, gamma_emission: float) -> Dict:
        """
        Zählt emittierte Photonen basierend auf der Dichtematrix
        
        Die Photonenrate ergibt sich aus:
        R = γ * Tr(L†L * ρ) = γ * ⟨e|ρ|e⟩
        
        Args:
            rho: 18×18 Dichtematrix
            dt: Zeitschritt in Sekunden
            gamma_emission: Spontane Emissionsrate in Hz
            
        Returns:
            Dict mit Zählergebnissen
        """
        # Berechne Population im angeregten Zustand
        # |e⟩ Zustände sind Indizes 9-17 (zweiter 9×9 Block)
        pop_excited = np.real(np.trace(rho[9:, 9:]))
        
        # Photonenrate = γ * Population im angeregten Zustand
        emission_rate = gamma_emission * pop_excited
        
        # Erwartete Photonen in diesem Zeitschritt
        expected_photons = emission_rate * dt
        
        # Poisson-Statistik für tatsächliche Photonenzahl
        actual_photons = np.random.poisson(expected_photons * self.detection_efficiency)
        
        # Dunkelzählungen
        dark_counts = np.random.poisson(self.dark_count_rate * dt)
        
        # Gesamtzahl detektierter Photonen
        detected_photons = actual_photons + dark_counts
        
        # Update Statistiken
        self.total_photons += detected_photons
        self.emission_events.append({
            'time': len(self.emission_events) * dt,
            'rate': emission_rate,
            'detected': detected_photons
        })
        
        return {
            'emission_rate': emission_rate,
            'expected_photons': expected_photons,
            'detected_photons': detected_photons,
            'dark_counts': dark_counts,
            'total_count': self.total_photons
        }
    
    def get_correlation_g2(self, tau: float, window: float = 1e-6) -> float:
        """
        Berechnet g²(τ) Korrelationsfunktion
        
        Args:
            tau: Verzögerungszeit
            window: Zeitfenster für Korrelation
            
        Returns:
            g²(τ) Wert
        """
        # Vereinfachte g²(τ) Berechnung
        # Für echte Implementierung: Photon-Ankunftszeiten speichern
        if tau == 0:
            # Antibunching für Einzelphotonen
            return 0.0
        else:
            # Tendiert zu 1 für große τ
            return 1.0 - np.exp(-abs(tau) / window)
    
    def get_statistics(self) -> Dict:
        """Gibt Zählstatistiken zurück"""
        if not self.emission_events:
            return {
                'total_photons': 0,
                'average_rate': 0.0,
                'fano_factor': 0.0
            }
        
        rates = [e['rate'] for e in self.emission_events]
        counts = [e['detected'] for e in self.emission_events]
        
        avg_rate = np.mean(rates)
        avg_count = np.mean(counts)
        var_count = np.var(counts)
        
        # Fano-Faktor F = σ²/μ (sollte ~1 für Poisson sein)
        fano_factor = var_count / avg_count if avg_count > 0 else 0.0
        
        return {
            'total_photons': self.total_photons,
            'average_rate': avg_rate,
            'average_count_rate': avg_count / (self.emission_events[-1]['time'] if self.emission_events else 1.0),
            'fano_factor': fano_factor,
            'detection_efficiency': self.detection_efficiency
        }
    
    def reset(self):
        """Setzt Zähler zurück"""
        self.total_photons = 0
        self.emission_events = []
        self.count_history = []