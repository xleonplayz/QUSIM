import numpy as np
import json
from hama import TotalHamiltonian

class LindbladEvolution:
    def __init__(self, config_path='system.json'):
        """
        Lindblad-Gleichung für das vollständige NV-System:
        
        dρ/dt = -i[H, ρ] + Σ_k L_k ρ L_k† - (1/2){L_k†L_k, ρ}
        
        wobei H der 18×18 Gesamt-Hamiltonian ist.
        """
        self.config_path = config_path
        self.H_builder = TotalHamiltonian(config_path)
        
        # Lade Lindblad-Parameter aus config
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.lindblad_params = config.get('lindblad', {})
        
        # System-Dimension
        self.dim = 18
        
    def build_lindblad_operators(self):
        """Baut die Lindblad-Operatoren L_k"""
        L_ops = []
        
        # 1) Spontane Emission |e⟩ → |g⟩
        if self.lindblad_params.get('spontaneous_emission', {}).get('enabled', False):
            gamma = self.lindblad_params['spontaneous_emission']['gamma']
            
            # |g⟩⟨e| Operator im 18×18 Raum
            # Struktur: (g/e) ⊗ (Elektron) ⊗ (Kern)
            # |g⟩⟨e| = [[0,0],[1,0]] ⊗ I₃ ⊗ I₃
            sigma_ge = np.array([[0, 0], [1, 0]], dtype=complex)
            I3 = np.eye(3, dtype=complex)
            L_emission = np.sqrt(gamma) * np.kron(np.kron(sigma_ge, I3), I3)
            L_ops.append(L_emission)
        
        # 2) Spin-Relaxation T₁
        if self.lindblad_params.get('spin_relaxation', {}).get('enabled', False):
            gamma_spin = self.lindblad_params['spin_relaxation']['gamma']
            
            # Spin-Flip Operatoren S₊, S₋
            Sx, Sy, Sz = self.spin1_ops()
            S_plus = Sx + 1j*Sy
            S_minus = Sx - 1j*Sy
            
            # Erweitere auf 18×18: I₂ ⊗ S ⊗ I₃
            I2 = np.eye(2, dtype=complex)
            I3 = np.eye(3, dtype=complex)
            
            L_plus = np.sqrt(gamma_spin) * np.kron(np.kron(I2, S_plus), I3)
            L_minus = np.sqrt(gamma_spin) * np.kron(np.kron(I2, S_minus), I3)
            L_ops.extend([L_plus, L_minus])
        
        # 3) Dephasing T₂*
        if self.lindblad_params.get('dephasing', {}).get('enabled', False):
            gamma_phi = self.lindblad_params['dephasing']['gamma']
            
            # Dephasing durch Sz
            Sx, Sy, Sz = self.spin1_ops()
            I2 = np.eye(2, dtype=complex)
            I3 = np.eye(3, dtype=complex)
            
            L_dephasing = np.sqrt(gamma_phi) * np.kron(np.kron(I2, Sz), I3)
            L_ops.append(L_dephasing)
        
        # 4) Thermisches Rauschen der Kerne
        if self.lindblad_params.get('nuclear_noise', {}).get('enabled', False):
            gamma_nuc = self.lindblad_params['nuclear_noise']['gamma']
            
            # Nukleare Spin-Flips
            Ix, Iy, Iz = self.spin1_ops()  # Für N14 (I=1)
            I_plus = Ix + 1j*Iy
            I_minus = Ix - 1j*Iy
            
            # Erweitere auf 18×18: I₂ ⊗ I₃ ⊗ I
            I2 = np.eye(2, dtype=complex)
            I3 = np.eye(3, dtype=complex)
            
            L_nuc_plus = np.sqrt(gamma_nuc) * np.kron(np.kron(I2, I3), I_plus)
            L_nuc_minus = np.sqrt(gamma_nuc) * np.kron(np.kron(I2, I3), I_minus)
            L_ops.extend([L_nuc_plus, L_nuc_minus])
        
        return L_ops
    
    def spin1_ops(self):
        """Spin-1 Operatoren"""
        Sx = (1/np.sqrt(2)) * np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=complex)
        Sy = (1/np.sqrt(2)) * np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=complex)
        Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=complex)
        return Sx, Sy, Sz
    
    def commutator(self, A, B):
        """Kommutator [A, B] = AB - BA"""
        return A @ B - B @ A
    
    def anticommutator(self, A, B):
        """Antikommutator {A, B} = AB + BA"""
        return A @ B + B @ A
    
    def lindblad_rhs(self, t, rho_vec):
        """
        Rechte Seite der Lindblad-Gleichung als Vektor
        
        Args:
            t: Zeit
            rho_vec: Dichtematrix als Vektor (18²,)
        
        Returns:
            drho_dt_vec: Zeitableitung als Vektor
        """
        # Reshape Vektor zu Matrix
        rho = rho_vec.reshape((self.dim, self.dim))
        
        # Hamiltonischer Teil: -i[H, ρ]
        H = self.H_builder.build_hamiltonian(t)
        H_term = -1j * self.commutator(H, rho)
        
        # Lindblad-Terme
        L_ops = self.build_lindblad_operators()
        L_term = np.zeros_like(rho)
        
        for L in L_ops:
            L_dag = L.conj().T
            L_term += L @ rho @ L_dag - 0.5 * self.anticommutator(L_dag @ L, rho)
        
        # Gesamte Zeitableitung
        drho_dt = H_term + L_term
        
        # Zurück zu Vektor
        return drho_dt.flatten()
    
    def initial_state(self, state_type='ground'):
        """
        Erzeugt Anfangszustand
        
        Args:
            state_type: 'ground', 'thermal', 'custom'
        
        Returns:
            rho0: 18×18 Anfangs-Dichtematrix
        """
        if state_type == 'ground':
            # Grundzustand |g,ms=0,mI=0⟩
            psi0 = np.zeros(self.dim, dtype=complex)
            psi0[9] = 1.0  # Index für |g⟩⊗|0⟩⊗|0⟩
            rho0 = np.outer(psi0, psi0.conj())
            
        elif state_type == 'thermal':
            # Thermischer Zustand bei gegebener Temperatur
            T = self.lindblad_params.get('temperature', 300.0)  # K
            kB = 1.380649e-23  # J/K
            
            # Hamiltonian bei t=0
            H = self.H_builder.build_hamiltonian(0.0)
            
            # Boltzmann-Verteilung
            beta = 1.0 / (kB * T)
            rho0 = np.exp(-beta * H)
            rho0 = rho0 / np.trace(rho0)
            
        elif state_type == 'custom':
            # Benutzerdefinierter Zustand aus config
            custom_state = self.lindblad_params.get('initial_state', {})
            # Implementation für custom state...
            rho0 = np.eye(self.dim, dtype=complex) / self.dim  # Fallback
            
        else:
            # Maximally mixed state
            rho0 = np.eye(self.dim, dtype=complex) / self.dim
        
        return rho0
    
    def evolve(self, t_span, initial_state_type='ground', method='RK45'):
        """
        Zeitentwicklung mit scipy.integrate
        
        Args:
            t_span: Zeitbereich [t0, tf] oder Array
            initial_state_type: Typ des Anfangszustands
            method: Integrationsmethode
            
        Returns:
            t_eval: Zeitpunkte
            rho_t: Dichtematrizen zu allen Zeiten
        """
        from scipy.integrate import solve_ivp
        
        # Anfangszustand
        rho0 = self.initial_state(initial_state_type)
        rho0_vec = rho0.flatten()
        
        # Zeitentwicklung
        if isinstance(t_span, (list, tuple)) and len(t_span) == 2:
            # Automatische Zeitauflösung
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        else:
            t_eval = np.array(t_span)
        
        sol = solve_ivp(
            self.lindblad_rhs, 
            [t_eval[0], t_eval[-1]], 
            rho0_vec,
            t_eval=t_eval,
            method=method,
            rtol=1e-8,
            atol=1e-10
        )
        
        # Reshape Ergebnisse
        rho_t = []
        for i in range(len(sol.t)):
            rho = sol.y[:, i].reshape((self.dim, self.dim))
            rho_t.append(rho)
        
        return sol.t, rho_t
    
    def observables(self, rho_t):
        """
        Berechnet Observablen
        
        Args:
            rho_t: Liste von Dichtematrizen
            
        Returns:
            obs: Dictionary mit Observablen
        """
        obs = {
            'population_g': [],  # Grundzustandspopulation
            'population_e': [],  # Angeregter Zustand
            'spin_z': [],        # ⟨Sz⟩
            'coherence': [],     # |⟨g|ρ|e⟩|
            'purity': []         # Tr(ρ²)
        }
        
        # Projektoren
        P_g = np.zeros((self.dim, self.dim), dtype=complex)
        P_e = np.zeros((self.dim, self.dim), dtype=complex)
        
        # |g⟩ und |e⟩ Projektoren (vereinfacht)
        P_g[:9, :9] = np.eye(9)  # Erste 9×9 Block
        P_e[9:, 9:] = np.eye(9)  # Zweite 9×9 Block
        
        # Sz Operator
        Sx, Sy, Sz = self.spin1_ops()
        I2 = np.eye(2, dtype=complex)
        I3 = np.eye(3, dtype=complex)
        Sz_full = np.kron(np.kron(I2, Sz), I3)
        
        for rho in rho_t:
            # Populationen
            obs['population_g'].append(np.real(np.trace(P_g @ rho)))
            obs['population_e'].append(np.real(np.trace(P_e @ rho)))
            
            # Spin-Erwartungswert
            obs['spin_z'].append(np.real(np.trace(Sz_full @ rho)))
            
            # Kohärenz |⟨g|ρ|e⟩|
            coherence = np.abs(np.trace(P_g @ rho @ P_e))
            obs['coherence'].append(coherence)
            
            # Reinheit
            purity = np.real(np.trace(rho @ rho))
            obs['purity'].append(purity)
        
        return obs