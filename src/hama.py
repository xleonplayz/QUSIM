# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
import json
from ZFS.zfs import H_ZFS, spin1_ops
from zeeman.zeeman import H_Zeeman
from HHN14.n14 import H_N14_realistic_18x18
from n14qu.n14 import H_Q_N14, extend_to_18
from HHFC13.c13 import H_C13_in_18x18
from strain.strain import H_strain_3x3
from stark.stark import H_stark_3x3
from MW.mw import H_MW
from LA.laser import H_laser
from JahnTeller.jt_numpy import H_JT_18x18

class TotalHamiltonian:
    def __init__(self, config_path='system.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.Sx, self.Sy, self.Sz = spin1_ops()
        
        # Laser-Interface (wird von außen gesetzt)
        self.laser_interface = None
    
    def set_laser_interface(self, laser_interface):
        """Setzt das Laser-Interface"""
        self.laser_interface = laser_interface
    
    def build_hamiltonian(self, t=0.0):
        """Baut den Gesamt-Hamiltonian durch Addition aller aktivierten Terme"""
        H_total = np.zeros((18, 18), dtype=complex)
        
        # ZFS Term
        if self.config.get('ZFS', {}).get('enabled', False):
            zfs_params = self.config['ZFS']
            D = zfs_params['D']
            Ex = zfs_params.get('Ex', 0.0)
            Ey = zfs_params.get('Ey', 0.0)
            
            # 3x3 ZFS Matrix
            H_zfs_3x3 = H_ZFS(D, Ex, Ey, self.Sx, self.Sy, self.Sz)
            
            # Erweitere auf 18x18 (Methode 2: Tensor-Produkt)
            I6 = np.eye(6, dtype=complex)
            H_zfs_18x18 = np.kron(I6, H_zfs_3x3)
            
            H_total += H_zfs_18x18
        
        # Zeeman Term
        if self.config.get('Zeeman', {}).get('enabled', False):
            zeeman_params = self.config['Zeeman']
            
            # 3x3 Zeeman Matrix
            H_zeeman_3x3 = H_Zeeman(t, zeeman_params, self.Sx, self.Sy, self.Sz)
            
            # Erweitere auf 18x18 (Methode 2: Tensor-Produkt)
            I6 = np.eye(6, dtype=complex)
            H_zeeman_18x18 = np.kron(I6, H_zeeman_3x3)
            
            H_total += H_zeeman_18x18
        
        # N14 Hyperfine Term
        if self.config.get('N14_Hyperfine', {}).get('enabled', False):
            n14_params = self.config['N14_Hyperfine']
            r_vec = np.array(n14_params['r_vec'])
            rho_e0 = n14_params.get('rho_e0', 2.688e28)
            A_iso = n14_params.get('A_iso', 2.14e6)
            P_quad = n14_params.get('P_quad', -5.0e6)
            
            # Direkt 18x18
            H_n14_hf = H_N14_realistic_18x18(r_vec, rho_e0, A_iso, P_quad)
            
            H_total += H_n14_hf
        
        # N14 Quadrupole Term  
        if self.config.get('N14_Quadrupole', {}).get('enabled', False):
            n14q_params = self.config['N14_Quadrupole']
            P = n14q_params['P']
            eta = n14q_params.get('eta', 0.0)
            
            # 3x3 Quadrupole Matrix
            H_quad_3x3 = H_Q_N14(P, eta)
            
            # Erweitere auf 18x18
            H_quad_18x18 = extend_to_18(H_quad_3x3)
            
            H_total += H_quad_18x18
        
        # C13 Hyperfine Term
        if self.config.get('C13_Hyperfine', {}).get('enabled', False):
            c13_params = self.config['C13_Hyperfine']
            r_vec = np.array(c13_params['r_vec'])
            A_iso = c13_params.get('A_iso', 0.0)
            
            # Direkt 18x18
            H_c13 = H_C13_in_18x18(r_vec, A_iso)
            
            H_total += H_c13
        
        # Strain Term
        if self.config.get('Strain', {}).get('enabled', False):
            strain_params = self.config['Strain']
            r_um = np.array(strain_params.get('r_um', [0.0, 0.0, 0.5]))
            
            # 3x3 Strain Matrix  
            H_strain = H_strain_3x3(t, r_um, S_ops=(self.Sx, self.Sy, self.Sz))
            
            # Erweitere auf 18x18 (Tensor-Produkt)
            I6 = np.eye(6, dtype=complex)
            H_strain_18x18 = np.kron(I6, H_strain)
            
            H_total += H_strain_18x18
        
        # Stark Term
        if self.config.get('Stark', {}).get('enabled', False):
            stark_params = self.config['Stark']
            
            # Parameter für Stark-Effekt
            params_stark = {
                "D_PA": np.array(stark_params.get('D_PA', [[17e-3, 0, 0], [0, -17e-3, 0], [0, 0, 3.5e-3]])),
                "euler_D": tuple(stark_params.get('euler_D', [0.0, 0.0, 0.0])),
                "E0_lab": np.array(stark_params.get('E0_lab', [0.0, 0.0, 0.0])),
                "E_modes": stark_params.get('E_modes', []),
                "E_noise": lambda t: np.zeros(3)  # Vereinfacht
            }
            
            # 3x3 Stark Matrix
            H_stark = H_stark_3x3(t, params_stark, S_ops=(self.Sx, self.Sy, self.Sz))
            
            # Erweitere auf 18x18 (Tensor-Produkt)
            I6 = np.eye(6, dtype=complex)
            H_stark_18x18 = np.kron(I6, H_stark)
            
            H_total += H_stark_18x18
        
        # MW (Mikrowellen) Term
        if self.config.get('MW', {}).get('enabled', False):
            mw_params = self.config['MW']
            
            # MW Parameter
            Omegas = np.array(mw_params.get('Omegas', []))
            Phis = np.array(mw_params.get('Phis', []))
            T = mw_params.get('T', 1e-6)  # Pulsdauer
            
            # MW ist bereits 18x18
            H_mw = H_MW(t, Omegas, Phis, T)
            
            H_total += H_mw
        
        # Laser Term (aus system.json)
        if self.config.get('Laser', {}).get('enabled', False):
            laser_params = self.config['Laser']
            
            # Laser Parameter
            omega_L = laser_params.get('omega_L', 2 * np.pi * 4.66e14)  # rad/s
            Omega_L_type = laser_params.get('Omega_L_type', 'constant')
            
            # Omega_L kann konstant oder zeitabhängig sein
            if Omega_L_type == 'constant':
                Omega_L = laser_params.get('Omega_L', 0.0)
            elif Omega_L_type == 'pulse':
                # Rechteckpuls
                pulse_amp = laser_params.get('pulse_amplitude', 2 * np.pi * 10e6)
                pulse_start = laser_params.get('pulse_start', 0.0)
                pulse_duration = laser_params.get('pulse_duration', 100e-9)
                Omega_L = lambda t: pulse_amp if pulse_start <= t < pulse_start + pulse_duration else 0.0
            else:
                Omega_L = 0.0
            
            # Laser ist bereits 18x18
            H_laser_18x18 = H_laser(t, Omega_L, omega_L)
            
            H_total += H_laser_18x18
        
        # Laser Interface (dynamisch schaltbar)
        if self.laser_interface is not None:
            H_laser_interface = self.laser_interface.get_hamiltonian_contribution(t)
            H_total += H_laser_interface
        
        # Jahn-Teller Term
        if self.config.get('JahnTeller', {}).get('enabled', False):
            jt_params = self.config['JahnTeller']
            
            # Parameter für Jahn-Teller
            params_jt = {
                'omega_x': jt_params.get('omega_x', 1.0 * 2*np.pi),
                'omega_y': jt_params.get('omega_y', 1.2 * 2*np.pi),
                'lambda_x': jt_params.get('lambda_x', 0.05),
                'lambda_y': jt_params.get('lambda_y', 0.04),
                'alpha_x': jt_params.get('alpha_x', 0.01),
                'alpha_y': jt_params.get('alpha_y', 0.01)
            }
            
            # Jahn-Teller ist bereits 18x18
            H_jt = H_JT_18x18(params_jt)
            
            H_total += H_jt
        
        return H_total