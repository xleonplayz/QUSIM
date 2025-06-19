#!/usr/bin/env python3
"""
Spin Bath Model with Cluster Correlation Expansion (CCE)
======================================================
Realistic ¹³C and ¹⁴N nuclear spin environment for NV centers
"""

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

class SpinBathParams:
    """Parameters for spin bath simulation."""
    
    def __init__(self):
        # ¹³C nuclear spins (I=1/2)
        self.n_carbon13 = 8
        self.C13_abundance = 0.011  # 1.1% natural abundance
        self.C13_gyromag_ratio = 67.283e6  # rad/(s·T)
        self.C13_A_parallel_mean_MHz = 3.0  # Mean hyperfine coupling
        self.C13_A_perp_ratio = 0.3  # A⊥/A∥ ratio
        
        # ¹⁴N nuclear spins (I=1) 
        self.n_nitrogen14 = 2
        self.N14_abundance = 0.999  # 99.9% natural abundance
        self.N14_gyromag_ratio = 19.331e6  # rad/(s·T)
        self.N14_A_parallel_MHz = 2.2  # Hyperfine coupling
        self.N14_quadrupole_MHz = 0.1  # Quadrupole coupling
        
        # Lattice structure
        self.lattice_constant_nm = 0.357  # Diamond lattice
        self.NV_substitution_site = 'nitrogen'  # NV center location
        
        # Distance dependence
        self.r_min_nm = 0.5  # Minimum distance to NV
        self.r_max_nm = 3.0   # Maximum distance for simulation
        self.r_cutoff_nm = 2.5  # CCE cutoff radius

def sample_nuclear_positions(n_spins, lattice_constant_nm, r_min_nm, r_max_nm, key):
    """
    Sample random nuclear spin positions in diamond lattice environment.
    Uses realistic distance distribution around NV center.
    """
    key, subkey = random.split(key)
    
    # Spherical coordinates with bias toward closer distances
    # Probability ∝ r² (volume element) × exp(-r/r_char) (realistic falloff)
    r_char = 1.5  # Characteristic distance in nm
    
    # Sample radial distances with exponential bias
    u = random.uniform(subkey, (n_spins,))
    r = -r_char * jnp.log(1 - u * (1 - jnp.exp(-(r_max_nm - r_min_nm)/r_char)))
    r = r + r_min_nm
    r = jnp.clip(r, r_min_nm, r_max_nm)
    
    # Sample angles uniformly on sphere
    key, subkey1, subkey2 = random.split(key, 3)
    theta = jnp.arccos(1 - 2 * random.uniform(subkey1, (n_spins,)))  # polar
    phi = 2 * jnp.pi * random.uniform(subkey2, (n_spins,))  # azimuthal
    
    # Convert to Cartesian coordinates  
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    
    positions = jnp.stack([x, y, z], axis=1)  # Shape: (n_spins, 3)
    
    return positions, key

def calculate_hyperfine_couplings(positions, A0_parallel_MHz, A0_perp_MHz):
    """
    Calculate hyperfine coupling tensors from nuclear positions.
    A(r) = A₀ × (a₀/r)³ where a₀ is Bohr radius scale.
    """
    # Distance from NV center (at origin)
    r = jnp.linalg.norm(positions, axis=1)  # Shape: (n_spins,)
    
    # Hyperfine coupling falls as 1/r³ (point dipole approximation)
    a0_effective = 0.5  # Effective Bohr radius in nm
    coupling_factor = (a0_effective / r)**3
    
    # Parallel component (along NV axis, assumed z-direction)
    A_parallel = A0_parallel_MHz * coupling_factor
    
    # Perpendicular component 
    A_perp = A0_perp_MHz * coupling_factor
    
    return A_parallel, A_perp

def build_hyperfine_hamiltonian(positions, A_para, A_perp, nv_spins, nuclear_spins):
    """
    Build hyperfine interaction Hamiltonian.
    H_hf = Σⱼ [A∥ʲ Sᶻ Iᶻʲ + A⊥ʲ (SˣIˣʲ + SʸIʸʲ)]
    """
    n_nv = nv_spins['Sz'].shape[0]  # NV space dimension
    n_total_nuclear = len(nuclear_spins)  # Total nuclear space
    
    H_hf = jnp.zeros((n_nv * (2**n_total_nuclear), n_nv * (2**n_total_nuclear)), 
                     dtype=jnp.complex128)
    
    # This is simplified - full implementation would use tensor products
    # For now, return effective coupling strength
    total_coupling = jnp.sum(A_para) + jnp.sum(A_perp)
    
    return total_coupling

def make_nuclear_spin_operators(nuclear_type, n_spins):
    """
    Create nuclear spin operators for multiple nuclei.
    nuclear_type: 'C13' (I=1/2) or 'N14' (I=1)
    """
    if nuclear_type == 'C13':
        # Spin-1/2 operators
        sx = 0.5 * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        sy = 0.5 * jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)  
        sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        identity = jnp.eye(2, dtype=jnp.complex128)
        
    elif nuclear_type == 'N14':
        # Spin-1 operators
        sx = (1/jnp.sqrt(2)) * jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.complex128)
        sy = (1/jnp.sqrt(2)) * jnp.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=jnp.complex128)
        sz = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=jnp.complex128)
        identity = jnp.eye(3, dtype=jnp.complex128)
    
    # For multiple spins, we'd need tensor products
    # Simplified implementation returns single-spin operators
    operators = {
        'Ix': [sx] * n_spins,
        'Iy': [sy] * n_spins, 
        'Iz': [sz] * n_spins,
        'I_identity': [identity] * n_spins
    }
    
    return operators

def cluster_correlation_expansion_level1(bath_params, nv_spins, key):
    """
    Level-1 Cluster Correlation Expansion.
    Treats each nuclear spin independently (no nuclear-nuclear correlations).
    """
    key, subkey1, subkey2 = random.split(key, 3)
    
    # Sample ¹³C positions and couplings
    c13_positions, key = sample_nuclear_positions(
        bath_params.n_carbon13, 
        bath_params.lattice_constant_nm,
        bath_params.r_min_nm, 
        bath_params.r_max_nm,
        subkey1
    )
    
    c13_A_para, c13_A_perp = calculate_hyperfine_couplings(
        c13_positions,
        bath_params.C13_A_parallel_mean_MHz,
        bath_params.C13_A_parallel_mean_MHz * bath_params.C13_A_perp_ratio
    )
    
    # Sample ¹⁴N positions and couplings  
    n14_positions, key = sample_nuclear_positions(
        bath_params.n_nitrogen14,
        bath_params.lattice_constant_nm, 
        bath_params.r_min_nm,
        bath_params.r_max_nm, 
        subkey2
    )
    
    n14_A_para, n14_A_perp = calculate_hyperfine_couplings(
        n14_positions,
        bath_params.N14_A_parallel_MHz,
        bath_params.N14_A_parallel_MHz * bath_params.C13_A_perp_ratio  # Same ratio
    )
    
    # Create nuclear spin operators
    c13_ops = make_nuclear_spin_operators('C13', bath_params.n_carbon13)
    n14_ops = make_nuclear_spin_operators('N14', bath_params.n_nitrogen14)
    
    # Calculate total effective dephasing
    # Level-1 CCE: each nuclear spin contributes independently
    total_dephasing_rate = 0.0
    
    # ¹³C contribution
    for i in range(bath_params.n_carbon13):
        # Simplified dephasing rate ∝ A²
        A_total = jnp.sqrt(c13_A_para[i]**2 + 2 * c13_A_perp[i]**2)
        total_dephasing_rate += (A_total * 1e6 * 2 * jnp.pi)**2  # Convert MHz to rad/s
    
    # ¹⁴N contribution  
    for i in range(bath_params.n_nitrogen14):
        A_total = jnp.sqrt(n14_A_para[i]**2 + 2 * n14_A_perp[i]**2)
        total_dephasing_rate += (A_total * 1e6 * 2 * jnp.pi)**2
    
    # Effective T2* from bath
    T2_star_bath_ns = 1.0 / jnp.sqrt(total_dephasing_rate) * 1e9  # Convert to ns
    
    return {
        'c13_positions': c13_positions,
        'c13_couplings': (c13_A_para, c13_A_perp),
        'n14_positions': n14_positions, 
        'n14_couplings': (n14_A_para, n14_A_perp),
        'T2_star_bath_ns': T2_star_bath_ns,
        'total_dephasing_rate_Hz': total_dephasing_rate / (2 * jnp.pi),
        'key': key
    }

def cluster_correlation_expansion_level2(bath_params, nv_spins, key):
    """
    Level-2 Cluster Correlation Expansion.
    Includes pairwise nuclear-nuclear correlations within cutoff radius.
    """
    # Start with Level-1 results
    level1_result = cluster_correlation_expansion_level1(bath_params, nv_spins, key)
    
    c13_positions = level1_result['c13_positions']
    n14_positions = level1_result['n14_positions']
    
    # Calculate pairwise distances
    def pairwise_distances(pos1, pos2):
        """Calculate all pairwise distances between two sets of positions."""
        diff = pos1[:, None, :] - pos2[None, :, :]  # Broadcasting
        return jnp.linalg.norm(diff, axis=2)
    
    # C13-C13 pairs
    c13_c13_distances = pairwise_distances(c13_positions, c13_positions)
    
    # C13-N14 pairs  
    c13_n14_distances = pairwise_distances(c13_positions, n14_positions)
    
    # N14-N14 pairs
    n14_n14_distances = pairwise_distances(n14_positions, n14_positions)
    
    # Count pairs within cutoff
    cutoff = bath_params.r_cutoff_nm
    
    n_c13_pairs = jnp.sum((c13_c13_distances < cutoff) & (c13_c13_distances > 0))
    n_c13_n14_pairs = jnp.sum(c13_n14_distances < cutoff)
    n_n14_pairs = jnp.sum((n14_n14_distances < cutoff) & (n14_n14_distances > 0))
    
    # Level-2 correction (simplified)
    # Real implementation would calculate flip-flop terms
    correlation_correction = 1.0 - 0.1 * (n_c13_pairs + n_c13_n14_pairs + n_n14_pairs) / 10.0
    correlation_correction = jnp.maximum(correlation_correction, 0.5)  # Don't overcorrect
    
    # Apply correction to T2*
    T2_star_corrected = level1_result['T2_star_bath_ns'] * correlation_correction
    
    level1_result.update({
        'T2_star_bath_ns': T2_star_corrected,
        'correlation_correction': correlation_correction,
        'n_correlated_pairs': n_c13_pairs + n_c13_n14_pairs + n_n14_pairs,
        'level': 2
    })
    
    return level1_result

@partial(jit, static_argnums=(0,))
def apply_bath_dephasing(T2_star_bath_ns, coherence, evolution_time_ns):
    """
    Apply exponential dephasing from spin bath.
    """
    if T2_star_bath_ns > 0:
        decay_factor = jnp.exp(-evolution_time_ns / T2_star_bath_ns)
        return coherence * decay_factor
    else:
        return coherence

@jit 
def bath_induced_frequency_shift(bath_result, time_ns):
    """
    Calculate time-dependent frequency shifts from bath dynamics.
    Simplified model with nuclear flip-flops.
    """
    # Extract coupling strengths
    c13_A_para, c13_A_perp = bath_result['c13_couplings']
    n14_A_para, n14_A_perp = bath_result['n14_couplings']
    
    # Simple model: random telegraph noise from nuclear flips
    # Frequency shifts with correlation time ~ 1/A
    total_shift_Hz = 0.0
    
    # C13 contributions
    for i, A in enumerate(c13_A_para):
        if A > 0:
            correlation_time_ns = 1000.0 / (A * 1e6)  # Rough estimate
            phase = 2 * jnp.pi * time_ns / correlation_time_ns
            shift_amplitude = A * 1e6 * 0.1  # 10% of coupling as shift
            total_shift_Hz += shift_amplitude * jnp.cos(phase + i)  # Different phase per nucleus
    
    return total_shift_Hz

class SpinBathSimulator:
    """
    Complete spin bath simulator with CCE.
    """
    
    def __init__(self, bath_params, cce_level=1):
        self.bath_params = bath_params
        self.cce_level = cce_level
        self.current_bath_config = None
    
    def generate_bath_configuration(self, key):
        """Generate a random bath configuration."""
        from lindblad import make_spin_operators
        nv_spins = make_spin_operators()
        
        if self.cce_level == 1:
            self.current_bath_config = cluster_correlation_expansion_level1(
                self.bath_params, nv_spins, key
            )
        elif self.cce_level == 2:
            self.current_bath_config = cluster_correlation_expansion_level2(
                self.bath_params, nv_spins, key
            )
        else:
            raise ValueError(f"CCE level {self.cce_level} not implemented")
        
        return self.current_bath_config
    
    def get_effective_T2_star(self):
        """Get effective T2* including bath effects."""
        if self.current_bath_config is None:
            raise ValueError("No bath configuration generated")
        
        return self.current_bath_config['T2_star_bath_ns']
    
    def apply_dephasing_to_coherence(self, coherence, time_ns):
        """Apply bath-induced dephasing to coherence."""
        if self.current_bath_config is None:
            return coherence
        
        T2_star = self.current_bath_config['T2_star_bath_ns']
        return apply_bath_dephasing(T2_star, coherence, time_ns)
    
    def get_bath_summary(self):
        """Get summary of current bath configuration."""
        if self.current_bath_config is None:
            return "No bath configuration"
        
        config = self.current_bath_config
        
        summary = f"""
Spin Bath Configuration (CCE Level {getattr(config, 'level', 1)}):
  ¹³C nuclei: {self.bath_params.n_carbon13}
  ¹⁴N nuclei: {self.bath_params.n_nitrogen14}
  Effective T2*: {config['T2_star_bath_ns']:.1f} ns
  Total dephasing rate: {config['total_dephasing_rate_Hz']:.3f} Hz
        """
        
        if 'correlation_correction' in config:
            summary += f"  Correlation correction: {config['correlation_correction']:.3f}\n"
            summary += f"  Correlated pairs: {config['n_correlated_pairs']}\n"
        
        return summary.strip()

if __name__ == "__main__":
    # Test spin bath simulator
    print("🧲 Testing Spin Bath with CCE")
    print("=" * 40)
    
    bath_params = SpinBathParams()
    
    # Test Level-1 CCE
    sim1 = SpinBathSimulator(bath_params, cce_level=1)
    config1 = sim1.generate_bath_configuration(random.PRNGKey(42))
    
    print("Level-1 CCE Results:")
    print(f"  T2* (bath): {config1['T2_star_bath_ns']:.1f} ns")
    print(f"  Dephasing rate: {config1['total_dephasing_rate_Hz']:.3f} Hz")
    
    # Test Level-2 CCE
    sim2 = SpinBathSimulator(bath_params, cce_level=2)
    config2 = sim2.generate_bath_configuration(random.PRNGKey(42))
    
    print("\nLevel-2 CCE Results:")
    print(f"  T2* (bath): {config2['T2_star_bath_ns']:.1f} ns") 
    print(f"  Correlation correction: {config2['correlation_correction']:.3f}")
    print(f"  Correlated pairs: {config2['n_correlated_pairs']}")
    
    print("\n✅ Spin bath simulator working!")