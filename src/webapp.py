#!/usr/bin/env python3
"""
NV Simulator - Web Application Interface
========================================
<� HYPERREALISTIC: Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics
"""

from flask import Flask, render_template, jsonify, request
import json
import numpy as np
import os
from .nv import NVSimulator

# Fix template path - go up one level from src/ to find templates/
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Global simulator instance
simulator = NVSimulator()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/physics_info')
def get_physics_info():
    """Get all physics parameters"""
    return jsonify(simulator.get_physics_info())

@app.route('/api/simulate_tau_sweep')
def simulate_tau_sweep():
    """Simulate a full tau sweep with progress tracking"""
    # Default tau range - reduced for faster initial simulation
    tau_list = list(range(0, 501, 25))  # 0 to 500ns in 25ns steps (21 points)
    
    # Get custom range if provided
    start = request.args.get('start', type=int)
    end = request.args.get('end', type=int)
    step = request.args.get('step', type=int)
    
    if start is not None and end is not None:
        step = step or 50
        tau_list = list(range(start, end + 1, step))
    
    # Progress callback for tracking
    def progress_callback(i, total, tau):
        progress = int((i / total) * 100)
        print(f"Progress: {progress}% - Simulating tau={tau}ns ({i+1}/{total})")
    
    try:
        results = simulator.simulate_tau_sweep(tau_list, progress_callback=progress_callback)
        return jsonify(results)
    except Exception as e:
        print(f"Simulation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pulse_plot/<int:tau>')
def get_pulse_plot(tau):
    """Get pulse readout plot for specific tau"""
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', 200, type=int)
    
    result = simulator.simulate_single_tau(tau, random_seed=42)
    
    time_ns = np.array(result['time_ns'])
    counts = np.array(result['counts'])
    
    # Filter time range
    mask = (time_ns >= start) & (time_ns <= end)
    time_filtered = time_ns[mask]
    counts_filtered = counts[mask]
    
    return jsonify({
        'time_ns': time_filtered.tolist(),
        'counts': counts_filtered.tolist(),
        'tau_ns': tau,
        'p_ms0': result['p_ms0'],
        'p_ms1': result['p_ms1'],
        'max_counts': int(max(counts_filtered)),
        'mean_counts': float(np.mean(counts_filtered))
    })

@app.route('/api/get_pulse_data/<int:tau>')
def get_pulse_data(tau):
    """Get pulse data for tau slider"""
    result = simulator.simulate_single_tau(tau, random_seed=42)
    
    counts = np.array(result['counts'])
    return jsonify({
        'tau_ns': tau,
        'p_ms0': result['p_ms0'],
        'p_ms1': result['p_ms1'],
        'max_counts': int(max(counts)),
        'mean_counts': float(np.mean(counts)),
        'total_counts': int(sum(counts))
    })

@app.route('/api/load_sample_data')
def load_sample_data():
    """Load and analyze sample data files"""
    import os
    
    # Path to sample directory
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample')
    
    try:
        sample_files = []
        
        # List all .dat files in sample directory
        for filename in os.listdir(sample_dir):
            if filename.endswith('.dat'):
                filepath = os.path.join(sample_dir, filename)
                
                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                sample_files.append({
                    'filename': filename,
                    'content': content,
                    'size': len(content)
                })
        
        return jsonify({
            'success': True,
            'files': sample_files,
            'count': len(sample_files)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rabi_plot')
def get_rabi_plot():
    """Get Rabi oscillation plot with time window averaging"""
    start_time = request.args.get('start', 0, type=int)
    end_time = request.args.get('end', 1000, type=int)
    current_tau = request.args.get('current_tau', 0, type=int)
    
    # Generate tau range with good resolution for smooth Rabi curve
    tau_list = list(range(0, 501, 5))  # Every 5ns for smooth curve
    
    # Progress callback for Rabi plot
    def rabi_progress(i, total, tau):
        if i % 10 == 0:  # Update every 10th point to avoid spam
            progress = int((i / total) * 100)
            print(f"Rabi Progress: {progress}% - tau={tau}ns")
    
    # Simulate
    try:
        results = simulator.simulate_tau_sweep(tau_list, progress_callback=rabi_progress)
    except Exception as e:
        print(f"Rabi simulation error: {e}")
        return jsonify({'error': str(e)}), 500
    
    # Extract data with time window averaging
    tau_values = []
    p_ms0_values = []
    mean_counts = []
    
    for tau in tau_list:
        if tau in results:
            tau_values.append(tau)
            p_ms0_values.append(results[tau]['p_ms0'])
            
            # Average counts over the specified time window
            time_ns = np.array(results[tau]['time_ns'])
            counts = np.array(results[tau]['counts'])
            
            # Filter to time window
            mask = (time_ns >= start_time) & (time_ns <= end_time)
            if np.any(mask):
                window_counts = counts[mask]
                mean_counts.append(float(np.mean(window_counts)))
            else:
                mean_counts.append(float(np.mean(counts)))
    
    return jsonify({
        'tau_values': tau_values,
        'p_ms0_values': p_ms0_values,
        'mean_counts': mean_counts,
        'current_tau': current_tau
    })

@app.route('/api/pulse_detail/<int:pulse_number>')
def get_pulse_detail(pulse_number):
    """Get detailed photon count data for a specific pulse"""
    # Load actual PODMR data
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample')
    data_file = os.path.join(sample_dir, '20250527-0002-49_Qubit_XQ1i_PODMR_amp_0.012V_laser_pulses.dat')
    
    try:
        # Read the file and skip header
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Find end of header
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '# ---- END HEADER ----':
                data_start = i + 1
                break
        
        # Parse pulses (each line is one pulse)
        pulses = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Split by tabs and convert to integers
                    photon_counts = [int(x) for x in line.split('\t') if x.strip()]
                    if photon_counts:  # Only add non-empty pulses
                        pulses.append(photon_counts)
                except ValueError:
                    continue
        
        # Check if pulse number is valid
        if pulse_number < 0 or pulse_number >= len(pulses):
            return jsonify({'error': f'Invalid pulse number. Available: 0-{len(pulses)-1}'}), 400
        
        # Get the specific pulse data
        pulse_data = pulses[pulse_number]
        time_points = list(range(len(pulse_data)))  # Time points from left to right
        
        return jsonify({
            'pulse_number': pulse_number,
            'time_points': time_points,
            'photon_counts': pulse_data,
            'total_photons': sum(pulse_data),
            'max_photons': max(pulse_data),
            'min_photons': min(pulse_data),
            'pulse_length': len(pulse_data)
        })
        
    except Exception as e:
        print(f"Error loading pulse detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/podmr_rabi_plot')
def get_podmr_rabi_plot():
    """Get PODMR Rabi oscillation plot with time window integration"""
    # Get time window parameters
    start_time_ns = request.args.get('start_time', 0, type=float)
    end_time_ns = request.args.get('end_time', 200, type=float)
    
    # Load actual PODMR data
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample')
    data_file = os.path.join(sample_dir, '20250527-0002-49_Qubit_XQ1i_PODMR_amp_0.012V_laser_pulses.dat')
    
    try:
        # Read the file and skip header
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Find end of header and extract bin width
        data_start = 0
        bin_width_s = 3.2e-9  # Default bin width in seconds
        for i, line in enumerate(lines):
            if line.strip() == '# ---- END HEADER ----':
                data_start = i + 1
                break
            if line.startswith('# bin width (s)='):
                bin_width_s = float(line.split('=')[1])
        
        # Convert bin width to nanoseconds
        bin_width_ns = bin_width_s * 1e9
        
        # Parse pulses (each line is one pulse)
        pulses = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Split by tabs and convert to integers
                    photon_counts = [int(x) for x in line.split('\t') if x.strip()]
                    if photon_counts:  # Only add non-empty pulses
                        pulses.append(photon_counts)
                except ValueError:
                    continue
        
        # Calculate TOTAL COUNTS for EACH pulse in the time window
        pulse_totals = []
        pulse_numbers = []
        
        for i, pulse in enumerate(pulses):
            # Create time axis for this pulse
            time_axis = np.array([j * bin_width_ns for j in range(len(pulse))])
            
            # Find indices within the time window
            mask = (time_axis >= start_time_ns) & (time_axis <= end_time_ns)
            
            if np.any(mask):
                # SUM all photon counts within the time window for THIS pulse
                windowed_counts = np.array(pulse)[mask]
                total_count = float(np.sum(windowed_counts))
                pulse_totals.append(total_count)
                pulse_numbers.append(i)
        
        return jsonify({
            'pulse_numbers': pulse_numbers,
            'pulse_totals': pulse_totals,
            'total_pulses': len(pulses),
            'time_window': {
                'start_ns': start_time_ns,
                'end_ns': end_time_ns,
                'bin_width_ns': bin_width_ns
            },
            'mean_total': float(np.mean(pulse_totals)) if pulse_totals else 0,
            'std_total': float(np.std(pulse_totals)) if pulse_totals else 0
        })
        
    except Exception as e:
        print(f"Error loading PODMR data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emission_spectrum/<int:tau>')
def get_emission_spectrum(tau):
    """Get NV emission spectrum (ZPL vs PSB) for specific tau"""
    try:
        # Get the NV parameters
        params = simulator.params
        
        # Calculate temperature-dependent Debye-Waller factor
        from .nv import debye_waller_factor
        DW_factor = debye_waller_factor(params.Temperature_K, params)
        
        # Wavelength range for NV emission (NV- center: 637nm ZPL + PSB up to 800nm)
        wavelengths_nm = np.linspace(630, 800, 1000)
        
        # NV- zero-phonon line at 637 nm (correct!)
        zpl_center_nm = 637.0
        zpl_width_nm = 0.13  # Natural linewidth ~0.13 nm at low T
        
        # Phonon sideband: vibronic progression starting from ZPL
        # Main vibronic peak around 700-750 nm region
        psb_center_nm = 700.0  # Center of vibronic envelope
        psb_width_nm = 50.0    # Broad vibronic envelope
        
        # ZPL intensity (Gaussian profile)
        zpl_intensity = DW_factor * np.exp(-0.5 * ((wavelengths_nm - zpl_center_nm) / zpl_width_nm)**2)
        
        # PSB intensity (broader Gaussian with vibronic structure)
        psb_base = (1 - DW_factor) * np.exp(-0.5 * ((wavelengths_nm - psb_center_nm) / psb_width_nm)**2)
        
        # Realistic vibronic structure for NV- PSB
        # Main vibronic peaks: 0-phonon (637nm), 1-phonon (~700nm), 2-phonon (~770nm)
        vibronic_peaks = [
            {'center': 700, 'width': 15, 'strength': 0.4},  # 1-phonon (strongest)
            {'center': 770, 'width': 25, 'strength': 0.2},  # 2-phonon 
            {'center': 650, 'width': 10, 'strength': 0.1},  # Weak local phonon
            {'center': 730, 'width': 20, 'strength': 0.3}   # Mixed vibronic mode
        ]
        
        psb_intensity = np.zeros_like(wavelengths_nm)
        
        for peak in vibronic_peaks:
            peak_contribution = peak['strength'] * (1 - DW_factor) * np.exp(
                -0.5 * ((wavelengths_nm - peak['center']) / peak['width'])**2
            )
            psb_intensity += peak_contribution
        
        # Normalize to maximum of 1
        max_total = max(np.max(zpl_intensity), np.max(psb_intensity))
        if max_total > 0:
            zpl_intensity = zpl_intensity / max_total
            psb_intensity = psb_intensity / max_total
        
        # Tau-dependent effects (excited state population affects spectrum)
        # Longer pulses -> more excited state population -> stronger emission
        if params.Omega_Rabi_Hz > 0:
            rabi_angle = params.Omega_Rabi_Hz * tau * 1e-9 * np.pi
            excitation_factor = np.sin(rabi_angle/2)**2  # Probability of exciting the NV
        else:
            excitation_factor = 0.0
        
        # Scale intensities by excitation
        zpl_intensity = zpl_intensity * (0.1 + 0.9 * excitation_factor)  # 10% baseline + 90% excited
        psb_intensity = psb_intensity * (0.1 + 0.9 * excitation_factor)
        
        return jsonify({
            'wavelengths_nm': wavelengths_nm.tolist(),
            'zpl_intensity': zpl_intensity.tolist(),
            'psb_intensity': psb_intensity.tolist(),
            'tau_ns': tau,
            'dw_factor': float(DW_factor),
            'temperature_K': params.Temperature_K,
            'zpl_fraction': float(DW_factor),
            'psb_fraction': float(1 - DW_factor)
        })
        
    except Exception as e:
        print(f"Emission spectrum error: {e}")
        return jsonify({'error': str(e)}), 500

def run_webapp(port=5002, debug=True):
    """Run the web application"""
    print("=� Starting NV Simulator Web App...")
    print("=" * 40)
    print("=� Interactive Plotly interface")
    print("=, Advanced physics simulation")
    print("< Web-based - works in any browser")
    print()
    print("Starting Flask server...")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Address already in use")
            print(f"Port {port} is in use by another program. Either identify and stop that program, or start the server with a different port.")
            # Try next port
            try:
                port += 1
                print(f"Trying port {port}...")
                app.run(host='0.0.0.0', port=port, debug=debug)
            except Exception as e2:
                print(f"Failed to start on port {port}: {e2}")
        else:
            print(f"Failed to start server: {e}")

if __name__ == "__main__":
    run_webapp()