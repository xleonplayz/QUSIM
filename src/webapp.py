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
    # Default tau range
    tau_list = list(range(0, 401, 50))  # 0 to 400ns in 50ns steps
    
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

@app.route('/api/rabi_plot')
def get_rabi_plot():
    """Get Rabi oscillation plot with time window averaging"""
    start_time = request.args.get('start', 0, type=int)
    end_time = request.args.get('end', 1000, type=int)
    current_tau = request.args.get('current_tau', 0, type=int)
    
    # Generate tau range with finer resolution (every 10ns up to 500ns)
    tau_list = list(range(0, 501, 10))
    
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