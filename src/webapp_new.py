#!/usr/bin/env python3
"""
NV Lab Simulator - Clean API Web Interface
==========================================
RESTful API for NV center laboratory equipment simulation.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
from nv_lab_simulator import NVLabSimulator

# Create Flask app
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Global simulator instance
simulator = NVLabSimulator()

@app.route('/')
def index():
    """Main interface"""
    return render_template('lab_interface.html')

# ===== LASER API =====

@app.route('/api/laser/on', methods=['POST'])
def laser_on():
    """Turn laser on"""
    data = request.json or {}
    power_mW = data.get('power_mW', 3.0)
    wavelength_nm = data.get('wavelength_nm', 532.0)
    
    try:
        result = simulator.laser_on(power_mW, wavelength_nm)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/laser/off', methods=['POST'])
def laser_off():
    """Turn laser off"""
    try:
        result = simulator.laser_off()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/laser/status', methods=['GET'])
def laser_status():
    """Get laser status"""
    return jsonify(simulator.laser_status())

# ===== MW API =====

@app.route('/api/mw/configure', methods=['POST'])
def mw_configure():
    """Configure MW generator"""
    data = request.json or {}
    
    try:
        frequency_Hz = data.get('frequency_Hz', 2.87e9)
        power_dBm = data.get('power_dBm', -10.0)
        pulse_length_ns = data.get('pulse_length_ns', 100.0)
        
        result = simulator.mw_configure(frequency_Hz, power_dBm, pulse_length_ns)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/mw/pulse', methods=['POST'])
def mw_pulse():
    """Send MW pulses"""
    data = request.json or {}
    
    try:
        number_of_pulses = data.get('number_of_pulses', 1)
        repetition_rate_Hz = data.get('repetition_rate_Hz', 1000.0)
        
        result = simulator.mw_pulse(number_of_pulses, repetition_rate_Hz)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/mw/status', methods=['GET'])
def mw_status():
    """Get MW status"""
    return jsonify(simulator.mw_status())

# ===== PHOTON COUNTER API =====

@app.route('/api/counter/start', methods=['POST'])
def counter_start():
    """Start photon counting"""
    data = request.json or {}
    
    try:
        integration_time_ms = data.get('integration_time_ms', 1000.0)
        bins = data.get('bins', 100)
        
        result = simulator.counter_start(integration_time_ms, bins)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/counter/read', methods=['GET'])
def counter_read():
    """Read photon counter"""
    return jsonify(simulator.counter_read())

@app.route('/api/counter/stop', methods=['POST'])
def counter_stop():
    """Stop photon counting"""
    try:
        result = simulator.counter_stop()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ===== CONFIGURATION API =====

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify(simulator.config)

@app.route('/api/config', methods=['PUT'])
def update_config():
    """Update configuration"""
    data = request.json or {}
    
    try:
        updated_keys = simulator.update_config(data)
        return jsonify({
            "success": True,
            "updated": updated_keys
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save configuration to file"""
    data = request.json or {}
    filename = data.get('filename')
    
    try:
        success = simulator.save_config(filename)
        if success:
            return jsonify({"success": True, "filename": filename or simulator.config_file})
        else:
            return jsonify({"success": False, "error": "Failed to save config"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/config/load', methods=['POST'])
def load_config():
    """Load configuration from file"""
    data = request.json or {}
    filename = data.get('filename')
    
    try:
        if filename:
            simulator.config_file = filename
        simulator.config = simulator.load_config()
        return jsonify({
            "success": True,
            "config": simulator.config
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ===== SYSTEM STATUS API =====

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get complete system status"""
    return jsonify(simulator.get_system_status())

# ===== EXPERIMENT API =====

@app.route('/api/experiment/rabi', methods=['POST'])
def experiment_rabi():
    """Start real Rabi oscillation experiment"""
    data = request.json or {}
    
    try:
        # Extract experiment parameters
        tau_start_ns = data.get('tau_start_ns', 0.0)
        tau_end_ns = data.get('tau_end_ns', 1000.0)
        tau_points = data.get('tau_points', 50)
        mw_power_dBm = data.get('mw_power_dBm', -10.0)
        mw_frequency_Hz = data.get('mw_frequency_Hz', 2.87e9)
        
        # Generate tau array
        import numpy as np
        tau_array = np.linspace(tau_start_ns, tau_end_ns, tau_points)
        
        # Update simulator MW parameters
        simulator.mw_configure(mw_frequency_Hz, mw_power_dBm, tau_start_ns)
        
        # Run realistic Rabi simulation
        experiment_id = f"rabi_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Perform tau sweep using the realistic NV simulator
        results = {}
        for i, tau in enumerate(tau_array):
            # Update MW pulse length and simulate
            simulator.mw_configure(mw_frequency_Hz, mw_power_dBm, tau)
            
            # Simulate single measurement
            result = simulator.simulate_single_tau(tau)
            
            # Extract key metrics
            counts = np.array(result['counts'])
            total_counts = np.sum(counts)
            peak_counts = np.max(counts)
            contrast = (peak_counts - np.min(counts)) / peak_counts if peak_counts > 0 else 0
            
            results[float(tau)] = {
                'total_counts': int(total_counts),
                'peak_counts': int(peak_counts),
                'contrast': float(contrast),
                'p_ms0': result['p_ms0'],
                'p_ms1': result['p_ms1'],
                'time_trace': result['counts'][:100]  # First 100 bins for preview
            }
        
        return jsonify({
            "experiment_id": experiment_id,
            "status": "completed",
            "parameters": {
                "tau_range_ns": [tau_start_ns, tau_end_ns],
                "tau_points": tau_points,
                "mw_power_dBm": mw_power_dBm,
                "mw_frequency_Hz": mw_frequency_Hz
            },
            "results": results,
            "tau_array": tau_array.tolist(),
            "physics_info": simulator.get_physics_info()
        })
        
    except Exception as e:
        return jsonify({
            "experiment_id": None,
            "status": "error", 
            "error": str(e),
            "message": "Failed to execute Rabi experiment"
        }), 400

@app.route('/api/experiment/status/<experiment_id>', methods=['GET'])
def experiment_status(experiment_id):
    """Get experiment status - real implementation"""
    # Parse experiment ID to extract info
    try:
        parts = experiment_id.split('_')
        if len(parts) >= 2:
            exp_type = parts[0]
            timestamp = int(parts[1])
            
            # Check if this is a recent experiment (last hour)
            import time
            current_time = int(time.time())
            age_seconds = current_time - timestamp
            
            if age_seconds < 3600:  # 1 hour
                status = "completed" if exp_type == "rabi" else "unknown"
                message = f"Experiment {exp_type} completed {age_seconds}s ago"
            else:
                status = "expired"
                message = f"Experiment data expired ({age_seconds}s old)"
        else:
            status = "invalid"
            message = "Invalid experiment ID format"
            
    except (ValueError, IndexError):
        status = "invalid"
        message = "Could not parse experiment ID"
    
    return jsonify({
        "experiment_id": experiment_id,
        "status": status,
        "message": message,
        "timestamp": timestamp if 'timestamp' in locals() else None
    })

# ===== QUICK TEST ENDPOINTS =====

@app.route('/api/test/simple_measurement', methods=['POST'])
def test_simple_measurement():
    """Quick test: Laser on -> MW pulse -> Count photons"""
    data = request.json or {}
    mw_frequency_Hz = data.get('mw_frequency_Hz', 2.87e9)
    pulse_length_ns = data.get('pulse_length_ns', 100.0)
    
    try:
        # 1. Turn laser on
        laser_result = simulator.laser_on(3.0)
        
        # 2. Configure MW
        mw_result = simulator.mw_configure(mw_frequency_Hz, -10.0, pulse_length_ns)
        
        # 3. Send MW pulse
        pulse_result = simulator.mw_pulse(1)
        
        # 4. Count photons
        counter_start = simulator.counter_start(1000.0, 100)
        
        # Wait a moment for counting to complete
        import time
        time.sleep(1.1)
        
        # 5. Read results
        counts_result = simulator.counter_read()
        
        return jsonify({
            "success": True,
            "sequence": {
                "laser": laser_result,
                "mw_config": mw_result,
                "mw_pulse": pulse_result,
                "counter_start": counter_start,
                "counts": counts_result
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    import time
    print("=== NV Lab Simulator API ===")
    print("Clean API Design")
    print("============================")
    print()
    print("API Endpoints:")
    print("  Laser:   /api/laser/{on,off,status}")
    print("  MW:      /api/mw/{configure,pulse,status}")
    print("  Counter: /api/counter/{start,read,stop}")
    print("  Config:  /api/config")
    print("  System:  /api/status")
    print()
    print("Starting server...")
    
    try:
        app.run(host='0.0.0.0', port=5003, debug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print("Port 5003 in use, trying 5004...")
            app.run(host='0.0.0.0', port=5004, debug=True)
        else:
            raise