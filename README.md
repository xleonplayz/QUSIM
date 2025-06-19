# HYPERREALISTIC NV Simulator

🔬 **Complete Advanced Physics Implementation**  
🆕 **Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics**

## Features

### ✅ All 9 Advanced Physics Phenomena:
1. **Phonon sidebands (Debye-Waller)** - Temperature-dependent ZPL/PSB emission
2. **T₁ vs T₂* relaxation** - Separate longitudinal/transverse decoherence  
3. **ISC rates (Singlet manifold)** - ms-dependent intersystem crossing
4. **Realistic saturation** - Lorentzian laser saturation with power broadening
5. **Charge-state dynamics** - NV⁻ ↔ NV⁰ transitions with laser enhancement
6. **🆕 Phonon-coupling & Debye-Waller** - Temperature-dependent collection efficiency
7. **🆕 Time-dependent MW pulses** - Gaussian/Hermite/Square with realistic noise
8. **🆕 Spektrale Diffusion** - Ornstein-Uhlenbeck process on zero-field splitting
9. **🆕 Detektor-Modell** - SPAD dead-time, afterpulsing, IRF, and dark counts

### 📊 Realistic Statistics:
- **Photon counts**: ~120 counts/bin (Poisson distributed)
- **Collection efficiency**: 15% effective 
- **Rabi oscillations**: Clean ms=0 ↔ ms=±1 transitions
- **Temperature dependence**: 4K to 300K physics

## Usage

### 🌐 Web Interface (Default)
```bash
python main.py                    # Start at http://localhost:5002
python main.py --port 5003        # Custom port
```

**New Navigation-Based UI:**
- **Sidebar Navigation**: Left-sliding menu with organized sections
- **3 Main Pages**: Dashboard | Parameters | Rabi Oszillation
- **Dashboard**: Key parameters overview + quick actions (Start/Update)
- **Parameters**: Complete physics parameter reference (37+ parameters)
- **Rabi Oszillation**: **Side-by-side** Pulse Timeline & Rabi Oscillation
- **Smart Controls**: Pulse controls only appear on Rabi experiment page
- **Minimalist Header**: Compact design with hamburger menu icon
- **Enhanced Resolution**: 51 pulse points (every 10ns, 0-500ns) for smooth curves
- **1000ns Time Window**: Default integration for better signal averaging
- **Organized Experiments**: Future experiments can be added under "Experimente"

### 🧪 Physics Tests
```bash
python main.py --test             # Test all physics features
```

### 💻 Console Interface  
```bash
python main.py --console          # Interactive console
```

## File Structure

```
simulator/
├── main.py              # Main launcher
├── src/
│   ├── __init__.py      # Package init
│   ├── nv.py           # Physics engine  
│   └── webapp.py       # Web interface
├── templates/
│   └── index.html      # Web UI
└── README.md           # This file
```

## Physics Parameters

### Core Parameters:
- **Rabi frequency**: 5.0 MHz
- **Collection efficiency**: 15%
- **Temperature**: 4.0 K (adjustable)
- **Shots**: 10,000 per simulation

### 🆕 Phonon-Coupling:
- **Debye-Waller factor**: DW₀ = 0.03 at T=0
- **Debye temperature**: θ_D = 150 K  
- **Orbital relaxation**: k_orb ∝ exp(-500K/T)

### 🆕 MW Pulse Noise:
- **Amplitude noise**: 1% standard
- **Phase noise**: ~1° drift
- **Frequency drift**: 1 kHz scale
- **Pulse shapes**: Gaussian, Hermite, Square

## Example Results

```
τ=100ns simulation:
  Max counts: 157
  Mean counts: 119.5
  ms=0 probability: 0.502
  ms=±1 probability: 0.498
```

**Temperature Comparison:**
- T=4K: 0.000 MHz orbital relaxation  
- T=77K: 0.151 MHz orbital relaxation
- T=300K: 18.9 MHz orbital relaxation

## Requirements

- Python 3.8+
- JAX
- NumPy  
- Flask (for web interface)
- Plotly (for interactive plots)

## Advanced Features

All physics phenomena work together to create **hyperrealistic** emergent quantum behavior:
- **No "mocked" data** - Everything emerges from proper quantum mechanics
- **Poisson photon statistics** with realistic count rates
- **Temperature-dependent** phonon coupling affects collection
- **Time-dependent MW pulses** with realistic experimental noise
- **All 35+ physics parameters** properly integrated

🎯 **Result**: The most physically accurate NV simulator with realistic readout statistics!