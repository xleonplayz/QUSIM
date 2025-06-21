        function displayPhysicsInfo() {
            const grid = document.getElementById('physics-grid');
            
            // Select key parameters to show on dashboard
            const keyParameters = [
                // Basic Physics
                {category: "Basic Physics", name: "Omega_Rabi_Hz", displayName: "Rabi Freq", unit: "Hz", scale: 1e-6, displayUnit: "MHz"},
                {category: "Basic Physics", name: "BIN_ns", displayName: "Bin Width", unit: "ns", scale: 1, displayUnit: "ns"},
                {category: "Basic Physics", name: "READ_NS", displayName: "Readout", unit: "ns", scale: 1, displayUnit: "ns"}, 
                {category: "Basic Physics", name: "Shots", displayName: "Shots", unit: "", scale: 1, displayUnit: ""},
                
                // Collection & Efficiency
                {category: "Collection & Efficiency", name: "collection_efficiency", displayName: "Collection η", unit: "", scale: 100, displayUnit: "%"},
                {category: "Collection & Efficiency", name: "DW_factor", displayName: "DW Factor", unit: "", scale: 1, displayUnit: ""},
                
                // Relaxation Times
                {category: "Relaxation Times", name: "T1_ms", displayName: "T₁", unit: "ms", scale: 1, displayUnit: "ms"},
                {category: "Relaxation Times", name: "T2_star_us", displayName: "T₂*", unit: "μs", scale: 1, displayUnit: "μs"},
                
                // ISC & Optical
                {category: "ISC & Optical", name: "gamma_ISC_ms1_MHz", displayName: "ISC ms=±1", unit: "MHz", scale: 1, displayUnit: "MHz"},
                {category: "ISC & Optical", name: "gamma_ISC_ms0_MHz", displayName: "ISC ms=0", unit: "MHz", scale: 1, displayUnit: "MHz"},
                {category: "ISC & Optical", name: "laser_power_mW", displayName: "Laser Power", unit: "mW", scale: 1, displayUnit: "mW"},
                
                // Temperature
                {category: "Temperature & Activation", name: "Temperature_K", displayName: "Temperature", unit: "K", scale: 1, displayUnit: "K"},
                
                // Spin Bath
                {category: "Spin Bath", name: "n_carbon13", displayName: "¹³C Bath", unit: "", scale: 1, displayUnit: ""},
                {category: "Spin Bath", name: "n_nitrogen14", displayName: "¹⁴N Bath", unit: "", scale: 1, displayUnit: ""},
                
                // Detector
                {category: "Detector Model", name: "dead_time_ns", displayName: "Dead Time", unit: "ns", scale: 1, displayUnit: "ns"},
                
                // MW Pulse
                {category: "MW Pulse", name: "mw_pulse_shape", displayName: "MW Shape", unit: "", scale: 1, displayUnit: ""},
                {category: "MW Pulse", name: "mw_sigma_ns", displayName: "MW σ", unit: "ns", scale: 1, displayUnit: "ns"},
                
                // Phonon
                {category: "Phonon & Debye-Waller", name: "theta_D_K", displayName: "θ_D", unit: "K", scale: 1, displayUnit: "K"},
                {category: "Phonon & Debye-Waller", name: "gamma_rad_MHz", displayName: "γ_rad", unit: "MHz", scale: 1, displayUnit: "MHz"},
            ];
            
            let html = '';
            
            for (const keyParam of keyParameters) {
                const categoryParams = simulationParameters[keyParam.category];
                if (categoryParams) {
                    const param = categoryParams.find(p => p.name === keyParam.name);
                    if (param) {
                        html += `<div class="physics-item">
                            <div class="physics-item-info">
                                <div class="physics-item-name">${keyParam.displayName}</div>
                                <div class="physics-item-desc">${param.description}</div>
                            </div>
                            <div class="physics-item-control">`;
                        
                        if (param.options) {
                            html += `<select class="physics-input" id="param_${param.name}" onchange="updateParameterValue('${param.name}', this.value)">`;
                            for (const option of param.options) {
                                const selected = option === param.value ? 'selected' : '';
                                html += `<option value="${option}" ${selected}>${option}</option>`;
                            }
                            html += `</select>`;
                        } else {
                            html += `<input type="number" 
                                class="physics-input" 
                                id="param_${param.name}" 
                                value="${param.value}" 
                                min="${param.min || ''}" 
                                max="${param.max || ''}" 
                                step="${param.step || 'any'}"
                                onchange="updateParameterValue('${param.name}', this.value)">`;
                        }
                        
                        html += `<span class="physics-unit">${keyParam.displayUnit}</span>
                            </div>
                        </div>`;
                    }
                }
            }
            
            grid.innerHTML = html;
        }