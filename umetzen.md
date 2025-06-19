Unten findest du zu **jedem** Punkt aus der Tabelle einen kurzen Text, was genau ergänzt werden muss, und ein *Code-Snippet*, das du in deinen Simulator integrieren kannst. Dabei gehe ich davon aus, dass du bereits eine Klasse oder ein Modul hast, in dem du den Hamiltonian, die Dissipatoren, das Photon-Sampling etc. unterbringst.

> **Hinweis**: Du musst natürlich die Importe (`import jax.numpy as jnp`, `from jax import jit` etc.) an den Anfang deiner Datei setzen, und die Parameterklassen (`NVParams` o.ä.) entsprechend um neue Felder erweitern.

---

## 1. Grundzustand-Hamiltonian

**Was fehlt?**
– Vollständige 3×3 Spin-1-Hamiltonian

**Ergänzung:**

```python
@jit
def H_gs(p: NVParams, B: jnp.ndarray, strain: float, delta: float):
    """
    Volles Spin-1 Hamiltonian im GS:
      Dgs, Zeeman in x,y,z, Hyperfein 14N, Strain/Stark, spektrale Diffusion (delta)
    """
    S = spin_ops()
    I3 = jnp.eye(3)
    # Zero‐Field Splitting
    H_zfs = 2 * jnp.pi * p.D_GS_Hz * (S["Sz"]@S["Sz"] - (2/3)*I3)
    # Zeeman-Term in x,y,z
    gamma = p.g_e * p.MU_B / p.HBAR
    H_zeeman = 2 * jnp.pi * gamma * (B[0]*S["Sx"] + B[1]*S["Sy"] + B[2]*S["Sz"])
    # Hyperfein 14N (axial A∥, transversal A⊥)
    H_hf = 2*jnp.pi * (
       p.A_para_N_Hz * S["Sz"]*p.mI 
     + p.A_perp_N_Hz * (S["Sx"]*p.Ix_N + S["Sy"]*p.Iy_N)
    )
    # Strain/Zener‐Stark
    H_strain = 2*jnp.pi * strain * (S["Sx"]@S["Sx"] - S["Sy"]@S["Sy"])
    # Spektrale Diffusion als zusätzlicher δ‐Term (detuning)
    H_delta = 2*jnp.pi * delta * (S["Sz"]@S["Sz"])
    return H_zfs + H_zeeman + H_hf + H_strain + H_delta
```

*Einbindung:* Rufe `H_gs` statt deines vereinfachten Hamiltonians und übergebe das B‐Feld als Vektor, das zufällige δ aus OU usw.

---

## 2. Angeregter Zustand ($^3E$)

**Was fehlt?**
– Eigener Hamiltonian mit Spin-Orbit‐Term, Grunde auf 6×6 oder 3×3+Orbitaldimension

**Ergänzung (vereinfachter Ansatz):**

```python
@jit
def H_es(p: NVParams, B: jnp.ndarray, strain: float, delta_es: float):
    """Simplifizierter ES-Hamiltonian mit D_ES, Zeeman und Orbitsplitting"""
    S = spin_ops()
    I3 = jnp.eye(3)
    # Zero‐Field Splitting ES
    H_zfs_es = 2*jnp.pi * p.D_ES_Hz * (S["Sz"]@S["Sz"] - (2/3)*I3)
    # Zeeman wie im GS
    gamma = p.g_e * p.MU_B / p.HBAR
    H_zeeman_es = 2*jnp.pi * gamma * (B[0]*S["Sx"] + B[1]*S["Sy"] + B[2]*S["Sz"])
    # Spin-Orbit Splitting (grobschlächtig als konstantes Pauli‐Term)
    H_so = 2*jnp.pi * p.SpinOrbitSplitting_Hz * (S["Sz"]@S["Sz"])
    # Strain/E-Feld im ES
    H_strain_es = 2*jnp.pi * p.ES_strain_Hz * (S["Sx"]@S["Sx"] - S["Sy"]@S["Sy"])
    # Detuning in ES
    H_delta_es = 2*jnp.pi * delta_es * (S["Sz"]@S["Sz"])
    return H_zfs_es + H_zeeman_es + H_so + H_strain_es + H_delta_es
```

*Einbindung:* Baue deinen Gesamt-Hamiltonian aus `H_gs` und `H_es` abhängig vom elektronischen Zustand (Ground vs. Excited).

---

## 3. Intersystem Crossing (ISC)

**Was fehlt?**
– Explizite Singulett‐Manifold‐Zustände im Lindblad‐Ansatz

**Ergänzung:**

```python
@jit
def collapse_ISC(p: NVParams):
    """Erzeuge Kollapsoperatoren für ISC-Übergänge"""
    S = spin_ops()
    # Projektoren auf ES ms=±1 und ms=0
    P_es0 = ...   # 3×3 Projektor auf |es,ms=0>
    P_es1 = ...   # 3×3 Projektor auf |es,ms=±1>
    # Radioaktive Zerfallspfad ES→GS
    c_em0    = jnp.sqrt(p.gamma_rad)*P_es0  
    c_em1    = jnp.sqrt(p.gamma_rad)*P_es1  
    # ISC Pfade ES→Singulett
    k_ISC0   = arrhenius(p.k_ISC0_ref, p.Ea_ISC0, p.temp_K)
    k_ISC1   = arrhenius(p.k_ISC1_ref, p.Ea_ISC1, p.temp_K)
    c_ISC0   = jnp.sqrt(k_ISC0) * P_es0_to_sing
    c_ISC1   = jnp.sqrt(k_ISC1) * P_es1_to_sing
    # Singulett → GS Rückkopplung
    k_sing   = 1/(p.tau_singlet_ns*1e-9)
    c_sing   = jnp.sqrt(k_sing) * P_sing_to_gs0
    return [c_em0, c_em1, c_ISC0, c_ISC1, c_sing]
```

*Einbindung:* Füge diese Collapse‐Ops in deine `lindblad_rhs` ein.

---

## 4. Spin-Relaxation T₁

**Was fehlt?**
– Dynamische T₁‐Relaxation mit exponentiellem Decay im Master‐Equation

**Ergänzung:**

```python
@jit
def collapse_T1(p: NVParams):
    P0, P1, Pm1 = ...  # Projektoren auf ms=0,±1 im GS
    k_T1 = 1.0 / (p.T1_ms*1e-3)
    # ms=±1 → ms=0
    c_T1p = jnp.sqrt(k_T1) * (P0 @ P1)  
    c_T1m = jnp.sqrt(k_T1) * (P0 @ Pm1)
    return [c_T1p, c_T1m]
```

Damit fügt deine Dichtematrix‐Propagator die thermische Relaxation sauber über Lindblad-Term ein.

---

## 5. Spin-Kohärenz T₂\*

**Was fehlt?**
– Pure Dephasierung als Lindblad‐Operator

**Ergänzung:**

```python
@jit
def collapse_T2(p: NVParams):
    S = spin_ops()
    gamma_phi = 1.0 / (p.T2_star_us*1e-6)
    return [jnp.sqrt(gamma_phi) * S["Sz"]]
```

So entsteht ein exponentieller Dephasierungs‐Decay zwischen ms‐Zuständen, ohne Populationswechsel.

---

## 6. Laseranregung (Spektral & Polarisations-Profil)

**Was fehlt?**
– Laser-Linienprofil (Lorentz/Voigt) und Strahlprofil

**Ergänzung:**

```python
@jit
def laser_rate(p: NVParams, I_rel: float, detuning: float):
    # Voigt‐Approximation: Lorentz‐Broadened Linienprofil
    gamma_L = p.laser_linewidth_MHz * 1e6
    lorentz = 1.0 / (1 + (2*detuning/gamma_L)**2)
    # Sättigung
    sat = I_rel / (I_rel + p.I_sat_mW)
    return p.gamma_laser * sat * lorentz
```

Ersetze dein einfaches Sättigungsmodell durch obiges Profil. Füge Polarisations‐Winkelabhängigkeit hinzu, indem du `I_rel *= cos²(θ)` je nach NV-Orientierung.

---

## 7. Photonen-Emission (ZPL vs PSB)

**Was fehlt?**
– Beide Spektren explizit simulieren und ggf. Spektren‐Filter

**Ergänzung:**

```python
@jit
def photon_emission_rates(p: NVParams, pop_es: float, T_K: float):
    # DW‐Faktor temperaturabhängig
    DW = debye_waller_factor(T_K, p)
    gamma_rad = p.gamma_rad * 1e9  # Hz
    rate_zpl = DW * gamma_rad * pop_es
    rate_psb = (1-DW)*gamma_rad * pop_es
    return rate_zpl, rate_psb
```

In deinem Detektor‐Modul kannst du dann diese beiden Raten unterschiedlich gewichten (z.B. nur ZPL mit IRF falten, PSB stumpf zählen).

---

## 8. MW-Spinmanipulation (Bloch-Gleichung)

**Was fehlt?**
– Zeit­aufgelöste Integration der Bloch-Gleichung statt statischer Rotation

**Ergänzung:**

```python
@jit
def bloch_step(rho, H_mw, dt):
    """Löse dρ/dt = -i[H_mw,ρ] + Dephasierung"""
    return rho + dt * (-1j*(H_mw@rho - rho@H_mw) 
                       + sum(c@rho@c.T.conj() - 0.5*(c.T.conj()@c@rho+rho@c.T.conj()@c)
                             for c in collapse_T2(params)))
```

Baue so eine sub-Zeitschleife für den MW-Puls, statt eines einzigen `U = exp(-iθSx)`.

---

## 9. Spektrale Diffusion (OU-Prozess)

**Was fehlt?**
– Anwendung nicht nur während Puls, sondern auch während Readout

**Ergänzung:**

```python
@jit
def ou_update(delta_prev, dt, tau, sigma):
    alpha = dt / tau
    noise = random.normal(key) * jnp.sqrt(2*alpha)*sigma
    return delta_prev*(1-alpha) + noise

# In deiner Hauptschleife:
delta = 0.0
for each timestep dt:
    delta = ou_update(delta, dt, p.specdiff_tau_corr_ns, p.specdiff_sigma_MHz)
    H = H_gs(p, B, strain, delta)  # aktualisiertes δ überall verwenden
```

---

## 10. NV-Ladungszustände

**Was fehlt?**
– Master-Equation für NV⁻ ↔ NV⁰

**Ergänzung:**

```python
@jit
def collapse_charge(p: NVParams):
    # projektor NV- und NV0-Zustand in erweitertem Rho
    P_minus = ...
    P_zero  = ...
    k_ion  = p.ionization_rate_ES_MHz*1e6  # im ES
    k_rec  = p.recombination_rate_MHz*1e6
    c_ion  = jnp.sqrt(k_ion) * (P_zero @ P_minus)
    c_rec  = jnp.sqrt(k_rec) * (P_minus @ P_zero)
    return [c_ion, c_rec]
```

Führe eine 4×4-Dichtematrix (GS-Triplet + NV⁰) statt rein 3×3.

---

## 11. Detektor-Modell (IRF, Dead-Time, Afterpulse)

**Was fehlt?**
– Ereignisweises Sampling und Faltung mit IRF

**Ergänzung:**

```python
def detect_photon_events(times, rates, params):
    events = []
    for i, t in enumerate(times):
        # Poisson‐Zufall
        n = np.random.poisson(rates[i]*params.dt)
        for _ in range(n):
            # sample IRF-Verzögerung
            dt_irf = np.random.normal(0, params.irf_sigma_ps*1e-3)
            if np.random.rand() < params.irf_tail_frac:
                dt_irf += np.random.exponential(params.irf_tail_tau_ns)
            events.append(t + dt_irf)
    # sortiere und wende Dead-Time & Afterpulse an...
    events = apply_deadtime_afterpulse(events, params)
    return events
```

Verwende `events` dann, um Histogramme in beliebigen Bins zu erstellen.

---

## 12. Photon-Sampling & Umweltrauschen

**Was fehlt?**
– Vektorisierte Monte-Carlo für viele Pulse, OU auf B und Strain

**Ergänzung:**

```python
# vmap-fähige Funktion für einen einzelnen Puls:
@partial(jit, static_argnums=(1,))
def simulate_one_pulse(key, p: NVParams):
    # OU für B- und Strain-Noise
    Bz, delta_strain = 0.0, 0.0
    trace = []
    for step in jnp.arange(0, p.READ_NS, p.BIN_ns):
        key, k1, k2 = random.split(key, 3)
        Bz = ou_update(Bz, p.B_noise_dt, p.B_noise_tau, p.B_noise_sigma)
        delta_strain = ou_update(delta_strain, p.strain_dt, p.strain_tau, p.strain_sigma)
        H = H_gs(p, jnp.array([0,0,Bz]), delta_strain, 0.0)
        # berechne Photonrate mit aktuellen Parametern...
        rate = photon_rate_from_rho(rho, p)
        counts = random.poisson(k1, rate*p.BIN_ns*1e-9)
        trace.append(counts)
    return jnp.array(trace)

# Für N Pulse:
keys = random.split(random.PRNGKey(0), p.n_pulses)
all_traces = vmap(lambda k: simulate_one_pulse(k, p))(keys)
mean_trace = jnp.mean(all_traces, axis=0)
```

---

### So integrierst du Schritt für Schritt:

1. **Erweitere** deine `NVParams` um alle neuen Parameterfelder (z.B. `D_ES_Hz`, `A_para_N_Hz`, `laser_linewidth_MHz` …).
2. **Ersetze** in deinem Code den vereinfachten Hamiltonian durch `H_gs` und `H_es`.
3. **Baue** in `collapse_ops` neue Funktionen `collapse_T1`, `collapse_T2`, `collapse_ISC`, `collapse_charge` ein und füge sie zum Lindblad-Term hinzu.
4. **Führe** in deiner Propagationsschleife (MW-Puls, Readout) echte Zeitintegration mit `bloch_step` statt statischer Rotationen ein.
5. **Simuliere** Photon-Arrival-Events mit `detect_photon_events` und wandle sie in Bins oder Korrelationsfunktionen um.
6. **Verwende** `vmap` und `jit` für große Shot‐Arrays, OU-Updates für B- und Strain-Noise, um Performance auf GPU/TPU zu sichern.

Mit diesen Snippets deckst du **alle** Punkte aus der Tabelle ab und machst deinen Simulator zu einem **vollständigen, hyper-realistischen NV-Zentrum-Modell**. Viel Erfolg beim Integrieren!
