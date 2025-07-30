# Laser-Photonenzähler Test

Dieses Verzeichnis enthält Test-Scripts für den NV-Zentrum Simulator mit Laser und Photonenzähler.

## Dateien

- `laser_photon_test.py` - Haupttest-Script
- `run_test.sh` - Ausführungs-Script
- `photon_counts_*.txt` - Ausgabedateien mit Photonenzählungen
- `photon_counts_summary.txt` - Zusammenfassung aller Tests

## Verwendung

### Standardtest (300ns, 1mW):
```bash
./run_test.sh
```

### Eigene Parameter:
```bash
./run_test.sh 500 2.0  # 500ns, 2mW Laser
```

## Ausgabe

Die Photonenzählungen werden in einer Datei mit Zeitstempel gespeichert:
- `photon_counts_YYYYMMDD_HHMMSS.txt`

Format:
```
Zeit[ns]  |g⟩    |e⟩    EmRate[MHz]  Photonen  Total
    1.0  0.9950 0.0050      50.00         0       0
    2.0  0.9803 0.0197     197.00         0       0
    ...
```

## Physik

Der Test simuliert:
1. NV-Zentrum startet im Grundzustand |g⟩
2. Laser treibt Rabi-Oszillationen zwischen |g⟩ und |e⟩
3. Spontane Emission führt zu Photonenemission
4. Photonenzähler detektiert Photonen mit 10% Effizienz

Erwartete Ergebnisse:
- Bei 1mW: ~50-100 MHz Emissionsrate
- In 300ns: ~1-3 detektierte Photonen (mit 10% Effizienz)