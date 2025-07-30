#!/bin/bash
# Führt Laser-Photonenzähler Test aus

echo "Laser-Photonenzähler Test"
echo "========================"
echo ""
echo "Standardtest: 300ns, 1mW Laser"
echo "Oder mit Parametern: ./run_test.sh <dauer_ns> <leistung_mw>"
echo ""

# Standard oder übergebene Parameter
DURATION=${1:-300}
POWER=${2:-1.0}

echo "Starte Test mit:"
echo "  Dauer: ${DURATION}ns"
echo "  Laserleistung: ${POWER}mW"
echo ""

# Python-Script ausführen
python3 laser_photon_test.py $DURATION $POWER

# Zeige letzte Zeilen der Ausgabe
echo ""
echo "Letzte Einträge der Photonenzählung:"
tail -n 10 photon_counts_*.txt | grep -v "^#"