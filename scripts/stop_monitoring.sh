#!/bin/bash
# Script pour arrêter la surveillance en temps réel de BitBotPro

# Chemin vers le répertoire principal
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/.."
cd "$DIR"

PID_FILE="logs/monitoring.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # Vérifier si le processus existe encore
    if ps -p $PID > /dev/null; then
        echo "Arrêt du processus de surveillance (PID: $PID)..."
        kill $PID
        echo "Signal d'arrêt envoyé. Le processus devrait se terminer proprement."
    else
        echo "Le processus (PID: $PID) n'est plus en cours d'exécution."
    fi
    
    # Supprimer le fichier PID
    rm "$PID_FILE"
else
    echo "Aucun processus de surveillance en cours d'exécution (fichier PID non trouvé)."
fi
