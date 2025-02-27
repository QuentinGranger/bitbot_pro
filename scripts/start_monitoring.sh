#!/bin/bash
# Script pour démarrer la surveillance en temps réel de BitBotPro en arrière-plan

# Chemin vers le répertoire principal
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/.."
cd "$DIR"

# Par défaut
SYMBOL="BTCUSDT"
TIMEFRAME="5m"
INTERVAL=5
LOG_FILE="logs/monitoring.log"

# Lire les arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --timeframe)
      TIMEFRAME="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Option inconnue: $1"
      exit 1
      ;;
  esac
done

# Créer le répertoire de logs si nécessaire
mkdir -p "$(dirname "$LOG_FILE")"

echo "Démarrage de la surveillance pour $SYMBOL (timeframe: $TIMEFRAME, intervalle: $INTERVAL minutes)"
echo "Les logs seront écrits dans $LOG_FILE"

# Démarrer le script en arrière-plan
nohup python3 scripts/run_live_monitoring.py --symbol "$SYMBOL" --timeframe "$TIMEFRAME" --interval "$INTERVAL" >> "$LOG_FILE" 2>&1 &

# Récupérer le PID
PID=$!
echo "Le processus de surveillance est démarré avec PID: $PID"
echo $PID > "logs/monitoring.pid"

echo "Pour arrêter la surveillance, exécutez: kill $PID"
