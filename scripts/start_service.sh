#!/bin/bash

# Chemin vers le projet
PROJECT_DIR="/Users/user/Desktop/BitBot Pro"

# Créer le dossier de logs s'il n'existe pas
mkdir -p "$PROJECT_DIR/logs"

# Activer l'environnement virtuel
source "$PROJECT_DIR/venv/bin/activate"

# Démarrer le service de données en temps réel en arrière-plan
python scripts/realtime_data.py >> "$PROJECT_DIR/logs/realtime.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/realtime.pid"

# Configurer la tâche cron pour la mise à jour des données historiques
(crontab -l 2>/dev/null; echo "*/5 * * * * $PROJECT_DIR/scripts/update_data.sh >> $PROJECT_DIR/logs/update.log 2>&1") | crontab -

echo "Services démarrés :"
echo "1. Flux temps réel en cours d'exécution (PID dans logs/realtime.pid)"
echo "2. Mise à jour historique configurée (toutes les 5 minutes)"
echo "Logs disponibles dans le dossier logs/"
