#!/bin/bash

# Chemin vers le projet
PROJECT_DIR="/Users/user/Desktop/BitBot Pro"

# Activer l'environnement virtuel
source "$PROJECT_DIR/venv/bin/activate"

# Aller dans le répertoire du projet
cd "$PROJECT_DIR"

# Exécuter le script de mise à jour
python scripts/update_data.py

# Désactiver l'environnement virtuel
deactivate
