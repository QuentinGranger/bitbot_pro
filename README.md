# BitBot Pro

Bot de trading automatique avec analyse technique et intelligence artificielle.

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/QuentinGranger/bitbot_pro.git
cd bitbot_pro
```

2. Créer un environnement virtuel Python :
```bash
python3.11 -m venv venv
source venv/bin/activate  # Sur Unix/MacOS
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du Projet

- `requirements.txt` : Liste des dépendances Python
- `.env` : Configuration des variables d'environnement (non versionné)
- `.gitignore` : Liste des fichiers ignorés par Git

## Structure des Données

Les données sont gérées avec DVC (Data Version Control) :

- `data/raw/` : Données brutes téléchargées
- `data/processed/` : Données nettoyées et préparées
- `data/models/` : Modèles entraînés et leurs métriques

### Gestion des Données avec DVC

Pour travailler avec les données :

```bash
# Télécharger les données
dvc pull

# Ajouter de nouvelles données
dvc add data/raw/nouveau_dataset.csv

# Pousser les modifications vers le stockage distant
dvc push
```

## Technologies Utilisées

- Python 3.11
- NumPy & Pandas pour l'analyse de données
- TA-Lib pour l'analyse technique
- TensorFlow pour le machine learning
- Dash & Plotly pour le dashboard
- Websockets pour la communication en temps réel
