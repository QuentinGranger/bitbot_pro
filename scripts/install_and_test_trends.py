#!/usr/bin/env python3
"""
Script pour installer et tester l'intégration de Google Trends.
"""

import os
import sys
import subprocess
import tempfile
import venv
import shutil
from pathlib import Path

def main():
    """
    Crée un environnement virtuel, installe les dépendances nécessaires
    et teste l'intégration de Google Trends.
    """
    # Créer un dossier temporaire pour l'environnement virtuel
    temp_dir = tempfile.mkdtemp()
    venv_dir = Path(temp_dir) / "venv"
    
    try:
        print(f"Création d'un environnement virtuel temporaire dans {venv_dir}...")
        venv.create(venv_dir, with_pip=True)
        
        # Obtenir le chemin vers pip dans l'environnement virtuel
        if sys.platform == 'win32':
            pip_path = venv_dir / "Scripts" / "pip"
            python_path = venv_dir / "Scripts" / "python"
        else:
            pip_path = venv_dir / "bin" / "pip"
            python_path = venv_dir / "bin" / "python"
        
        # Installer les dépendances
        print("Installation des dépendances...")
        subprocess.run([str(pip_path), "install", "pytrends", "pandas", "matplotlib", "seaborn"], check=True)
        
        # Charger le chemin du projet
        project_dir = Path(__file__).parent.parent
        
        # Créer un script de test simple
        test_script = Path(temp_dir) / "test_pytrends.py"
        with open(test_script, "w") as f:
            f.write("""
import pandas as pd
from pytrends.request import TrendReq

# Configuration du client pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Requête simple pour vérifier que tout fonctionne
pytrends.build_payload(kw_list=['Bitcoin'])
interest_over_time_df = pytrends.interest_over_time()

# Afficher les résultats
print(interest_over_time_df.head())
            """)
        
        # Exécuter le script de test
        print("Test de l'installation de pytrends...")
        subprocess.run([str(python_path), str(test_script)], check=True)
        
        print("\nPytrends fonctionne correctement ! Voici les étapes à suivre :")
        print("1. Assurez-vous que pytrends est bien installé dans votre environnement principal:")
        print("   pip3 install --upgrade pytrends --user")
        print("2. Vérifiez que le PYTHONPATH inclut bien le dossier des modules Python:")
        print("   export PYTHONPATH=$PYTHONPATH:/Users/user/Library/Python/3.11/lib/python/site-packages")
        print("3. Relancez ensuite le script de test Google Trends:")
        print("   python3 scripts/test_google_trends.py")
        
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")
    
    finally:
        # Nettoyer l'environnement virtuel
        print(f"Nettoyage de l'environnement temporaire...")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
