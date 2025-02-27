#!/usr/bin/env python3
"""
Script principal de lancement du BitBot Pro avec watchdog et mécanismes de reprise.
"""

import os
import sys
import argparse
from pathlib import Path
import signal
import asyncio
from typing import Optional
import json

from bitbot.utils.logger import setup_logger, logger
from bitbot.utils.watchdog import run_watchdog

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="BitBot Pro Trading Bot")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Mode d'exécution du bot"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug"
    )
    
    parser.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Désactive le watchdog"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    Charge la configuration du bot.
    
    Args:
        config_path: Chemin vers le fichier de configuration
    
    Returns:
        Configuration sous forme de dictionnaire
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration chargée depuis {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        sys.exit(1)

def setup_environment(config: dict, mode: str):
    """
    Configure l'environnement d'exécution.
    
    Args:
        config: Configuration du bot
        mode: Mode d'exécution (live, paper, backtest)
    """
    # Définir les variables d'environnement nécessaires
    os.environ["BITBOT_MODE"] = mode
    os.environ["BITBOT_DEBUG"] = "1" if args.debug else "0"
    
    # Configurer le logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    
    # Afficher la configuration de démarrage
    logger.info("=== BitBot Pro ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Debug: {'activé' if args.debug else 'désactivé'}")
    logger.info(f"Watchdog: {'désactivé' if args.no_watchdog else 'activé'}")

def run_directly():
    """Lance le bot directement sans watchdog."""
    try:
        # Import différé pour éviter les imports circulaires
        from bitbot.main import main
        
        # Créer et démarrer la boucle d'événements
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        raise
    finally:
        # Nettoyage
        loop.close()

def main():
    """Point d'entrée principal."""
    global args
    args = parse_args()
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Configurer l'environnement
    setup_environment(config, args.mode)
    
    # Obtenir les chemins absolus
    project_root = Path(__file__).parent
    main_script = project_root / "bitbot" / "main.py"
    
    if args.no_watchdog:
        # Lancement direct
        run_directly()
    else:
        # Lancement avec watchdog
        run_watchdog(
            bot_script=str(main_script),
            cwd=str(project_root)
        )

if __name__ == "__main__":
    main()
