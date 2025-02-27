"""
Configuration du système de journalisation pour BitBot Pro.
Utilise loguru pour une gestion avancée des logs avec rotation des fichiers
et formatage personnalisé.
"""

import sys
from pathlib import Path
from loguru import logger

# Configuration des niveaux de log personnalisés
TRADE_LEVEL_NO = 25  # Entre INFO et WARNING
EXCHANGE_LEVEL_NO = 15  # Entre DEBUG et INFO

# Ajout des niveaux personnalisés
logger.level("TRADE", no=TRADE_LEVEL_NO, color="<cyan>")
logger.level("EXCHANGE", no=EXCHANGE_LEVEL_NO, color="<blue>")

def setup_logger(log_dir: str = "logs"):
    """
    Configure le logger avec rotation des fichiers et formats personnalisés.
    
    Args:
        log_dir: Chemin vers le dossier des logs
    """
    # Création du dossier de logs s'il n'existe pas
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Format personnalisé pour les logs
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Configuration des handlers
    config = {
        "handlers": [
            # Console - tous les logs à partir de INFO
            {
                "sink": sys.stderr,
                "format": log_format,
                "level": "INFO",
                "colorize": True,
            },
            # Fichier général - tous les logs
            {
                "sink": f"{log_dir}/bitbot.log",
                "format": log_format,
                "level": "DEBUG",
                "rotation": "100 MB",
                "retention": "1 week",
                "compression": "zip",
            },
            # Fichier d'erreurs - ERROR et CRITICAL
            {
                "sink": f"{log_dir}/errors.log",
                "format": log_format,
                "level": "ERROR",
                "rotation": "100 MB",
                "retention": "1 month",
                "compression": "zip",
            },
            # Fichier de trades - niveau TRADE et supérieur
            {
                "sink": f"{log_dir}/trades.log",
                "format": log_format,
                "level": "TRADE",
                "filter": lambda record: record["level"].no >= TRADE_LEVEL_NO,
                "rotation": "100 MB",
                "retention": "1 month",
                "compression": "zip",
            },
        ],
    }
    
    # Configuration du logger
    logger.configure(**config)

# Fonctions d'aide pour la journalisation
def log_trade(message: str, **kwargs):
    """Log un événement lié au trading."""
    logger.log("TRADE", message, **kwargs)

def log_exchange(message: str, **kwargs):
    """Log un événement lié à l'exchange."""
    logger.log("EXCHANGE", message, **kwargs)

def log_startup():
    """Log le démarrage de l'application."""
    logger.info("🚀 Démarrage de BitBot Pro")

def log_shutdown():
    """Log l'arrêt de l'application."""
    logger.info("🛑 Arrêt de BitBot Pro")

def log_error(error: Exception, context: str = ""):
    """Log une erreur avec son contexte."""
    if context:
        logger.error(f"{context}: {str(error)}")
    else:
        logger.error(str(error))
    logger.exception(error)

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration initiale
    setup_logger()
    
    # Exemples de logs
    log_startup()
    
    try:
        # Simulation d'événements
        logger.debug("Chargement de la configuration...")
        log_exchange("Connexion à Binance établie")
        logger.info("Bot prêt pour le trading")
        
        log_trade("Nouvel ordre : ACHAT BTC/USDT @ 50000")
        log_trade("Ordre exécuté : 0.1 BTC @ 50000 USDT")
        
        # Simulation d'une erreur
        raise ConnectionError("Perte de connexion à l'exchange")
    
    except Exception as e:
        log_error(e, "Erreur lors du trading")
    
    finally:
        log_shutdown()
