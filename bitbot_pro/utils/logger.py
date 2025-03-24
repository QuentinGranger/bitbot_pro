"""
Module de journalisation pour BitBot Pro utilisant loguru.
Fournit des fonctionnalités de journalisation structurées et personnalisables.
"""
import sys
import os
from datetime import datetime
from pathlib import Path
from loguru import logger

# Obtenir le chemin absolu du répertoire logs
LOGS_DIR = Path(__file__).parents[2] / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class BitBotLogger:
    """
    Classe de gestion de la journalisation pour BitBot Pro.
    Utilise loguru pour fournir des logs colorés, structurés et détaillés.
    """
    
    # Niveaux de logs personnalisés
    LEVELS = {
        "DEBUG": {"color": "<cyan>"},
        "INFO": {"color": "<green>"},
        "WARNING": {"color": "<yellow>"},
        "ERROR": {"color": "<red>"},
        "CRITICAL": {"color": "<RED><bold>"},
        "TRADE": {"color": "<magenta>", "no": 25},  # Niveau personnalisé pour les opérations de trading
        "API": {"color": "<blue>", "no": 26},       # Niveau personnalisé pour les communications API
    }
    
    def __init__(self, log_level="INFO", enable_console=True, enable_file=True, rotation="1 day", retention="30 days"):
        """
        Initialise le logger BitBot Pro.
        
        Args:
            log_level (str): Niveau de log minimum (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console (bool): Activer la sortie console
            enable_file (bool): Activer la journalisation dans des fichiers
            rotation (str): Fréquence de rotation des fichiers logs ("1 day", "100 MB", etc.)
            retention (str): Durée de conservation des fichiers logs ("30 days", "5 weeks", etc.)
        """
        # Supprimer les handlers par défaut
        logger.remove()
        
        # Ajouter les niveaux personnalisés
        for level_name, params in self.LEVELS.items():
            if level_name not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.level(level_name, **params)
        
        # Format détaillé pour tous les handlers
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Handler pour la console
        if enable_console:
            logger.add(
                sys.stderr,
                format=log_format,
                level=log_level,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )
        
        # Handler pour les fichiers
        if enable_file:
            # Fichier général de logs
            logger.add(
                LOGS_DIR / "bitbot_{time}.log",
                format=log_format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )
            
            # Fichier spécifique pour les erreurs
            logger.add(
                LOGS_DIR / "errors_{time}.log",
                format=log_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
                filter=lambda record: record["level"].name in ["ERROR", "CRITICAL"]
            )
            
            # Fichier spécifique pour les trades
            logger.add(
                LOGS_DIR / "trades_{time}.log",
                format=log_format,
                level="TRADE",
                rotation=rotation,
                retention=retention,
                compression="zip",
                filter=lambda record: record["level"].name == "TRADE"
            )
    
    @staticmethod
    def get_logger():
        """
        Retourne l'instance globale du logger.
        
        Returns:
            loguru.logger: L'instance du logger configurée
        """
        return logger


# Exemples d'utilisation:
def log_exchange_connection(exchange_name, status, details=None):
    """
    Journalise une connexion à un exchange.
    
    Args:
        exchange_name (str): Nom de l'exchange
        status (bool): État de la connexion (True=succès, False=échec)
        details (dict, optional): Détails supplémentaires
    """
    if status:
        logger.info(f"Connexion réussie à l'exchange {exchange_name}")
    else:
        logger.error(f"Échec de connexion à l'exchange {exchange_name}: {details}")


def log_order_execution(exchange, symbol, order_type, side, amount, price=None):
    """
    Journalise l'exécution d'un ordre de trading.
    
    Args:
        exchange (str): Nom de l'exchange
        symbol (str): Paire de trading (ex: BTC/USDT)
        order_type (str): Type d'ordre (market, limit, etc.)
        side (str): Direction (buy, sell)
        amount (float): Quantité
        price (float, optional): Prix pour les ordres limit
    """
    price_str = f" à {price}" if price else ""
    logger.log("TRADE", f"[{exchange}] {side.upper()} {order_type} {amount} {symbol}{price_str}")


def log_system_error(component, error_msg, exception=None):
    """
    Journalise une erreur système.
    
    Args:
        component (str): Composant concerné
        error_msg (str): Message d'erreur
        exception (Exception, optional): Exception Python
    """
    logger.error(f"Erreur dans {component}: {error_msg}")
    if exception:
        logger.exception(exception)


def log_network_issue(service, error_msg, retry_count=0):
    """
    Journalise un problème réseau.
    
    Args:
        service (str): Service ou API concerné
        error_msg (str): Message d'erreur
        retry_count (int, optional): Nombre de tentatives
    """
    if retry_count > 0:
        logger.warning(f"Problème réseau avec {service} (tentative #{retry_count}): {error_msg}")
    else:
        logger.warning(f"Problème réseau avec {service}: {error_msg}")


# Initialisation globale du logger
# Peut être personnalisé selon les besoins spécifiques du projet
bitbot_logger = BitBotLogger(
    log_level="DEBUG",  # En développement, utiliser DEBUG
    enable_console=True,
    enable_file=True,
    rotation="1 day",
    retention="30 days"
)

# Exporter l'instance du logger pour utilisation dans d'autres modules
logger = bitbot_logger.get_logger()

# Usage dans d'autres modules:
# from bitbot_pro.utils.logger import logger
# logger.debug("Message de debug")
# logger.info("Information importante")
# logger.warning("Attention!")
# logger.error("Erreur critique")
# logger.log("TRADE", "Exécution d'un ordre de trading")
# logger.log("API", "Communication avec une API externe")
