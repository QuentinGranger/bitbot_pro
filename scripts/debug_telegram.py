#!/usr/bin/env python3
"""
Script de diagnostic pour le module telegram_alerts.
Ce script teste les fonctionnalités du module sans dépendre du reste de l'application.
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Charger explicitement les variables d'environnement
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Variables d'environnement chargées depuis {env_path}")
else:
    logger.error(f"Fichier .env non trouvé à {env_path}")
    sys.exit(1)

# Tester les variables d'environnement
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
telegram_chat_ids = os.environ.get("TELEGRAM_CHAT_IDS")

if not telegram_token:
    logger.error("TELEGRAM_BOT_TOKEN n'est pas défini dans le fichier .env")
    sys.exit(1)

if not telegram_chat_ids:
    logger.error("TELEGRAM_CHAT_IDS n'est pas défini dans le fichier .env")
    sys.exit(1)

logger.info(f"TELEGRAM_BOT_TOKEN: {telegram_token[:5]}...{telegram_token[-5:]}")
logger.info(f"TELEGRAM_CHAT_IDS: {telegram_chat_ids}")

# Importer et initialiser le module telegram_alerts
try:
    from bitbot.utils.telegram_alerts import (
        telegram_alerts, send_telegram_alert, AlertType, AlertPriority
    )
    logger.info("Module telegram_alerts importé avec succès")
    logger.info(f"TelegramAlertManager initialisé: {telegram_alerts.is_configured}")
    logger.info(f"Chat IDs configurés: {telegram_alerts.chat_ids}")
except Exception as e:
    logger.error(f"Erreur lors de l'importation ou de l'initialisation: {type(e).__name__}: {e}")
    sys.exit(1)

async def main():
    """Fonction principale du diagnostic."""
    logger.info("Démarrage du diagnostic des alertes Telegram...")
    
    # Tester l'envoi d'une alerte simple
    try:
        logger.info("Envoi d'une alerte de test...")
        success = await send_telegram_alert(
            message="📊 Test direct de diagnostic Telegram",
            alert_type=AlertType.SYSTEM,
            priority=AlertPriority.MEDIUM,
            title="Test de diagnostic",
            details={
                "test_id": "diagnostic-001",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "debug": "true"
            }
        )
        logger.info(f"Résultat de l'envoi: {'succès' if success else 'échec'}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'alerte: {type(e).__name__}: {e}")
    
    logger.info("Diagnostic terminé")

if __name__ == "__main__":
    asyncio.run(main())
