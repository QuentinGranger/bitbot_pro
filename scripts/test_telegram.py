#!/usr/bin/env python3
"""
Script de test pour les notifications Telegram.
"""
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType

# Charger les variables d'environnement
load_dotenv()

# Configurer le logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Fonction principale du test."""
    logger.info("Démarrage du test Telegram...")
    
    # Vérifier la configuration Telegram
    if notification_manager.telegram_notifier:
        logger.info(f"Telegram configuré avec {len(notification_manager.telegram_notifier.chat_ids)} chat IDs")
        logger.info(f"Token: {notification_manager.telegram_notifier.token[:5]}...{notification_manager.telegram_notifier.token[-5:]}")
        logger.info(f"Chat IDs: {notification_manager.telegram_notifier.chat_ids}")
    else:
        logger.error("Telegram n'est pas configuré!")
        logger.info("Vérifiez vos variables d'environnement TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_IDS")
        return

    # Envoyer un message de test
    logger.info("Envoi d'un message test...")
    success = await notification_manager.notify(
        message="🔔 Ceci est un message de test depuis BitBotPro",
        priority=NotificationPriority.MEDIUM,
        notification_type=NotificationType.SYSTEM,
        title="Test de Notification",
        details={
            "environnement": "test",
            "timestamp": "maintenant",
            "version": "1.0.0"
        }
    )
    
    if success:
        logger.info("✅ Message envoyé avec succès!")
    else:
        logger.error("❌ Échec de l'envoi du message!")
    
    logger.info("Test terminé")

if __name__ == "__main__":
    asyncio.run(main())
