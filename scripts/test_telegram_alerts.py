#!/usr/bin/env python3
"""
Script de test pour les alertes Telegram utilisant notre nouveau module.
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.utils.telegram_alerts import send_telegram_alert, AlertType, AlertPriority

# Configurer le logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_alerts():
    """Teste les différents types d'alertes Telegram."""
    # Charger les variables d'environnement
    load_dotenv()
    
    logger.info("Démarrage des tests d'alertes Telegram...")
    
    # Test 1: Alerte système basique
    logger.info("Test 1: Alerte système basique")
    success = await send_telegram_alert(
        message="Test d'alerte système basique",
        alert_type=AlertType.SYSTEM,
        priority=AlertPriority.MEDIUM,
        title="Test Système"
    )
    logger.info(f"Résultat: {'Succès' if success else 'Échec'}")
    
    # Attendre un peu pour éviter le rate limiting
    await asyncio.sleep(1)
    
    # Test 2: Alerte de prix avec détails
    logger.info("Test 2: Alerte de prix avec détails")
    success = await send_telegram_alert(
        message="Prix de BTC en forte hausse!",
        alert_type=AlertType.PRICE,
        priority=AlertPriority.HIGH,
        title="Alerte de Prix",
        details={
            "Symbol": "BTCUSDT",
            "Prix actuel": "68,500.00",
            "Variation": "+5.2%",
            "Délai": "5 minutes",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    logger.info(f"Résultat: {'Succès' if success else 'Échec'}")
    
    # Attendre un peu pour éviter le rate limiting
    await asyncio.sleep(1)
    
    # Test 3: Alerte d'anomalie critique
    logger.info("Test 3: Alerte d'anomalie critique")
    success = await send_telegram_alert(
        message="ANOMALIE GRAVE DÉTECTÉE: Spike de volume extrême sur ETH",
        alert_type=AlertType.ANOMALY,
        priority=AlertPriority.CRITICAL,
        title="Anomalie Critique",
        details={
            "Symbol": "ETHUSDT",
            "Volume": "10x au-dessus de la moyenne",
            "Niveau": "CRITIQUE",
            "Action requise": "Vérification manuelle recommandée"
        }
    )
    logger.info(f"Résultat: {'Succès' if success else 'Échec'}")
    
    logger.info("Tests terminés")

if __name__ == "__main__":
    asyncio.run(test_alerts())
