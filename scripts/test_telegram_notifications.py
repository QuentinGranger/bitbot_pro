#!/usr/bin/env python
"""
Script de test pour les notifications Telegram.
Configure et envoie différents types de notifications pour vérifier leur bon fonctionnement.
"""
import os
import asyncio
import random
import time
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Ajouter le répertoire parent au PATH pour les imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.utils.notifications import (
    notification_manager, NotificationPriority, NotificationType,
    test_notifications
)
from bitbot.utils.anomaly_detector import AnomalyType, Anomaly
from bitbot.utils.logger import setup_logger, logger


async def test_simple_notification(token, chat_id):
    """Test simple d'envoi d'un message."""
    from telegram import Bot
    from telegram.constants import ParseMode
    
    try:
        bot = Bot(token=token)
        await bot.initialize()  # Initialiser le bot avant de l'utiliser
        
        # Formater le token en toute sécurité pour ne pas révéler de données sensibles
        token_preview = token[:5] + "..." + token[-5:]
        
        message = (
            "🔍 *Test de connexion BitBotPro* 🔍\n\n"
            "Si vous voyez ce message, la connexion avec votre bot Telegram "
            "fonctionne correctement!\n\n"
            f"📅 *Date et heure*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🤖 *Bot Token*: `{token_preview}`\n"
            f"👤 *Chat ID*: `{chat_id}`"
        )
        
        print(f"Tentative d'envoi d'un message de test au Chat ID: {chat_id}")
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )
        print("✅ Message envoyé avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message: {str(e)}")
        if "can't parse entities" in str(e).lower():
            print("Problème de formatage du message. Tentative avec un format simplifié...")
            try:
                # Essayer un message sans formatage
                simple_message = f"Test de connexion BitBotPro - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                await bot.send_message(chat_id=chat_id, text=simple_message)
                print("✅ Message simplifié envoyé avec succès!")
                return True
            except Exception as e2:
                print(f"❌ Erreur lors de l'envoi du message simplifié: {str(e2)}")
        
        if "chat not found" in str(e).lower():
            print("\n⚠️ Le Chat ID est incorrect ou le bot n'a pas été démarré.")
            print("Assurez-vous d'avoir:")
            print("1. Trouvé votre bot dans Telegram (pas BotFather)")
            print("2. Démarré une conversation avec lui en envoyant /start")
            print("3. Obtenu votre vrai Chat ID via @username_to_id_bot")
        return False


async def simulate_connection_notifications():
    """Simule une série de notifications liées aux connexions."""
    logger.info("Simulation des notifications de connexion...")
    
    # Déconnexion initiale
    await notification_manager.notify_connection_issue(
        message="Connexion WebSocket interrompue",
        details={
            "Heure de déconnexion": datetime.now().strftime('%H:%M:%S'),
            "Streams actifs": 12,
            "Durée de connexion": "562s"
        },
        critical=False
    )
    
    # Tentatives de reconnexion
    for i in range(3):
        await asyncio.sleep(2)  # Simuler un délai entre les tentatives
        await notification_manager.notify(
            message=f"Tentative de reconnexion {i+1}/5",
            priority=NotificationPriority.MEDIUM,
            notification_type=NotificationType.CONNECTION,
            title="Reconnexion en cours",
            details={
                "Tentative": f"{i+1}/5",
                "Backoff": f"{2**i}s",
                "Streams affectés": 12
            }
        )
    
    # Reconnexion réussie
    await asyncio.sleep(2)
    await notification_manager.notify(
        message=f"Connexion WebSocket rétablie après 14s",
        priority=NotificationPriority.MEDIUM,
        notification_type=NotificationType.CONNECTION,
        title="Reconnexion Réussie",
        details={
            "Tentative": "3/5",
            "Streams réabonnés": 12
        }
    )
    
    # Simuler une déconnexion critique
    await asyncio.sleep(5)
    await notification_manager.notify_connection_issue(
        message="Connexion WebSocket perdue",
        details={
            "Heure de déconnexion": datetime.now().strftime('%H:%M:%S'),
            "Streams actifs": 12,
            "Durée de connexion": "23s"
        },
        critical=False
    )
    
    # Échec de reconnexion
    await asyncio.sleep(3)
    await notification_manager.notify_connection_issue(
        message="CRITIQUE: Échec de toutes les tentatives de reconnexion WebSocket",
        details={
            "Tentatives": 5,
            "Durée de déconnexion": "180s",
            "Streams perdus": 12,
            "Action requise": "Intervention manuelle nécessaire"
        },
        critical=True
    )


async def simulate_anomaly_notifications():
    """Simule une série de notifications d'anomalies."""
    logger.info("Simulation des notifications d'anomalies...")
    
    # Générer différents symboles
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
    
    # 1. Simuler des spikes de volume
    for i in range(3):
        symbol = random.choice(symbols)
        volume = random.uniform(100, 1000)
        avg_volume = random.uniform(20, 100)
        severity = min(1.0, (volume / avg_volume) / 10)
        
        await notification_manager.notify_anomaly(
            anomaly_type="volume_spike",
            symbol=symbol,
            severity=severity,
            message=f"Spike de volume détecté sur {symbol} (×{volume/avg_volume:.1f} la moyenne)",
            details={
                "Volume": f"{volume:.8f}",
                "Volume moyen": f"{avg_volume:.8f}",
                "Ratio": f"{volume/avg_volume:.1f}x",
                "Timestamp": datetime.now().strftime('%H:%M:%S'),
                "Sévérité": f"{severity:.2f}"
            }
        )
        await asyncio.sleep(2)
    
    # 2. Simuler des spikes de prix
    for i in range(3):
        symbol = random.choice(symbols)
        price = random.uniform(10000, 60000) if "BTC" in symbol else random.uniform(100, 3000)
        avg_price = price * (1 - random.uniform(0.05, 0.20))
        percentage = abs((price - avg_price) / avg_price * 100)
        severity = min(1.0, percentage / 30)
        
        await notification_manager.notify_anomaly(
            anomaly_type="price_spike",
            symbol=symbol,
            severity=severity,
            message=f"Spike de prix détecté sur {symbol} (variation de {percentage:.1f}%)",
            details={
                "Prix": f"{price:.8f}",
                "Prix moyen": f"{avg_price:.8f}",
                "Variation": f"{percentage:.1f}%",
                "Timestamp": datetime.now().strftime('%H:%M:%S'),
                "Sévérité": f"{severity:.2f}"
            }
        )
        await asyncio.sleep(2)
    
    # 3. Simuler un gap de données
    symbol = random.choice(symbols)
    gap_duration = random.uniform(30, 300)
    last_timestamp = datetime.now() - timedelta(seconds=gap_duration)
    severity = min(1.0, gap_duration / 300)
    
    await notification_manager.notify_anomaly(
        anomaly_type="data_gap",
        symbol=symbol,
        severity=severity,
        message=f"Gap de données détecté sur {symbol} (durée: {gap_duration:.1f}s)",
        details={
            "Dernier timestamp": last_timestamp.strftime('%H:%M:%S'),
            "Durée du gap": f"{gap_duration:.1f}s",
            "Timestamp actuel": datetime.now().strftime('%H:%M:%S'),
            "Sévérité": f"{severity:.2f}"
        }
    )
    
    # 4. Simuler un spread négatif (anomalie critique)
    symbol = random.choice(symbols)
    ask = random.uniform(10000, 60000) if "BTC" in symbol else random.uniform(100, 3000)
    bid = ask * 1.05  # Bid > Ask (anormal)
    spread = bid - ask
    
    await notification_manager.notify_anomaly(
        anomaly_type="negative_spread",
        symbol=symbol,
        severity="CRITIQUE",
        message=f"Spread négatif détecté sur {symbol} (spread: {spread:.8f})",
        details={
            "Meilleur bid": f"{bid:.8f}",
            "Meilleur ask": f"{ask:.8f}",
            "Spread": f"{spread:.8f}",
            "Timestamp": datetime.now().strftime('%H:%M:%S')
        }
    )


async def main():
    """Fonction principale du script de test."""
    
    parser = argparse.ArgumentParser(description="Test des notifications Telegram")
    parser.add_argument("--token", help="Token du bot Telegram", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", help="ID de chat Telegram", default=os.environ.get("TELEGRAM_CHAT_IDS"))
    parser.add_argument("--simple", action="store_true", help="Effectue uniquement un test simple d'envoi de message")
    parser.add_argument("--connection", action="store_true", help="Simuler des notifications de connexion")
    parser.add_argument("--anomaly", action="store_true", help="Simuler des notifications d'anomalies")
    parser.add_argument("--all", action="store_true", help="Simuler tous les types de notifications")
    
    args = parser.parse_args()
    
    # Configurer le logging
    setup_logger()
    
    # Vérifier si les identifiants Telegram sont fournis
    if not args.token or not args.chat_id:
        print("ERREUR: Token Telegram et/ou ID de chat manquants.")
        print("Utilisez --token et --chat-id ou définissez les variables d'environnement:")
        print("  TELEGRAM_BOT_TOKEN='votre_token'")
        print("  TELEGRAM_CHAT_IDS='votre_chat_id'")
        return
    
    # Si l'argument --simple est fourni, faire uniquement un test simple
    if args.simple:
        success = await test_simple_notification(args.token, args.chat_id)
        if success:
            print("\n🎉 Test réussi! Votre bot est correctement configuré.")
            print(f"Votre Chat ID est: {args.chat_id}")
            print("Vous pouvez maintenant exécuter des tests complets avec --all")
        return
    
    # Configurer manuellement les variables d'environnement si fournies par arguments
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    if args.chat_id:
        os.environ["TELEGRAM_CHAT_IDS"] = args.chat_id
    
    # Réinitialiser le notification_manager pour prendre en compte les nouvelles variables d'environnement
    notification_manager._initialize_from_env()
    
    # Exécuter les tests sélectionnés
    if args.all or (not args.connection and not args.anomaly and not args.simple):
        # Si --all ou aucun argument spécifique n'est fourni, exécuter tous les tests
        await test_notifications()  # Test de base
        await asyncio.sleep(2)
        await simulate_connection_notifications()
        await asyncio.sleep(2)
        await simulate_anomaly_notifications()
    else:
        # Sinon, exécuter uniquement les tests demandés
        if args.connection:
            await simulate_connection_notifications()
        
        if args.anomaly:
            await simulate_anomaly_notifications()
    
    logger.info("Tests de notification terminés.")


if __name__ == "__main__":
    # Si aucun argument n'est fourni, utiliser --simple par défaut
    if len(sys.argv) == 1:
        sys.argv.append("--simple")
    
    asyncio.run(main())
