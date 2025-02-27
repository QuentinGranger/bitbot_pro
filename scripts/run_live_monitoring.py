#!/usr/bin/env python3
"""
Script pour surveiller en continu le marché et envoyer des notifications Telegram.
Ce script utilise le WebSocket de Binance pour recevoir les données en temps réel
et envoie des notifications lorsque des anomalies ou signaux sont détectés.
"""

import os
import sys
import asyncio
import signal
import logging
from datetime import datetime, timedelta
import argparse

# Ajouter le répertoire parent au chemin pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.websocket import BinanceWebSocketManager, StreamType
from bitbot.utils.anomaly_detector import AnomalyDetector, AnomalyType
from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType
from bitbot.trader import Trader
from bitbot.config import Config
from bitbot.utils.logger import logger


async def handle_anomaly(anomaly):
    """Fonction de callback pour traiter les anomalies détectées."""
    logger.warning(f"Anomalie détectée: {anomaly}")
    
    # La notification sera déjà envoyée par le WebSocketManager


async def process_market_data(symbol, timeframe, interval_minutes):
    """
    Traite périodiquement les données de marché pour générer des signaux
    et envoyer des notifications.
    
    Args:
        symbol: Symbole à surveiller
        timeframe: Timeframe pour l'analyse
        interval_minutes: Intervalle en minutes entre chaque analyse
    """
    trader = Trader()
    
    while True:
        try:
            # Mettre à jour les données et générer des signaux
            market_data = trader.update_market_data(symbol, timeframe)
            signals = trader.generate_signals(market_data)
            
            if signals:
                # Les signaux seront traités et les notifications envoyées par execute_signals
                trader.execute_signals(signals)
            else:
                logger.info(f"Aucun signal détecté pour {symbol} à {datetime.now().strftime('%H:%M:%S')}")
            
            # Attendre l'intervalle spécifié
            await asyncio.sleep(interval_minutes * 60)
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            await notification_manager.notify(
                message=f"Erreur lors du traitement des données: {str(e)}",
                title="BitBotPro - Erreur",
                notification_type=NotificationType.ERROR,
                priority=NotificationPriority.HIGH
            )
            await asyncio.sleep(60)  # Attendre 1 minute avant de réessayer


async def main():
    """Fonction principale de surveillance."""
    parser = argparse.ArgumentParser(description='Surveillance en temps réel avec BitBotPro')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbole à surveiller')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe pour l\'analyse')
    parser.add_argument('--interval', type=int, default=5, help='Intervalle en minutes entre les analyses')
    args = parser.parse_args()
    
    # Signaler le démarrage
    await notification_manager.notify(
        message=f"🚀 BitBotPro - Surveillance en direct démarrée pour {args.symbol}",
        title="BitBotPro - Surveillance Active",
        details={
            "Date de démarrage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbole": args.symbol,
            "Timeframe": args.timeframe,
            "Intervalle d'analyse": f"{args.interval} minutes"
        }
    )
    
    # Configurer le gestionnaire de WebSocket
    ws_manager = BinanceWebSocketManager()
    
    # Configurer le détecteur d'anomalies
    anomaly_detector = AnomalyDetector()
    
    # Enregistrer le callback pour les anomalies
    ws_manager.register_anomaly_callback(handle_anomaly)
    
    # Démarrer le WebSocket
    await ws_manager.add_subscription(f"{args.symbol.lower()}@kline_{args.timeframe}")
    await ws_manager.add_subscription(f"{args.symbol.lower()}@trade")
    await ws_manager.add_subscription(f"{args.symbol.lower()}@ticker")
    
    # Démarrer la tâche de traitement des données
    data_task = asyncio.create_task(process_market_data(args.symbol, args.timeframe, args.interval))
    
    # Gérer la fermeture propre de l'application
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Signal de fermeture reçu, arrêt en cours...")
        data_task.cancel()
        loop.create_task(ws_manager.close())
        loop.create_task(
            notification_manager.notify(
                message="BitBotPro - Surveillance arrêtée",
                title="BitBotPro - Arrêt",
                details={"Date d'arrêt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            )
        )
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Maintenir le script en exécution
        while True:
            await asyncio.sleep(3600)  # Vérifier toutes les heures si le script est toujours actif
            await notification_manager.notify(
                message="BitBotPro - Surveillance toujours active",
                title="BitBotPro - Statut",
                notification_type=NotificationType.SYSTEM,
                priority=NotificationPriority.LOW,
                silent=True
            )
    except asyncio.CancelledError:
        logger.info("Tâche principale annulée")
    finally:
        # Fermer proprement les connexions
        await ws_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
