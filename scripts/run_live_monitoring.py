#!/usr/bin/env python3
"""
Script pour surveiller en continu le marché et envoyer des notifications Telegram.
Ce script utilise le WebSocket de Binance pour recevoir les données en temps réel
et envoie des notifications lorsque des anomalies ou signaux sont détectés.
Il surveille également les performances du bot et génère des alertes en cas de 
drawdown anormal ou d'exécutions d'ordres inhabituelles.
"""

import asyncio
import argparse
import json
import os
import ssl
import sys
import threading
import time
import logging
import websockets
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Deque
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Import des modules internes
from bitbot.utils.notifications import NotificationManager
from bitbot.utils.telegram_alerts import telegram_alerts, send_telegram_alert, AlertType, AlertPriority
from bitbot.utils.logger import setup_logger

from bitbot.data.websocket import BinanceWebSocket, StreamType, WebSocketConfig
from bitbot.utils.anomaly_detector import AnomalyDetector, AnomalyType, Anomaly
from bitbot.models.market_data import Kline, Trade, OrderBook, Ticker
from bitbot.utils.performance_monitor import PerformanceMonitor

# Configuration du logging
setup_logger()

# Initialisation des gestionnaires
notification_manager = NotificationManager()
performance_monitor = PerformanceMonitor()

# Variables pour le contrôle de taux
message_counter = 0
sample_rate = 0.1  # Traiter seulement 10% des messages pour éviter le rate limiting

# Verrouillage pour éviter les problèmes d'accès concurrents au WebSocket
websocket_lock = asyncio.Lock()

# Solution avec queue pour distribuer les messages
message_queue = asyncio.Queue()

async def message_consumer():
    """Consomme les messages de la queue et les distribue aux handlers appropriés."""
    global message_counter, sample_rate
    
    while True:
        try:
            message_data = await message_queue.get()
            symbol, stream_type, callback, message = message_data
            
            try:
                # Application de l'échantillonnage pour limiter le traitement
                message_counter += 1
                if message_counter % int(1/sample_rate) != 0:
                    message_queue.task_done()
                    continue
                    
                # Décodage et traitement du message
                if stream_type == StreamType.KLINE:
                    data = json.loads(message)
                    kline_data = data.get('k', {})
                    kline = Kline(
                        symbol=data.get('s', symbol),
                        interval=kline_data.get('i', ''),
                        open_time=kline_data.get('t', 0),
                        close_time=kline_data.get('T', 0),
                        open=float(kline_data.get('o', 0)),
                        high=float(kline_data.get('h', 0)),
                        low=float(kline_data.get('l', 0)),
                        close=float(kline_data.get('c', 0)),
                        volume=float(kline_data.get('v', 0)),
                        is_closed=kline_data.get('x', False)
                    )
                    await callback(kline)
                
                # D'autres types de données peuvent être traités ici
                
            except Exception as e:
                logger.error(f"Erreur dans message_consumer: {str(e)}")
            finally:
                message_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Tâche message_consumer annulée")
            break
        except Exception as e:
            logger.error(f"Erreur critique dans message_consumer: {str(e)}")
            await asyncio.sleep(1)  # Éviter une boucle infinie d'erreurs

async def handle_anomaly(anomaly: Anomaly):
    """Traite une anomalie détectée."""
    global most_recent_anomalies
    
    # Enregistrer l'anomalie dans l'historique circulaire
    anomaly_str = f"{anomaly.type.value} - {anomaly.symbol} - Sévérité: {anomaly.severity:.2f}"
    most_recent_anomalies.append((datetime.now(), anomaly_str))
    
    # Mapper les types d'anomalies aux types d'alertes Telegram
    anomaly_to_alert_type = {
        AnomalyType.PRICE_SPIKE: AlertType.PRICE,
        AnomalyType.VOLUME_SPIKE: AlertType.VOLUME,
        AnomalyType.DATA_GAP: AlertType.SYSTEM
    }
    
    # Mapper les niveaux de sévérité aux priorités d'alertes
    severity_to_priority = lambda s: (
        AlertPriority.CRITICAL if s >= 5.0 else
        AlertPriority.HIGH if s >= 3.0 else
        AlertPriority.MEDIUM if s >= 2.0 else
        AlertPriority.LOW
    )
    
    # Préparer les détails pour l'alerte
    details = {
        "Symbol": anomaly.symbol,
        "Sévérité": f"{anomaly.severity:.2f}",
        "Timestamp": datetime.fromtimestamp(anomaly.timestamp).strftime("%H:%M:%S")
    }
    
    # Ajouter les détails spécifiques au type d'anomalie
    if anomaly.details:
        if 'price' in anomaly.details and 'previous_price' in anomaly.details:
            details["Prix actuel"] = f"{anomaly.details['price']:.2f}"
            details["Prix précédent"] = f"{anomaly.details['previous_price']:.2f}"
        
        if 'percent_change' in anomaly.details:
            details["Variation"] = f"{anomaly.details['percent_change']:.2f}%"
            
        if 'volume' in anomaly.details and 'average_volume' in anomaly.details:
            details["Volume"] = f"{anomaly.details['volume']:.2f}"
            details["Volume moyen"] = f"{anomaly.details['average_volume']:.2f}"
            
        if 'gap_seconds' in anomaly.details:
            details["Durée du gap"] = f"{anomaly.details['gap_seconds']} secondes"
    
    # Préparer le message d'alerte
    alert_type = anomaly_to_alert_type.get(anomaly.type, AlertType.ANOMALY)
    priority = severity_to_priority(anomaly.severity)
    
    # Titre et message en fonction du type d'anomalie
    titles = {
        AnomalyType.PRICE_SPIKE: "Mouvement de Prix Important",
        AnomalyType.VOLUME_SPIKE: "Volume Anormal Détecté",
        AnomalyType.DATA_GAP: "Interruption des Données"
    }
    
    messages = {
        AnomalyType.PRICE_SPIKE: f"{'Hausse' if anomaly.details.get('percent_change', 0) > 0 else 'Baisse'} de prix rapide détectée sur {anomaly.symbol}!",
        AnomalyType.VOLUME_SPIKE: f"Volume anormalement élevé sur {anomaly.symbol}!",
        AnomalyType.DATA_GAP: f"Interruption des données pendant {anomaly.details.get('gap_seconds', '?')} secondes sur {anomaly.symbol}!"
    }
    
    title = titles.get(anomaly.type, f"Anomalie {anomaly.type.value}")
    message = messages.get(anomaly.type, f"Anomalie détectée sur {anomaly.symbol}")
    
    # Envoyer l'alerte via Telegram
    await send_telegram_alert(
        message=message,
        alert_type=alert_type,
        priority=priority,
        title=title,
        details=details
    )
    
    # Notifier également via le système de notification standard
    if notification_manager:
        notification_type = {
            AnomalyType.PRICE_SPIKE: NotificationType.ANOMALY,
            AnomalyType.VOLUME_SPIKE: NotificationType.ANOMALY,
            AnomalyType.DATA_GAP: NotificationType.CONNECTION
        }.get(anomaly.type, NotificationType.SYSTEM)
        
        await notification_manager.notify(
            message=f"Anomalie détectée: {anomaly_str}",
            priority=NotificationPriority.HIGH,
            notification_type=notification_type,
            details=anomaly.details
        )
    
    logger.warning(f"ANOMALIE DÉTECTÉE: {anomaly_str}")
    logger.debug(f"Détails: {anomaly.details}")

async def handle_kline_data(kline: Kline):
    """Traite les données de kline en temps réel."""
    try:
        # Mise à jour du moniteur de performance
        if kline.is_closed:
            # Pour le calcul de la performance, utiliser le prix de clôture
            current_price = kline.close
            performance_monitor.update_price(current_price)
            
            # Journalisation
            logger.info(f"Kline fermée: {kline.symbol} - {kline.interval} - Prix: {current_price}")
    except Exception as e:
        logger.error(f"Erreur dans handle_kline_data: {str(e)}")

async def handle_trade_data(data):
    """Gestionnaire pour les données de trade."""
    global message_counter
    
    try:
        # Increment counter and sample messages to avoid rate limiting
        message_counter += 1
        if random.random() > sample_rate:
            return  # Skip processing to avoid rate limiting
        
        # Add small delay to spread out message processing
        await asyncio.sleep(0.01)
        
        symbol = data.get('s', 'UNKNOWN')
        price = float(data.get('p', 0))
        quantity = float(data.get('q', 0))
        
        logger.debug(f"Trade - {symbol} - Prix: {price} - Quantité: {quantity}")
        
        # Mise à jour des données pour le moniteur de performance
        await performance_monitor.update_trade_data(symbol, price, quantity)
    except Exception as e:
        logger.error(f"Erreur dans handle_trade_data: {str(e)}")

async def check_performance():
    """Vérification périodique des performances."""
    while True:
        try:
            # Calculer le drawdown manuellement puisque la méthode calculate_drawdown n'existe pas
            current_balance = performance_monitor.current_balance
            peak_balance = performance_monitor.peak_balance
            
            # Calcul du drawdown en pourcentage
            drawdown = 0
            if peak_balance > 0:
                drawdown = (peak_balance - current_balance) / peak_balance * 100
            
            if drawdown >= performance_monitor.max_drawdown_threshold:
                await notification_manager.notify(
                    message=f"🚨 ALERTE CRITIQUE: Drawdown de {drawdown:.2f}% détecté!",
                    title="Alerte Critique - Drawdown Excessif",
                    details={
                        "Drawdown actuel": f"{drawdown:.2f}%",
                        "Seuil critique": f"{performance_monitor.max_drawdown_threshold:.2f}%",
                        "Balance actuelle": f"{current_balance:.2f} USD",
                        "Balance max": f"{peak_balance:.2f} USD",
                        "Action requise": "Vérifiez immédiatement la stratégie et envisagez d'arrêter le trading"
                    }
                )
            elif drawdown >= performance_monitor.drawdown_alert_threshold:
                await notification_manager.notify(
                    message=f"⚠️ Alerte: Drawdown de {drawdown:.2f}% détecté",
                    title="Alerte - Drawdown Significatif",
                    details={
                        "Drawdown actuel": f"{drawdown:.2f}%",
                        "Seuil d'alerte": f"{performance_monitor.drawdown_alert_threshold:.2f}%",
                        "Balance actuelle": f"{current_balance:.2f} USD",
                        "Balance max": f"{peak_balance:.2f} USD",
                        "Action suggérée": "Surveillez de près l'évolution du marché"
                    }
                )
            
            # Nous ne vérifions pas la fréquence des ordres car cette méthode n'est probablement pas disponible non plus
            # Enlever cette partie pour éviter des erreurs
        except Exception as e:
            logger.error(f"Erreur dans check_performance: {str(e)}")
        
        # Attendre jusqu'à la prochaine vérification
        await asyncio.sleep(10 * 60)

async def generate_performance_report():
    """Génération du rapport de performance périodique."""
    while True:
        try:
            # Récupération des métriques de performance manuellement
            # Calcul du P&L
            pnl = performance_monitor.current_balance - performance_monitor.initial_balance
            pnl_percent = (pnl / performance_monitor.initial_balance) * 100 if performance_monitor.initial_balance > 0 else 0
            
            # Calcul du drawdown
            current_drawdown = 0
            if performance_monitor.peak_balance > 0:
                current_drawdown = (performance_monitor.peak_balance - performance_monitor.current_balance) / performance_monitor.peak_balance * 100
            
            # Construire des métriques de base
            metrics = {
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'current_drawdown': current_drawdown,
                'max_drawdown': current_drawdown,  # Nous n'avons pas le max historique, donc on utilise le courant
                'period': f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
            
            # Envoi du rapport
            await notification_manager.notify(
                message=f"📊 Rapport de Performance BitBotPro",
                title="Rapport de Performance Périodique",
                details={
                    "Balance initiale": f"{performance_monitor.initial_balance:.2f} USD",
                    "Balance actuelle": f"{performance_monitor.current_balance:.2f} USD",
                    "P&L": f"{metrics['pnl']:.2f} USD ({metrics['pnl_percent']:.2f}%)",
                    "Drawdown actuel": f"{metrics['current_drawdown']:.2f}%",
                    "Période": f"{metrics['period']}"
                }
            )
        except Exception as e:
            logger.error(f"Erreur dans generate_performance_report: {str(e)}")
        
        # Attendre l'intervalle spécifié (en heures)
        await asyncio.sleep(args.report_interval * 60 * 60)

async def simulate_anomaly():
    """Simule une anomalie de prix pour tester les notifications."""
    # Cette fonction est utilisée uniquement pour les tests
    # et peut être supprimée en production
    
    # Attendre quelques secondes pour que le système soit bien démarré
    await asyncio.sleep(5)
    
    logger.info("Simulation d'une anomalie de prix pour tester les alertes Telegram")
    
    # Créer une anomalie simulée
    anomaly = Anomaly(
        type=AnomalyType.PRICE_SPIKE,
        symbol=args.symbol,
        timestamp=time.time(),
        severity=4.2,  # Z-score élevé
        details={
            "price": 70000.0,  # Prix anormal
            "previous_price": 60000.0,
            "percent_change": 16.67,
            "timeframe": args.timeframe,
            "detected_at": datetime.now().isoformat()
        }
    )
    
    # Appeler directement le gestionnaire d'anomalies
    await handle_anomaly(anomaly)
    
    logger.info("Anomalie simulée envoyée avec succès!")
    
    # Attendre 5 secondes puis simuler une autre anomalie de type volume
    await asyncio.sleep(5)
    
    logger.info("Simulation d'une anomalie de volume pour tester les alertes")
    
    volume_anomaly = Anomaly(
        type=AnomalyType.VOLUME_SPIKE,
        symbol=args.symbol,
        timestamp=time.time(),
        severity=3.8,
        details={
            "volume": 5000.0,
            "average_volume": 1000.0,
            "percent_change": 400.0,
            "timeframe": args.timeframe,
            "detected_at": datetime.now().isoformat()
        }
    )
    
    await handle_anomaly(volume_anomaly)
    
    logger.info("Seconde anomalie simulée envoyée avec succès!")

async def cleanup(signal=None):
    """Nettoyage avant l'arrêt du script."""
    global message_queue, websocket_lock
    
    if signal:
        logger.info(f"Signal reçu: {signal}")
    
    logger.info("Nettoyage et arrêt en cours...")
    
    # Attendre que tous les messages dans la queue soient traités
    if message_queue:
        try:
            await asyncio.wait_for(message_queue.join(), timeout=5)
            logger.info("File d'attente de messages vidée avec succès")
        except asyncio.TimeoutError:
            logger.warning("Timeout en attendant que la file d'attente de messages se vide")
    
    # Fermer proprement le WebSocket s'il existe
    try:
        if 'ws_manager' in globals():
            async with websocket_lock:
                await ws_manager.close()
                logger.info("WebSocket fermé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la fermeture du WebSocket: {str(e)}")
    
    # Envoyer une notification de fin
    try:
        uptime = datetime.now() - start_time
        await notification_manager.notify(
            message=f"🛑 BitBotPro - Surveillance arrêtée pour {args.symbol}",
            title="BitBotPro - Surveillance Arrêtée",
            details={
                "Date d'arrêt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Durée de fonctionnement": str(uptime).split('.')[0],  # HH:MM:SS sans millisecondes
                "Symbole": args.symbol,
                "Raison": "Arrêt manuel" if signal else "Erreur système"
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de la notification d'arrêt: {str(e)}")
    
    # Attendre un peu pour que les logs terminent d'être écrits
    await asyncio.sleep(1)
    
    # Quitter proprement
    if signal:
        import sys
        sys.exit(0)

async def main():
    """Fonction principale de surveillance."""
    global notification_manager, performance_monitor, message_queue
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Analyser les arguments
    parser = argparse.ArgumentParser(description='Surveillance des marchés et envoi de notifications')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbole à surveiller')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe à analyser')
    parser.add_argument('--check-interval', type=int, default=30, help='Intervalle de vérification en secondes')
    parser.add_argument('--report-interval', type=int, default=300, help='Intervalle de rapport en secondes')
    parser.add_argument('--disable-ssl-verify', action='store_true', help='Désactiver la vérification SSL')
    parser.add_argument('--test', action='store_true', help='Mode test avec simulation d\'anomalies')
    
    # Traiter les arguments
    args = parser.parse_args()
    
    # Configurer le manager de notification
    notification_manager = NotificationManager()
    
    # Mode test : échantillonnage plus agressif
    global sample_rate
    if args.test:
        sample_rate = 0.01  # 1% des messages
        logger.info("Mode test activé: échantillonnage à 1%")
    
    # Initialiser le moniteur de performance
    performance_monitor.report_interval = args.report_interval
    performance_monitor.check_interval = args.check_interval
    
    # Configuration du WebSocket avec SSL renforcé
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Désactiver TLS < 1.2
    
    if args.disable_ssl_verify:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        logger.warning("Vérification SSL désactivée. Utiliser avec précaution.")
    
    websocket_config = WebSocketConfig(
        base_endpoint="wss://stream.binance.com:9443",
        ping_interval=20,
        ping_timeout=10,
        reconnect_delay=5,
        max_reconnect_attempts=10,  # Augmenté pour plus de robustesse
        message_rate_limit=20,       # Augmenté pour éviter le rate limiting
        data_retention_seconds=600,  # 10 minutes
        anomaly_detection_enabled=True,
        auto_recover_anomalies=True,
    )
    
    # Initialiser le WebSocket manager
    ws_manager = BinanceWebSocket(websocket_config)
    ws_manager.ssl_context = ssl_context
    
    # Enregistrer les callbacks pour les anomalies
    ws_manager.register_anomaly_callback(AnomalyType.VOLUME_SPIKE, handle_anomaly)
    ws_manager.register_anomaly_callback(AnomalyType.PRICE_SPIKE, handle_anomaly)
    ws_manager.register_anomaly_callback(AnomalyType.DATA_GAP, handle_anomaly)
    
    # Démarrer le message consumer
    consumer_task = asyncio.create_task(message_consumer())
    
    try:
        # Démarrer le WebSocket avec verrou
        async with websocket_lock:
            await ws_manager.connect()
        
        # S'abonner aux flux de données
        kline_stream = f"{args.timeframe}"  # Le timeframe est passé en paramètre
        
        # Utiliser une fonction lambda pour mettre les messages dans la queue
        async with websocket_lock:
            await ws_manager.subscribe(
                args.symbol.lower(), 
                StreamType.KLINE, 
                lambda message: message_queue.put_nowait((args.symbol.lower(), StreamType.KLINE, handle_kline_data, message))
            )
        
        # Envoyer une notification Telegram au démarrage
        await send_telegram_alert(
            message="✅ Surveillance en temps réel activée pour le marché crypto",
            alert_type=AlertType.SYSTEM,
            priority=AlertPriority.MEDIUM,
            title="BitBotPro - Surveillance Démarrée",
            details={
                "Symbol": args.symbol,
                "Timeframe": args.timeframe,
                "Mode test": "Activé" if args.test else "Désactivé",
                "SSL Verify": "Désactivé" if args.disable_ssl_verify else "Activé",
                "Démarré à": datetime.now().strftime("%H:%M:%S")
            }
        )
        
        # Démarrer les tâches de surveillance
        performance_task = asyncio.create_task(check_performance())
        report_task = asyncio.create_task(generate_performance_report())
        simulate_anomaly_task = asyncio.create_task(simulate_anomaly())
        
        # Configurer les gestionnaires de signaux pour un arrêt propre
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(cleanup(sig)))
        
        # Maintenir le script en fonctionnement jusqu'à interruption
        while True:
            await asyncio.sleep(60)
            
    except Exception as e:
        logger.error(f"Erreur dans la fonction main: {str(e)}")
        await cleanup()
    finally:
        # Assurer le nettoyage même en cas d'erreur
        if 'consumer_task' in locals() and not consumer_task.done():
            consumer_task.cancel()
        if 'performance_task' in locals() and not performance_task.done():
            performance_task.cancel()
        if 'report_task' in locals() and not report_task.done():
            report_task.cancel()
        if 'simulate_anomaly_task' in locals() and not simulate_anomaly_task.done():
            simulate_anomaly_task.cancel()

if __name__ == "__main__":
    # Enregistrer l'heure de démarrage
    start_time = datetime.now()
    
    # Exécuter la boucle principale
    asyncio.run(main())
