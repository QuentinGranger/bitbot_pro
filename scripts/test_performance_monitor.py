#!/usr/bin/env python
"""
Script de test pour le moniteur de performance et les alertes Telegram.

Ce script simule des scénarios de trading pour tester les fonctionnalités
d'alerte de drawdown et d'exécution d'ordres inhabituelles.
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

from bitbot.utils.performance_monitor import performance_monitor
from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType
from bitbot.utils.logger import setup_logger, logger


async def test_drawdown_alerts():
    """Simule un scénario avec un drawdown progressif puis une récupération."""
    logger.info("Test des alertes de drawdown...")
    
    # Réinitialiser le moniteur
    performance_monitor.initial_balance = 10000.0
    performance_monitor.current_balance = 10000.0
    performance_monitor.peak_balance = 10000.0
    performance_monitor.trades_history = []
    performance_monitor.balance_history = []
    performance_monitor.drawdown_history = []
    
    # Configuration pour le test
    performance_monitor.drawdown_alert_threshold = 3.0  # Alerte à 3%
    performance_monitor.max_drawdown_threshold = 10.0   # Critique à 10%
    
    # Simuler une croissance initiale
    current_balance = performance_monitor.initial_balance
    timestamps = []
    base_time = datetime.now()
    
    for i in range(10):
        # Croissance de 1% par jour pendant 10 jours
        timestamp = base_time + timedelta(days=i)
        timestamps.append(timestamp)
        
        growth = current_balance * 0.01
        current_balance += growth
        
        performance_monitor.update_balance(current_balance, timestamp)
        
        # Simuler quelques trades gagnants
        trade = {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 40000 + (i * 1000),
            "quantity": 0.01,
            "profit_loss": growth
        }
        performance_monitor.record_trade(trade)
        
        await asyncio.sleep(0.1)  # Petit délai pour espacer les opérations
    
    # Le pic est atteint
    peak_balance = current_balance
    logger.info(f"Balance de pic atteinte: {peak_balance:.2f}")
    
    # Maintenant simuler un drawdown progressif
    for i in range(15):
        # Perte progressive sur 15 jours
        timestamp = timestamps[-1] + timedelta(days=i+1)
        
        # Calculer le pourcentage de perte pour cette étape
        # Plus agressif à mesure que le temps passe
        loss_percentage = 0.5 + (i * 0.2)
        loss = current_balance * (loss_percentage / 100)
        current_balance -= loss
        
        performance_monitor.update_balance(current_balance, timestamp)
        
        # Calculer le drawdown actuel pour le log
        current_drawdown = (peak_balance - current_balance) / peak_balance * 100
        logger.info(f"Jour {i+1}: Balance: {current_balance:.2f}, Drawdown: {current_drawdown:.2f}%")
        
        # Simuler des trades perdants
        trade = {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "price": 40000 - (i * 500),
            "quantity": 0.01,
            "profit_loss": -loss
        }
        performance_monitor.record_trade(trade)
        
        # Un délai plus long pour voir les alertes se déclencher
        await asyncio.sleep(0.5)
    
    # Récupération partielle
    for i in range(5):
        timestamp = timestamps[-1] + timedelta(days=i+16)
        
        recovery = current_balance * 0.02
        current_balance += recovery
        
        performance_monitor.update_balance(current_balance, timestamp)
        
        current_drawdown = (peak_balance - current_balance) / peak_balance * 100
        logger.info(f"Récupération jour {i+1}: Balance: {current_balance:.2f}, Drawdown: {current_drawdown:.2f}%")
        
        trade = {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 38000 + (i * 200),
            "quantity": 0.01,
            "profit_loss": recovery
        }
        performance_monitor.record_trade(trade)
        
        await asyncio.sleep(0.5)
    
    # Générer un rapport final
    report = performance_monitor.generate_performance_report()
    logger.info("\nRapport de performance final:")
    for key, value in report.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")


async def test_order_frequency_alerts():
    """Simule un scénario avec une fréquence d'ordres anormalement élevée."""
    logger.info("\nTest des alertes de fréquence d'ordres...")
    
    # Réinitialiser le moniteur mais garder la balance
    current_balance = performance_monitor.current_balance
    performance_monitor.trades_history = []
    
    # Configuration pour le test
    performance_monitor.order_frequency_threshold = 5  # 5 ordres/heure max
    performance_monitor.order_frequency_window = 1     # Fenêtre de 1 heure
    performance_monitor.last_order_frequency_alert_time = datetime.now() - timedelta(days=1)
    
    # Simuler une période normale d'abord (quelques ordres espacés)
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(3):
        timestamp = base_time + timedelta(minutes=i*15)
        
        trade = {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "price": 40000 + (i * 100),
            "quantity": 0.01,
            "profit_loss": random.uniform(-50, 50)
        }
        performance_monitor.record_trade(trade)
        
        await asyncio.sleep(0.1)
    
    logger.info("Période de trading normal simulée (3 ordres en 1 heure)")
    
    # Maintenant simuler une rafale d'ordres anormale
    base_time = datetime.now() - timedelta(minutes=30)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    sides = ["BUY", "SELL"]
    
    logger.info("Simulation d'une rafale d'ordres...")
    for i in range(20):  # Beaucoup d'ordres en peu de temps
        timestamp = base_time + timedelta(minutes=i*1.5)  # Un ordre toutes les 1.5 minutes
        
        symbol = random.choice(symbols)
        side = random.choice(sides)
        
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "price": 40000 + random.uniform(-1000, 1000),
            "quantity": 0.01 + random.uniform(0, 0.05),
            "profit_loss": random.uniform(-100, 100)
        }
        performance_monitor.record_trade(trade)
        
        # Simuler un changement de balance
        current_balance += trade["profit_loss"]
        performance_monitor.update_balance(current_balance, timestamp)
        
        if i % 5 == 0:
            logger.info(f"Exécution de {i} ordres simulés...")
        
        await asyncio.sleep(0.1)
    
    logger.info("Rafale d'ordres terminée (20 ordres en 30 minutes)")
    
    # Attendre un peu pour voir l'alerte se déclencher
    await asyncio.sleep(1)


async def test_performance_report():
    """Génère et envoie un rapport de performance complet."""
    logger.info("\nTest d'envoi du rapport de performance...")
    
    # Générer un rapport complet
    report = performance_monitor.generate_performance_report()
    
    # Formater le message
    message = (
        "📊 *Rapport de Performance BitBotPro* 📊\n\n"
        f"*Symbole:* BTCUSDT\n"
        f"*Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"*Balance initiale:* {report['initial_balance']:.2f} USD\n"
        f"*Balance actuelle:* {report['current_balance']:.2f} USD\n"
        f"*Balance maximale:* {report['peak_balance']:.2f} USD\n\n"
        f"*P&L:* {report['profit_loss']:.2f} USD ({report['profit_loss_pct']:.2f}%)\n"
        f"*Drawdown actuel:* {report['current_drawdown']:.2f}%\n"
        f"*Drawdown maximum:* {report['max_drawdown']:.2f}%\n\n"
        f"*Trades total:* {report['total_trades']}\n"
        f"*Trades gagnants:* {report['profitable_trades']}\n"
        f"*Win rate:* {report['win_rate']:.2f}%"
    )
    
    # Envoyer via le gestionnaire de notifications
    await notification_manager.notify(
        message=message,
        title="Rapport de Performance (Test)",
        priority=NotificationPriority.MEDIUM,
        notification_type=NotificationType.SYSTEM,
        details=report
    )
    
    logger.info("Rapport de performance envoyé")


async def main():
    """Fonction principale du script de test."""
    parser = argparse.ArgumentParser(description="Test du moniteur de performance et des alertes Telegram")
    parser.add_argument("--drawdown", action="store_true", help="Tester les alertes de drawdown")
    parser.add_argument("--orders", action="store_true", help="Tester les alertes de fréquence d'ordres")
    parser.add_argument("--report", action="store_true", help="Tester l'envoi de rapport de performance")
    parser.add_argument("--all", action="store_true", help="Tester toutes les fonctionnalités")
    args = parser.parse_args()
    
    # Configurer le logger
    setup_logger()
    
    # Vérifier si les identifiants Telegram sont configurés
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_ids = os.environ.get("TELEGRAM_CHAT_IDS")
    
    if not telegram_token or not telegram_chat_ids:
        logger.error("ERREUR: Token Telegram et/ou ID de chat manquants.")
        logger.error("Définissez les variables d'environnement suivantes:")
        logger.error("  TELEGRAM_BOT_TOKEN='votre_token'")
        logger.error("  TELEGRAM_CHAT_IDS='votre_chat_id'")
        return
    
    logger.info("---------------------------------------------------")
    logger.info("Test du moniteur de performance et alertes Telegram")
    logger.info("---------------------------------------------------")
    
    # Si aucun argument spécifique n'est fourni, exécuter tous les tests
    if not (args.drawdown or args.orders or args.report):
        args.all = True
    
    if args.drawdown or args.all:
        await test_drawdown_alerts()
    
    if args.orders or args.all:
        await test_order_frequency_alerts()
    
    if args.report or args.all:
        await test_performance_report()
    
    logger.info("Tests terminés")


if __name__ == "__main__":
    # Si aucun argument n'est fourni, utiliser --all par défaut
    if len(sys.argv) == 1:
        sys.argv.append("--all")
        
    asyncio.run(main())
