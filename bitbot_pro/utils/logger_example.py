"""
Exemple d'utilisation du système de journalisation de BitBot Pro.
Ce script démontre les différentes fonctionnalités du logger.
"""
from bitbot_pro.utils.logger import (
    logger, 
    log_exchange_connection,
    log_order_execution,
    log_system_error,
    log_network_issue
)
import time
import random
from datetime import datetime

def simulate_bot_activities():
    """Simule diverses activités du bot pour démontrer le système de journalisation."""
    
    # Démarrage du bot
    logger.info("Démarrage de BitBot Pro")
    
    # Connexion aux exchanges
    exchanges = ["Binance", "Bybit", "OKX"]
    for exchange in exchanges:
        success = random.random() > 0.2  # 80% de chance de réussite
        log_exchange_connection(exchange, success, 
                               {} if success else {"raison": "Erreur d'authentification API"})
        time.sleep(0.5)
    
    # Simulation d'analyse de marché
    logger.debug("Analyse technique en cours sur BTC/USDT")
    time.sleep(1)
    logger.debug("Calcul des indicateurs: RSI=42, MACD=-0.0002, BB=2.5%")
    
    # Simulation de décision de trading
    logger.info("Signal d'achat détecté sur BTC/USDT")
    
    # Simulation d'exécution d'ordre
    try:
        # 50% de chance d'avoir un problème réseau
        if random.random() > 0.5:
            raise ConnectionError("Timeout en attente de réponse")
            
        # Exécution de l'ordre
        log_order_execution("Binance", "BTC/USDT", "market", "buy", 0.01)
        
        # Recherche d'arbitrage
        logger.log("API", "Vérification des prix sur différents exchanges")
        prices = {
            "Binance": 68452.12,
            "Bybit": 68447.89,
            "OKX": 68459.75
        }
        for exch, price in prices.items():
            logger.debug(f"Prix BTC/USDT sur {exch}: {price}")
        
        # Opportunité d'arbitrage
        diff = prices["OKX"] - prices["Bybit"]
        logger.info(f"Opportunité d'arbitrage: Différence de prix BTC/USDT = {diff:.2f} USD")
        
        if diff > 10:
            log_order_execution("Bybit", "BTC/USDT", "market", "buy", 0.01)
            log_order_execution("OKX", "BTC/USDT", "market", "sell", 0.01)
            logger.log("TRADE", f"Profit d'arbitrage estimé: {diff * 0.01:.2f} USD")
        
    except ConnectionError as e:
        # Journalisation de l'erreur réseau
        log_network_issue("Binance API", str(e), retry_count=1)
        time.sleep(1)
        log_network_issue("Binance API", str(e), retry_count=2)
        time.sleep(1)
        
        # Simulation d'échec persistant
        log_system_error("OrderExecutor", "Échec d'exécution après plusieurs tentatives", e)
    
    # Simulation d'un événement critique
    if random.random() > 0.7:  # 30% de chance d'erreur critique
        try:
            # Simuler une division par zéro
            result = 1 / 0
        except Exception as e:
            log_system_error("CalculEngine", "Erreur de calcul critique", e)
    
    # Journalisation de la fin du cycle
    logger.info(f"Cycle de trading complété à {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Simuler plusieurs cycles de trading pour générer des logs
    logger.info("================ DÉMARRAGE DE LA SIMULATION ================")
    for i in range(3):
        logger.info(f"Début du cycle de trading #{i+1}")
        simulate_bot_activities()
        time.sleep(2)
    logger.info("================ FIN DE LA SIMULATION ================")
