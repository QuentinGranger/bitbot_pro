#!/usr/bin/env python3
"""
Script de test pour la détection d'anomalies dans les flux de données en temps réel.
Ce script simule des données de marché avec diverses anomalies et montre comment 
le système les détecte et les gère.
"""

import sys
import os
import time
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.utils.logger import setup_logger, logger
from bitbot.data.websocket import (
    WebSocketConfig, BinanceWebSocket, SlidingWindowBuffer,
    StreamType
)
from bitbot.utils.anomaly_detector import AnomalyDetector, AnomalyType, Anomaly


# Configuration du logger
setup_logger()
logger.info("Démarrage du test de détection d'anomalies")


class MarketDataSimulator:
    """
    Simulateur de données de marché avec diverses anomalies.
    """
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialise le simulateur.
        
        Args:
            symbol: Symbole à simuler
        """
        self.symbol = symbol
        self.base_price = 50000.0  # Prix de base pour BTC
        self.base_volume = 2.0     # Volume de base
        self.current_timestamp = time.time()
        
        # Paramètres de simulation
        self.price_volatility = 0.001  # Volatilité normale du prix (0.1%)
        self.volume_volatility = 0.02   # Volatilité normale du volume (2%)
        
    def generate_kline(self, 
                      timestamp: Optional[float] = None,
                      interval: str = "1m",
                      add_anomaly: Optional[str] = None,
                      anomaly_strength: float = 1.0) -> Dict:
        """
        Génère une kline simulée, avec ou sans anomalie.
        
        Args:
            timestamp: Timestamp spécifique, si None, incrémentation normale
            interval: Intervalle de la kline
            add_anomaly: Type d'anomalie à ajouter (price_spike, volume_spike, None)
            anomaly_strength: Force de l'anomalie (0.0 à 1.0)
            
        Returns:
            Données de kline au format WebSocket
        """
        # Gérer le timestamp
        if timestamp is None:
            # Avancer le temps
            self.current_timestamp += 60  # 1 minute par défaut
        else:
            self.current_timestamp = timestamp
        
        # Générer un mouvement de prix aléatoire normal
        price_change = random.normalvariate(0, self.price_volatility) * self.base_price
        
        # Générer un volume aléatoire normal
        volume = max(0.001, random.normalvariate(self.base_volume, self.base_volume * self.volume_volatility))
        
        # Prix de base avec mouvement aléatoire
        current_price = self.base_price + price_change
        
        # Appliquer les anomalies si demandé
        if add_anomaly == "price_spike":
            # Créer un spike de prix (jusqu'à 5% avec force maximale)
            spike_percent = 0.05 * anomaly_strength
            direction = 1 if random.random() > 0.5 else -1
            current_price += direction * self.base_price * spike_percent
            
        elif add_anomaly == "volume_spike":
            # Créer un spike de volume (jusqu'à 10x avec force maximale)
            spike_factor = 1 + 9 * anomaly_strength
            volume *= spike_factor
        
        # Construire la kline
        open_price = current_price - (price_change / 2)
        close_price = current_price + (price_change / 2)
        high_price = max(open_price, close_price) + abs(price_change) * 0.2
        low_price = min(open_price, close_price) - abs(price_change) * 0.2
        
        # Mettre à jour le prix de base pour la prochaine kline
        self.base_price = close_price
        
        # Formater la kline au format WebSocket de Binance
        kline_data = {
            "e": "kline",
            "E": int(self.current_timestamp * 1000),
            "s": self.symbol,
            "k": {
                "t": int(self.current_timestamp * 1000),
                "T": int((self.current_timestamp + 60) * 1000),
                "s": self.symbol,
                "i": interval,
                "f": random.randint(1000000, 9999999),
                "L": random.randint(1000000, 9999999),
                "o": str(round(open_price, 2)),
                "c": str(round(close_price, 2)),
                "h": str(round(high_price, 2)),
                "l": str(round(low_price, 2)),
                "v": str(round(volume, 8)),
                "n": random.randint(100, 500),
                "x": True,
                "q": str(round(volume * close_price, 2)),
                "V": str(round(volume * 0.6, 8)),
                "Q": str(round(volume * close_price * 0.6, 2))
            }
        }
        
        return kline_data
    
    def generate_trade(self, 
                      timestamp: Optional[float] = None,
                      add_anomaly: Optional[str] = None,
                      anomaly_strength: float = 1.0) -> Dict:
        """
        Génère une transaction simulée, avec ou sans anomalie.
        
        Args:
            timestamp: Timestamp spécifique, si None, incrémentation normale
            add_anomaly: Type d'anomalie à ajouter (price_spike, volume_spike, None)
            anomaly_strength: Force de l'anomalie (0.0 à 1.0)
            
        Returns:
            Données de trade au format WebSocket
        """
        # Gérer le timestamp
        if timestamp is None:
            # Avancer le temps par une petite quantité (0.1 à 2 secondes)
            self.current_timestamp += random.uniform(0.1, 2.0)
        else:
            self.current_timestamp = timestamp
        
        # Générer un mouvement de prix aléatoire normal
        price_change = random.normalvariate(0, self.price_volatility) * self.base_price
        
        # Générer un volume aléatoire normal pour cette transaction
        volume = max(0.0001, random.normalvariate(self.base_volume / 20, self.base_volume * self.volume_volatility / 20))
        
        # Prix actuel avec mouvement aléatoire
        current_price = self.base_price + price_change
        
        # Appliquer les anomalies si demandé
        if add_anomaly == "price_spike":
            # Créer un spike de prix (jusqu'à 5% avec force maximale)
            spike_percent = 0.05 * anomaly_strength
            direction = 1 if random.random() > 0.5 else -1
            current_price += direction * self.base_price * spike_percent
            
        elif add_anomaly == "volume_spike":
            # Créer un spike de volume (jusqu'à 10x avec force maximale)
            spike_factor = 1 + 9 * anomaly_strength
            volume *= spike_factor
        
        # Déterminer si c'est un achat ou une vente
        is_buyer = random.random() > 0.5
        
        # Mettre à jour le prix de base pour la prochaine transaction
        self.base_price = current_price
        
        # Formater la transaction au format WebSocket de Binance
        trade_data = {
            "e": "trade",
            "E": int(self.current_timestamp * 1000),
            "s": self.symbol,
            "t": random.randint(100000000, 999999999),
            "p": str(round(current_price, 2)),
            "q": str(round(volume, 8)),
            "b": random.randint(10000000, 99999999),
            "a": random.randint(10000000, 99999999),
            "T": int(self.current_timestamp * 1000),
            "m": is_buyer,
            "M": True
        }
        
        return trade_data


async def handle_anomaly(anomaly: Anomaly):
    """
    Callback pour traiter une anomalie.
    
    Args:
        anomaly: Anomalie détectée
    """
    logger.info(f"[CALLBACK] Anomalie détectée: {anomaly}")
    
    # Exemple: Stratégie spécifique selon le type d'anomalie
    if anomaly.anomaly_type == AnomalyType.VOLUME_SPIKE:
        logger.info(f"  -> Action: Ajustement des limites de position pour {anomaly.symbol}")
    elif anomaly.anomaly_type == AnomalyType.PRICE_SPIKE:
        logger.info(f"  -> Action: Vérification de la fiabilité du prix pour {anomaly.symbol}")


async def test_anomaly_detection():
    """
    Teste la détection d'anomalies avec des données simulées.
    """
    # Configuration du client WebSocket
    config = WebSocketConfig(
        data_retention_seconds=300,
        volume_spike_z_score=3.0,
        price_spike_z_score=3.0,
        data_gap_threshold=2.0,
        anomaly_detection_enabled=True,
        auto_recover_anomalies=True,
        message_rate_limit=100  # Augmenter pour éviter les problèmes de rate limit
    )
    
    # Initialiser le client WebSocket (simulé)
    ws = BinanceWebSocket(config)
    
    # Enregistrer des callbacks pour les anomalies
    ws.register_anomaly_callback(AnomalyType.VOLUME_SPIKE, handle_anomaly)
    ws.register_anomaly_callback(AnomalyType.PRICE_SPIKE, handle_anomaly)
    ws.register_anomaly_callback(AnomalyType.DATA_GAP, handle_anomaly)
    
    # Simulateur de données
    sim = MarketDataSimulator(symbol="BTCUSDT")
    
    # Simuler un flux normal de données (établir une baseline)
    logger.info("Phase 1: Génération de données normales pour établir une baseline...")
    start_time = sim.current_timestamp
    
    # Générer 50 klines normales
    for _ in range(50):
        kline = sim.generate_kline()
        await ws._handle_message(json.dumps(kline))
        await asyncio.sleep(0.01)  # Petite pause pour voir la progression
    
    logger.info(f"Baseline établie avec {ws.stats['messages_processed']} messages traités")
    
    # Phase 2: Simuler un spike de volume
    logger.info("\nPhase 2: Simulation d'un spike de volume...")
    
    # Générer 5 klines avec un spike de volume croissant
    for strength in [0.2, 0.4, 0.6, 0.8, 1.0]:
        kline = sim.generate_kline(add_anomaly="volume_spike", anomaly_strength=strength)
        await ws._handle_message(json.dumps(kline))
        await asyncio.sleep(0.5)  # Pause plus longue pour observer les logs
    
    # Revenir à la normale
    for _ in range(5):
        kline = sim.generate_kline()
        await ws._handle_message(json.dumps(kline))
        await asyncio.sleep(0.01)
    
    # Phase 3: Simuler un spike de prix
    logger.info("\nPhase 3: Simulation d'un spike de prix...")
    
    # Générer 5 klines avec un spike de prix croissant
    for strength in [0.2, 0.4, 0.6, 0.8, 1.0]:
        kline = sim.generate_kline(add_anomaly="price_spike", anomaly_strength=strength)
        await ws._handle_message(json.dumps(kline))
        await asyncio.sleep(0.5)  # Pause plus longue pour observer les logs
    
    # Revenir à la normale
    for _ in range(5):
        kline = sim.generate_kline()
        await ws._handle_message(json.dumps(kline))
        await asyncio.sleep(0.01)
    
    # Phase 4: Simuler un gap de données
    logger.info("\nPhase 4: Simulation d'un gap de données...")
    
    # Avancer considérablement le temps pour créer un gap
    current_time = sim.current_timestamp
    next_timestamp = current_time + 300  # Gap de 5 minutes
    kline = sim.generate_kline(timestamp=next_timestamp)
    await ws._handle_message(json.dumps(kline))
    await asyncio.sleep(0.5)
    
    # Vérifier manuellement si un gap est détecté (en utilisant la version avec start_time/end_time)
    has_gap = ws.detect_data_gap(
        symbol="BTCUSDT",
        stream_type="kline",
        start_time=current_time,
        end_time=next_timestamp,
        expected_interval=60  # 1 minute pour les klines
    )
    
    logger.info(f"Gap de données détecté manuellement: {has_gap}")
    
    # Phase 5: Afficher les statistiques
    logger.info("\nPhase 5: Statistiques sur les anomalies détectées...")
    
    stats = ws.get_anomaly_stats()
    logger.info(f"Anomalies détectées: {stats['detected_total']}")
    logger.info(f"Anomalies récupérées: {stats['recovered_total']}")
    
    if "by_type" in stats:
        logger.info("\nDistribution par type:")
        for type_name, count in stats["by_type"].items():
            logger.info(f"  - {type_name}: {count}")
    
    if "most_severe" in stats and stats["most_severe"]:
        logger.info("\nAnomalies les plus sévères:")
        for i, anomaly in enumerate(stats["most_severe"][:3], 1):
            logger.info(f"  {i}. {anomaly}")
    
    # Phase 6: Tester le remplissage d'un gap
    logger.info("\nPhase 6: Test de récupération de données historiques pour combler un gap...")
    
    # Simuler un appel pour combler un gap
    gap_start = sim.current_timestamp - 300
    gap_end = sim.current_timestamp - 60
    
    result = ws.fill_historical_gap(
        symbol="BTCUSDT",
        stream_type="kline",
        start_time=gap_start,
        end_time=gap_end
    )
    
    logger.info(f"Tentative de récupération des données historiques: {'Lancée' if result else 'Échouée'}")
    
    # Note: Dans un cas réel, cet appel déclencherait une requête à l'API REST
    # et comblerait effectivement le gap avec des données historiques
    
    logger.info("\nTest terminé!")


if __name__ == "__main__":
    try:
        # Exécuter le test
        asyncio.run(test_anomaly_detection())
    except KeyboardInterrupt:
        logger.info("Test interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur pendant le test: {str(e)}")
        import traceback
        traceback.print_exc()
