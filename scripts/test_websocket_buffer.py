#!/usr/bin/env python3
"""
Script de test pour la fonctionnalité de buffer de fenêtre glissante.
Ce script simule une connexion WebSocket avec des microcoupures réseau
et montre comment le buffer local permet de maintenir la continuité des données.
"""

import sys
import os
import asyncio
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au sys.path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bitbot.data.websocket import BinanceWebSocket, WebSocketConfig, StreamType
from bitbot.utils.logger import logger


class MockWebSocket:
    """Simulation d'un WebSocket avec des microcoupures réseau."""
    
    def __init__(self, failure_rate=0.1, recovery_time=2.0):
        """
        Args:
            failure_rate: Probabilité de microcoupure (0.0 à 1.0)
            recovery_time: Temps moyen de récupération en secondes
        """
        self.failure_rate = failure_rate
        self.recovery_time = recovery_time
        self.connected = False
        self.data_generator = None
        
    async def connect(self):
        """Simule une connexion au WebSocket."""
        await asyncio.sleep(0.5)  # Simuler le temps de connexion
        self.connected = True
        logger.info("Connexion WebSocket simulée établie")
        return True
        
    async def simulate_market_data(self, client, symbol, interval=1.0):
        """
        Simule la réception de données de marché avec des microcoupures réseau.
        
        Args:
            client: Client WebSocket
            symbol: Symbole de trading à simuler
            interval: Intervalle entre les données en secondes
        """
        last_data_time = time.time()
        counter = 0
        
        while client.running:
            try:
                current_time = time.time()
                
                # Simuler une microcoupure réseau aléatoire
                if random.random() < self.failure_rate:
                    logger.warning("🔴 Simulation d'une microcoupure réseau")
                    self.connected = False
                    
                    # Simuler un temps de récupération variable
                    downtime = random.uniform(0.5, self.recovery_time * 2)
                    await asyncio.sleep(downtime)
                    
                    logger.info(f"🟢 Reconnexion après {downtime:.2f}s de coupure")
                    self.connected = True
                    
                    # Mettre à jour le timestamp pour mesurer la perte de données
                    data_gap = current_time - last_data_time
                    
                    # Calculer combien de points de données ont été manqués
                    missed_points = int(data_gap / interval)
                    logger.warning(f"📊 Gap de données détecté: {missed_points} points manqués sur {data_gap:.2f}s")
                    
                elif self.connected:
                    # Simuler la réception de données à intervalles réguliers
                    if current_time - last_data_time >= interval:
                        counter += 1
                        
                        # Créer une donnée simulée
                        timestamp = int(current_time * 1000)  # Convertir en ms
                        mock_data = {
                            'e': 'kline',
                            'E': timestamp,
                            's': symbol,
                            'k': {
                                't': timestamp,
                                'T': timestamp + int(interval * 1000),
                                's': symbol,
                                'i': '1m',
                                'f': 100 + counter,
                                'L': 100 + counter,
                                'o': str(100 + random.uniform(-1, 1)),
                                'c': str(100 + random.uniform(-1, 1)),
                                'h': str(100 + random.uniform(0, 2)),
                                'l': str(100 + random.uniform(-2, 0)),
                                'v': str(random.uniform(10, 100)),
                                'n': random.randint(10, 100),
                                'x': False,
                                'q': str(random.uniform(1000, 10000)),
                                'V': str(random.uniform(5, 50)),
                                'Q': str(random.uniform(500, 5000))
                            }
                        }
                        
                        # Notifier le client
                        await client._handle_message(json.dumps(mock_data))
                        
                        last_data_time = current_time
                
                await asyncio.sleep(0.1)  # Éviter de surcharger la CPU
                
            except Exception as e:
                logger.error(f"Erreur pendant la simulation: {str(e)}")
                await asyncio.sleep(1)


async def main():
    """Fonction principale."""
    logger.info("Démarrage du test de buffer de fenêtre glissante")
    
    # Configuration avec un buffer plus petit pour les tests
    config = WebSocketConfig(
        sliding_window_size=20,
        data_retention_seconds=60,
        gap_detection_threshold=1.0
    )
    
    # Créer le client WebSocket
    client = BinanceWebSocket(config)
    
    try:
        # Établir la connexion
        await client.connect()
        
        # S'abonner à un stream (simulation)
        symbol = "BTCUSDT"
        await client.subscribe(symbol, StreamType.KLINE, None)
        
        # Créer un mock WebSocket pour simuler les données et les pannes
        mock = MockWebSocket(failure_rate=0.1, recovery_time=2.0)
        
        # Démarrer la simulation de données
        simulation_task = asyncio.create_task(mock.simulate_market_data(client, symbol, interval=1.0))
        
        # Attendre un peu pour laisser la simulation s'exécuter
        for i in range(60):
            await asyncio.sleep(1)
            
            # Vérifier le buffer toutes les 5 secondes
            if i % 5 == 0:
                logger.info(f"État du buffer après {i}s:")
                buffer_content = client.get_buffer_content(symbol, StreamType.KLINE)
                logger.info(f"  - Nombre d'éléments dans le buffer: {len(buffer_content)}")
                
                last_ts = client.get_last_timestamp(symbol, StreamType.KLINE)
                if last_ts:
                    last_dt = datetime.fromtimestamp(last_ts)
                    logger.info(f"  - Dernier timestamp: {last_dt.strftime('%H:%M:%S.%f')}")
                
        logger.info("Test terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur pendant le test: {str(e)}")
    finally:
        # Arrêter le client
        await client.close()


if __name__ == "__main__":
    import json
    import logging
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exécuter la boucle asyncio
    asyncio.run(main())
