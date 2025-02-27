"""
Client WebSocket Binance optimisé pour le trading haute fréquence.
Gère la connexion, la reconnexion et le traitement des données en temps réel.
"""

import json
import asyncio
import websockets
import logging
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Deque
from datetime import datetime, timedelta
from enum import Enum
import time
from dataclasses import dataclass
import hmac
import hashlib
from collections import defaultdict, deque
import pandas as pd

from bitbot.utils.logger import logger
from bitbot.models.market_data import Kline, Trade, OrderBook, Ticker
from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.anomaly_detector import AnomalyDetector, AnomalyType, Anomaly
from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType

class StreamType(Enum):
    """Types de flux de données disponibles."""
    TRADE = "trade"
    KLINE = "kline"
    DEPTH = "depth"
    BOOK_TICKER = "bookTicker"
    TICKER = "ticker"
    AGG_TRADE = "aggTrade"
    MINI_TICKER = "miniTicker"

@dataclass
class WebSocketConfig:
    """Configuration du client WebSocket."""
    base_endpoint: str = "wss://stream.binance.com:9443"
    ping_interval: int = 20  # secondes
    ping_timeout: int = 10   # secondes
    reconnect_delay: int = 5 # secondes
    max_reconnect_attempts: int = 5
    message_rate_limit: int = 5  # messages par seconde
    max_subscriptions: int = 1024
    buffer_size: int = 1000  # taille du buffer par stream
    sliding_window_size: int = 100  # taille de la fenêtre glissante
    data_retention_seconds: int = 300  # rétention des données en secondes (5 min)
    gap_detection_threshold: float = 2.0  # seuil pour détecter un gap (en secondes)
    backoff_factor: float = 1.5  # facteur pour le délai exponentiel
    max_backoff_delay: int = 300  # délai maximum en secondes (5 minutes)
    connection_health_check_interval: int = 30  # intervalle de vérification de la santé en secondes
    
    # Paramètres de détection d'anomalies
    volume_spike_z_score: float = 3.0  # Z-score pour considérer un spike de volume
    price_spike_z_score: float = 4.0   # Z-score pour considérer un spike de prix
    data_gap_threshold: float = 2.0    # Multiple de l'intervalle attendu pour considérer un gap
    anomaly_detection_enabled: bool = True  # Activer/désactiver la détection d'anomalies
    auto_recover_anomalies: bool = True  # Récupération automatique des données lors d'anomalies

class SlidingWindowBuffer:
    """
    Buffer avec fenêtre glissante pour stocker temporairement les données
    et pallier aux microcoupures réseau.
    """
    
    def __init__(self, max_size: int = 100, retention_seconds: int = 300):
        """
        Args:
            max_size: Taille maximale du buffer
            retention_seconds: Durée de rétention des données en secondes
        """
        self.data: Dict[str, Deque[Tuple[float, Any]]] = defaultdict(lambda: deque(maxlen=max_size))
        self.last_timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.retention_seconds = retention_seconds
    
    def add(self, stream_key: str, timestamp: float, data: Any) -> None:
        """
        Ajoute une donnée au buffer.
        
        Args:
            stream_key: Clé du stream
            timestamp: Timestamp de la donnée en secondes
            data: Donnée à stocker
        """
        self.data[stream_key].append((timestamp, data))
        self.last_timestamps[stream_key] = timestamp
    
    def get_since(self, stream_key: str, since_timestamp: float) -> List[Any]:
        """
        Récupère toutes les données depuis un timestamp donné.
        
        Args:
            stream_key: Clé du stream
            since_timestamp: Timestamp de départ en secondes
            
        Returns:
            Liste des données depuis le timestamp
        """
        if stream_key not in self.data:
            return []
        
        # Filtrer les données plus récentes que since_timestamp
        return [
            data for ts, data in self.data[stream_key]
            if ts >= since_timestamp
        ]
    
    def get_last_n(self, stream_key: str, n: int) -> List[Any]:
        """
        Récupère les n dernières données.
        
        Args:
            stream_key: Clé du stream
            n: Nombre de données à récupérer
            
        Returns:
            Liste des n dernières données
        """
        if stream_key not in self.data:
            return []
        
        items = list(self.data[stream_key])
        return [data for _, data in items[-n:]]
    
    def get_last_timestamp(self, stream_key: str) -> Optional[float]:
        """
        Récupère le timestamp de la dernière donnée.
        
        Args:
            stream_key: Clé du stream
            
        Returns:
            Dernier timestamp ou None
        """
        return self.last_timestamps.get(stream_key)
    
    def detect_gaps(self, stream_key: str, current_timestamp: float, expected_interval: float) -> bool:
        """
        Détecte s'il y a un gap dans les données.
        
        Args:
            stream_key: Clé du stream
            current_timestamp: Timestamp actuel en secondes
            expected_interval: Intervalle attendu entre deux données en secondes
            
        Returns:
            True s'il y a un gap, False sinon
        """
        if stream_key not in self.last_timestamps:
            return False
        
        last_ts = self.last_timestamps[stream_key]
        time_diff = current_timestamp - last_ts
        
        # Si l'écart est plus grand que 2 fois l'intervalle attendu, c'est un gap
        return time_diff > expected_interval * 2
    
    def clean_old_data(self) -> None:
        """Supprime les données plus anciennes que retention_seconds."""
        current_time = time.time()
        min_timestamp = current_time - self.retention_seconds
        
        for stream_key in list(self.data.keys()):
            while self.data[stream_key] and self.data[stream_key][0][0] < min_timestamp:
                self.data[stream_key].popleft()
    
    def get_between(self, stream_key: str, start_timestamp: float, end_timestamp: float) -> List[Any]:
        """
        Récupère toutes les données entre deux timestamps donné.
        
        Args:
            stream_key: Clé du stream
            start_timestamp: Timestamp de début en secondes
            end_timestamp: Timestamp de fin en secondes
            
        Returns:
            Liste des données entre les timestamps
        """
        if stream_key not in self.data:
            return []
        
        # Filtrer les données entre les timestamps donnés
        return [
            data for ts, data in self.data[stream_key]
            if start_timestamp <= ts <= end_timestamp
        ]

class BinanceWebSocket:
    """Client WebSocket Binance avec gestion avancée des connexions."""
    
    def __init__(self, config: WebSocketConfig):
        """
        Args:
            config: Configuration du client
        """
        self.config = config
        self.ws = None
        self.connected = False
        self.subscriptions: Set[str] = set()
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.connection_time = None
        self.last_ping_time = None
        self.last_pong_time = None
        self.rate_limiter = RateLimiter(
            rate=self.config.message_rate_limit,
            per=1
        )
        
        # Buffers pour les données
        self.kline_buffer: Dict[str, List[Kline]] = defaultdict(list)
        self.trade_buffer: Dict[str, List[Trade]] = defaultdict(list)
        self.orderbook_buffer: Dict[str, OrderBook] = {}
        self.sliding_window_buffer = SlidingWindowBuffer(
            max_size=self.config.sliding_window_size,
            retention_seconds=self.config.data_retention_seconds
        )
        
        # Détecteur d'anomalies
        if self.config.anomaly_detection_enabled:
            self.anomaly_detector = AnomalyDetector({
                "volume_spike_z_score": self.config.volume_spike_z_score,
                "price_spike_z_score": self.config.price_spike_z_score,
                "data_gap_threshold": self.config.data_gap_threshold,
            })
            # Callbacks pour les anomalies
            self.anomaly_callbacks: Dict[AnomalyType, List[Callable]] = defaultdict(list)
        else:
            self.anomaly_detector = None
        
        # Tâches asyncio
        self.tasks = []
        self.running = False
        
        # Statistiques
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'reconnections': 0,
            'latency': [],
            'connection_drops': 0,
            'ping_timeouts': 0,
            'last_disconnect_reason': None,
            'connection_uptime': 0,
            'reconnection_attempts': [],
            'anomalies_detected': 0,
            'anomalies_recovered': 0,
        }
        
        # Client REST pour récupérer les données historiques
        self.rest_client = None
        
        # État de la reconnexion
        self.reconnecting = False
        self.current_reconnect_delay = self.config.reconnect_delay

    async def connect(self):
        """Établit la connexion WebSocket avec gestion des erreurs."""
        try:
            self.ws = await websockets.connect(
                self.config.base_endpoint + "/ws",
                ping_interval=None,  # On gère nous-mêmes les pings
                ping_timeout=None,
                close_timeout=5  # Timeout pour la fermeture propre
            )
            self.connected = True
            self.reconnecting = False
            self.current_reconnect_delay = self.config.reconnect_delay  # Réinitialiser le délai
            self.connection_time = time.time()
            self.last_ping_time = time.time()
            self.last_pong_time = time.time()
            logger.info("Connexion WebSocket établie")
            
            # Démarrer les tâches de maintenance
            self.tasks = [
                asyncio.create_task(self._ping_loop()),
                asyncio.create_task(self._process_messages()),
                asyncio.create_task(self._monitor_connection()),
                asyncio.create_task(self._health_check())
            ]
            self.running = True
            
            # Réabonner aux streams précédents
            if self.subscriptions:
                await self._resubscribe()
            
        except Exception as e:
            logger.error(f"Erreur de connexion: {str(e)}")
            self.connected = False
            self.stats['last_disconnect_reason'] = str(e)
            await self._handle_connection_error()

    async def _ping_loop(self):
        """Envoie des pings réguliers pour maintenir la connexion."""
        while self.running:
            try:
                if self.connected:
                    await self.ws.ping()
                    self.last_ping_time = time.time()
                    
                    # Vérifier si on a reçu un pong pour le dernier ping
                    if self.last_pong_time is None or time.time() - self.last_pong_time > self.config.ping_timeout:
                        logger.warning(f"Pas de pong reçu depuis {time.time() - self.last_pong_time:.2f}s")
                        self.stats['ping_timeouts'] += 1
                        await self._handle_connection_error("Ping timeout")
                    
                    await asyncio.sleep(self.config.ping_interval)
            except Exception as e:
                logger.error(f"Erreur ping: {str(e)}")
                await self._handle_connection_error("Erreur ping")

    async def _process_messages(self):
        """Traite les messages entrants avec gestion de la latence."""
        while self.running:
            try:
                if not self.connected:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=self.config.ping_interval + self.config.ping_timeout
                )
                
                # Si on reçoit un pong (message vide ou spécifique de pong)
                if message == "" or message == "pong":
                    self.last_pong_time = time.time()
                    continue
                
                receive_time = time.time()
                
                # Mesurer la latence
                try:
                    data = json.loads(message)
                    if 'E' in data:  # Event time
                        latency = (receive_time * 1000) - data['E']  # en ms
                        self.stats['latency'].append(latency)
                        if len(self.stats['latency']) > 1000:
                            self.stats['latency'] = self.stats['latency'][-1000:]
                except:
                    pass
                
                self.stats['messages_received'] += 1
                
                # Traiter le message
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("Timeout en attente de message")
                await self._handle_connection_error("Timeout réception")
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"Connexion fermée: {str(e)}")
                await self._handle_connection_error("Connexion fermée")
            except Exception as e:
                logger.error(f"Erreur traitement message: {str(e)}")
                self.stats['errors'] += 1

    async def _handle_message(self, message: str):
        """
        Traite un message reçu et le dispatch aux callbacks appropriés.
        
        Args:
            message: Message JSON du WebSocket
        """
        try:
            data = json.loads(message)
            
            # Vérifier le rate limit
            if not await self.rate_limiter.acquire():
                logger.warning("Rate limit dépassé, message ignoré")
                return
            
            # Extraire le type de stream et le symbole
            stream_type = None
            symbol = None
            
            if 'e' in data:
                stream_type = data['e']
                symbol = data.get('s')
            elif 'stream' in data:
                stream_parts = data['stream'].split('@')
                symbol = stream_parts[0]
                stream_type = stream_parts[1]
            
            if not stream_type:
                return
            
            # Mettre à jour les buffers
            if stream_type == 'kline':
                kline = Kline.from_ws_message(data)
                self._update_buffer(
                    self.kline_buffer,
                    f"{symbol}_{kline.interval}",
                    kline
                )
            elif stream_type in ('trade', 'aggTrade'):
                trade = Trade.from_ws_message(data)
                self._update_buffer(
                    self.trade_buffer,
                    symbol,
                    trade
                )
            elif stream_type == 'depth':
                self._update_orderbook(symbol, data)
            
            # Ajouter les données à la fenêtre glissante
            stream_key = f"{symbol.lower()}@{stream_type}"
            self.sliding_window_buffer.add(stream_key, time.time(), data)
            
            # Détecter les anomalies
            if self.anomaly_detector and self.config.anomaly_detection_enabled:
                anomalies = self.anomaly_detector.add_data_point(symbol, stream_type, data)
                
                if anomalies:
                    self.stats['anomalies_detected'] += len(anomalies)
                    
                    # Journaliser les anomalies
                    for anomaly in anomalies:
                        severity_level = "ÉLEVÉE" if anomaly.severity >= 0.7 else "MOYENNE" if anomaly.severity >= 0.3 else "FAIBLE"
                        logger.warning(f"Anomalie détectée: {anomaly.anomaly_type.value} pour {symbol} - Sévérité {severity_level} ({anomaly.severity:.2f})")
                        
                        # Traiter automatiquement certaines anomalies
                        if self.config.auto_recover_anomalies:
                            asyncio.create_task(self._handle_anomaly(anomaly))
                        
                        # Appeler les callbacks spécifiques pour ce type d'anomalie
                        for callback in self.anomaly_callbacks.get(anomaly.anomaly_type, []):
                            try:
                                await callback(anomaly)
                            except Exception as e:
                                logger.error(f"Erreur callback anomalie: {str(e)}")
            
            # Appeler les callbacks
            for callback in self.callbacks.get(stream_key, []):
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Erreur callback: {str(e)}")
            
            self.stats['messages_processed'] += 1
            
        except json.JSONDecodeError:
            logger.error("Message JSON invalide")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Erreur traitement message: {str(e)}")
            self.stats['errors'] += 1

    def _update_buffer(self, buffer: Dict, key: str, item: Any):
        """
        Met à jour un buffer de données avec gestion de la taille.
        
        Args:
            buffer: Buffer à mettre à jour
            key: Clé du buffer
            item: Item à ajouter
        """
        buffer[key].append(item)
        if len(buffer[key]) > self.config.buffer_size:
            buffer[key] = buffer[key][-self.config.buffer_size:]
    
    def _update_orderbook(self, symbol: str, data: Dict):
        """
        Met à jour le carnet d'ordres local.
        
        Args:
            symbol: Symbole de trading
            data: Données de mise à jour
        """
        if symbol not in self.orderbook_buffer:
            self.orderbook_buffer[symbol] = OrderBook(symbol)
        
        orderbook = self.orderbook_buffer[symbol]
        
        # Mettre à jour les bids
        for bid in data.get('b', []):
            price, quantity = float(bid[0]), float(bid[1])
            if quantity > 0:
                orderbook.bids[price] = quantity
            else:
                orderbook.bids.pop(price, None)
        
        # Mettre à jour les asks
        for ask in data.get('a', []):
            price, quantity = float(ask[0]), float(ask[1])
            if quantity > 0:
                orderbook.asks[price] = quantity
            else:
                orderbook.asks.pop(price, None)
        
        orderbook.update_id = data['u']  # Update ID

    async def _handle_connection_error(self, reason="Erreur inconnue"):
        """
        Gère les erreurs de connexion avec tentatives de reconnexion et délai exponentiel.
        
        Args:
            reason: Raison de la déconnexion
        """
        self.connected = False
        disconnect_time = time.time()
        
        # Sauvegarder l'état précédent des derniers timestamps
        last_timestamps = {}
        for stream_type in StreamType:
            for symbol in [s.split('@')[0] for s in self.subscriptions if f"@{stream_type.value}" in s]:
                stream_key = f"{symbol.lower()}@{stream_type.value}"
                last_timestamp = self.sliding_window_buffer.get_last_timestamp(stream_key)
                if last_timestamp:
                    last_timestamps[stream_key] = last_timestamp
        
        self.stats['connection_drops'] += 1
        self.stats['last_disconnect_reason'] = reason
        
        logger.warning(f"Connexion WebSocket perdue: {reason}")
        
        # Notifier la déconnexion
        await notification_manager.notify_connection_issue(
            message=f"Connexion WebSocket interrompue: {reason}",
            details={
                "Heure de déconnexion": datetime.fromtimestamp(disconnect_time).strftime('%H:%M:%S'),
                "Streams actifs": len(self.subscriptions),
                "Durée de connexion": f"{int(disconnect_time - (self.connection_time or disconnect_time))}s"
            },
            critical=False  # Pas critique pour la première tentative
        )
        
        # Tentatives de reconnexion avec backoff exponentiel
        reconnect_attempts = 0
        while reconnect_attempts < self.config.max_reconnect_attempts and not self.connected:
            reconnect_attempts += 1
            self.stats['reconnection_attempts'].append(time.time())
            
            logger.info(f"Tentative de reconnexion {reconnect_attempts}/{self.config.max_reconnect_attempts} "
                      f"dans {self.current_reconnect_delay}s...")
            
            await asyncio.sleep(self.current_reconnect_delay)
            
            # Si c'est la dernière tentative, marquer comme critique
            if reconnect_attempts == self.config.max_reconnect_attempts - 1:
                await notification_manager.notify_connection_issue(
                    message="Dernière tentative de reconnexion WebSocket",
                    details={
                        "Tentative": f"{reconnect_attempts+1}/{self.config.max_reconnect_attempts}",
                        "Heure": datetime.now().strftime('%H:%M:%S'),
                        "Durée de déconnexion": f"{int(time.time() - disconnect_time)}s",
                        "Streams affectés": len(self.subscriptions)
                    },
                    critical=True  # Critique pour la dernière tentative
                )
            
            try:
                # Tenter de reconnecter
                await self.connect()
                if self.connected:
                    # Réussie, réinitialiser le délai
                    reconnect_time = time.time()
                    logger.info(f"Reconnexion réussie après {reconnect_attempts} tentatives "
                              f"({int(reconnect_time - disconnect_time)}s d'interruption)")
                    
                    self.current_reconnect_delay = self.config.reconnect_delay
                    self.stats['reconnections'] += 1
                    
                    # Se réabonner aux streams
                    await self._resubscribe()
                    
                    # Récupérer les données manquantes
                    if last_timestamps:
                        await self._recover_missing_data(last_timestamps, disconnect_time, reconnect_time)
                    
                    # Notifier la reconnexion réussie
                    await notification_manager.notify(
                        message=f"Connexion WebSocket rétablie après {int(reconnect_time - disconnect_time)}s",
                        priority=NotificationPriority.MEDIUM,
                        notification_type=NotificationType.CONNECTION,
                        title="Reconnexion Réussie",
                        details={
                            "Tentative": f"{reconnect_attempts}/{self.config.max_reconnect_attempts}",
                            "Streams réabonnés": len(self.subscriptions)
                        }
                    )
                    
                    return
            except Exception as e:
                logger.error(f"Échec de la tentative de reconnexion: {str(e)}")
            
            # Augmenter le délai de façon exponentielle
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * self.config.backoff_factor,
                self.config.max_backoff_delay
            )
        
        # Si on arrive ici, toutes les tentatives ont échoué
        logger.critical(f"Toutes les tentatives de reconnexion ont échoué après "
                      f"{int(time.time() - disconnect_time)}s. Abandon.")
        
        # Notifier l'échec final
        await notification_manager.notify_connection_issue(
            message="CRITIQUE: Échec de toutes les tentatives de reconnexion WebSocket",
            details={
                "Tentatives": self.config.max_reconnect_attempts,
                "Durée de déconnexion": f"{int(time.time() - disconnect_time)}s",
                "Streams perdus": len(self.subscriptions),
                "Action requise": "Intervention manuelle nécessaire"
            },
            critical=True
        )

    async def _monitor_connection(self):
        """Surveille l'état de la connexion et la durée de vie."""
        while self.running:
            try:
                if not self.connected or self.reconnecting:
                    await asyncio.sleep(1)
                    continue
                    
                # Vérifier la durée de vie de 24h
                if self.connection_time and time.time() - self.connection_time >= 86400:
                    logger.info("Reconnexion programmée (24h)")
                    await self.close()
                    await self.connect()
                
                # Vérifier le dernier ping/pong
                ping_pong_timeout = self.config.ping_interval + self.config.ping_timeout
                if (self.last_ping_time and 
                    time.time() - self.last_ping_time > ping_pong_timeout):
                    logger.warning("Ping timeout")
                    self.stats['ping_timeouts'] += 1
                    await self._handle_connection_error("Ping timeout")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur monitoring: {str(e)}")
    
    async def _health_check(self):
        """Vérifie régulièrement la santé de la connexion WebSocket."""
        while self.running:
            try:
                if self.connected and not self.reconnecting:
                    # Vérifier si on reçoit des messages
                    if (self.stats['messages_received'] > 0 and 
                        time.time() - self.last_pong_time > self.config.connection_health_check_interval):
                        logger.warning(f"Aucun message reçu depuis {time.time() - self.last_pong_time:.2f}s")
                        
                        # Envoyer un ping pour vérifier la connexion
                        try:
                            await self.ws.ping()
                            self.last_ping_time = time.time()
                        except Exception as e:
                            logger.error(f"Erreur health check ping: {str(e)}")
                            await self._handle_connection_error("Échec health check")
                
                # Nettoyer les anciennes données du buffer
                self.sliding_window_buffer.clean_old_data()
                
                await asyncio.sleep(self.config.connection_health_check_interval)
                
            except Exception as e:
                logger.error(f"Erreur health check: {str(e)}")

    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques du client.
        
        Returns:
            Statistiques de performance
        """
        stats = self.stats.copy()
        
        # Calculer les statistiques de latence
        if self.stats['latency']:
            stats['avg_latency'] = sum(self.stats['latency']) / len(self.stats['latency'])
            stats['max_latency'] = max(self.stats['latency'])
            stats['min_latency'] = min(self.stats['latency'])
        
        # Ajouter des informations sur la connexion actuelle
        stats['connected'] = self.connected
        stats['reconnecting'] = self.reconnecting
        
        if self.connection_time:
            stats['current_uptime'] = time.time() - self.connection_time
        
        # Limiter le nombre de tentatives de reconnexion dans les statistiques
        if len(stats['reconnection_attempts']) > 100:
            stats['reconnection_attempts'] = stats['reconnection_attempts'][-100:]
            
        return stats

    async def subscribe(self, symbol: str, stream_type: StreamType, callback: Optional[Callable] = None):
        """
        S'abonne à un stream.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
            callback: Fonction de callback optionnelle
        """
        if len(self.subscriptions) >= self.config.max_subscriptions:
            raise ValueError("Nombre maximum de subscriptions atteint")
        
        stream = f"{symbol.lower()}@{stream_type.value}"
        
        if callback:
            self.callbacks[stream].append(callback)
        
        if stream in self.subscriptions:
            return
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": len(self.subscriptions) + 1
        }
        
        await self._send_message(subscribe_message)
        self.subscriptions.add(stream)

    async def unsubscribe(self, symbol: str, stream_type: StreamType):
        """
        Se désabonne d'un stream.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
        """
        stream = f"{symbol.lower()}@{stream_type.value}"
        
        if stream not in self.subscriptions:
            return
        
        unsubscribe_message = {
            "method": "UNSUBSCRIBE",
            "params": [stream],
            "id": len(self.subscriptions) + 1
        }
        
        await self._send_message(unsubscribe_message)
        self.subscriptions.remove(stream)
        self.callbacks.pop(stream, None)

    async def _send_message(self, message: Dict):
        """
        Envoie un message sur le WebSocket avec gestion des erreurs.
        
        Args:
            message: Message à envoyer
        """
        if not self.connected:
            raise ConnectionError("WebSocket non connecté")
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Erreur envoi message: {str(e)}")
            await self._handle_connection_error()

    async def _resubscribe(self):
        """Réabonne à tous les streams après une reconnexion."""
        if not self.subscriptions:
            return
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": list(self.subscriptions),
            "id": 1
        }
        
        await self._send_message(subscribe_message)

    async def close(self):
        """Ferme proprement la connexion."""
        self.running = False
        
        # Annuler toutes les tâches
        for task in self.tasks:
            task.cancel()
        
        # Fermer le WebSocket
        if self.ws:
            await self.ws.close()
        
        self.connected = False
        logger.info("Connexion WebSocket fermée")

    def get_buffer_content(self, symbol: str, stream_type: StreamType, count: int = 10) -> List[Any]:
        """
        Récupère les dernières données du buffer pour un stream donné.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
            count: Nombre d'éléments à récupérer
            
        Returns:
            Liste des dernières données
        """
        stream_key = f"{symbol.lower()}@{stream_type.value}"
        return self.sliding_window_buffer.get_last_n(stream_key, count)
    
    def get_buffer_since(self, symbol: str, stream_type: StreamType, since_timestamp: float) -> List[Any]:
        """
        Récupère les données du buffer depuis un timestamp donné.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
            since_timestamp: Timestamp de départ en secondes
            
        Returns:
            Liste des données depuis le timestamp
        """
        stream_key = f"{symbol.lower()}@{stream_type.value}"
        return self.sliding_window_buffer.get_since(stream_key, since_timestamp)
    
    def get_last_timestamp(self, symbol: str, stream_type: StreamType) -> Optional[float]:
        """
        Récupère le timestamp de la dernière donnée reçue pour un stream.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
            
        Returns:
            Dernier timestamp ou None
        """
        stream_key = f"{symbol.lower()}@{stream_type.value}"
        return self.sliding_window_buffer.get_last_timestamp(stream_key)
    
    def detect_data_gap(self, symbol: str, stream_type: str, 
                       current_timestamp: float, expected_interval: float) -> bool:
        """
        Détecte s'il y a un gap dans les données pour un stream donné.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de stream
            current_timestamp: Timestamp actuel en secondes
            expected_interval: Intervalle attendu entre deux données en secondes
            
        Returns:
            True s'il y a un gap, False sinon
        """
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # Récupérer toutes les données dans l'intervalle
        data = self.sliding_window_buffer.get_between(stream_key, current_timestamp - expected_interval, current_timestamp)
        
        if not data:
            # Aucune donnée dans l'intervalle = gap
            return True
        
        # Extraire les timestamps
        timestamps = []
        for timestamp, item in data:
            timestamps.append(timestamp)
        
        if not timestamps:
            return True
            
        # Trier les timestamps
        timestamps.sort()
        
        # Si expected_interval n'est pas fourni, l'estimer
        if expected_interval is None:
            if len(timestamps) > 1:
                diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                expected_interval = sum(diffs) / len(diffs)
            else:
                # Impossible d'estimer, on retourne False par défaut
                return False
        
        # Calculer les différences successives
        diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        
        # Détecter des gaps en utilisant le seuil configuré
        threshold = expected_interval * self.config.data_gap_threshold
        
        for i, diff in enumerate(diffs):
            if diff > threshold:
                # Gap détecté
                logger.warning(f"Gap détecté dans {stream_key} entre "
                             f"{datetime.fromtimestamp(timestamps[i])} et "
                             f"{datetime.fromtimestamp(timestamps[i+1])}, "
                             f"durée: {diff:.2f}s (seuil: {threshold:.2f}s)")
                return True
        
        # Vérifier si le premier timestamp est trop éloigné du start_time
        if timestamps[0] - (current_timestamp - expected_interval) > threshold:
            logger.warning(f"Gap détecté au début de l'intervalle {stream_key} entre "
                         f"{datetime.fromtimestamp(current_timestamp - expected_interval)} et "
                         f"{datetime.fromtimestamp(timestamps[0])}, "
                         f"durée: {timestamps[0] - (current_timestamp - expected_interval):.2f}s (seuil: {threshold:.2f}s)")
            return True
            
        # Vérifier si le dernier timestamp est trop éloigné du end_time
        if current_timestamp - timestamps[-1] > threshold:
            logger.warning(f"Gap détecté à la fin de l'intervalle {stream_key} entre "
                         f"{datetime.fromtimestamp(timestamps[-1])} et "
                         f"{datetime.fromtimestamp(current_timestamp)}, "
                         f"durée: {current_timestamp - timestamps[-1]:.2f}s (seuil: {threshold:.2f}s)")
            return True
        
        # Pas de gap détecté
        return False
    
    def fill_historical_gap(self, 
                          symbol: str, 
                          stream_type: str, 
                          start_time: float, 
                          end_time: float) -> bool:
        """
        Lance une requête de récupération de données historiques pour combler un gap.
        
        Args:
            symbol: Symbole à compléter
            stream_type: Type de stream (kline, trade, etc.)
            start_time: Timestamp de début
            end_time: Timestamp de fin
            
        Returns:
            True si la récupération a été lancée, False sinon
        """
        if not self.rest_client:
            logger.warning("Client REST non configuré, impossible de récupérer les données historiques")
            return False
            
        # Lancer la tâche de récupération
        asyncio.create_task(self._recover_from_gap(symbol, start_time, end_time))
        return True
    
    def register_anomaly_callback(self, 
                                anomaly_type: AnomalyType, 
                                callback: Callable[[Anomaly], Any]) -> None:
        """
        Enregistre un callback pour un type d'anomalie spécifique.
        
        Args:
            anomaly_type: Type d'anomalie à monitorer
            callback: Fonction à appeler lorsqu'une anomalie est détectée
        """
        if not self.config.anomaly_detection_enabled:
            logger.warning("Détection d'anomalies désactivée, callback non enregistré")
            return
            
        self.anomaly_callbacks[anomaly_type].append(callback)
        logger.info(f"Callback enregistré pour les anomalies de type {anomaly_type.value}")
    
    def get_anomaly_stats(self) -> Dict:
        """
        Récupère les statistiques sur les anomalies détectées.
        
        Returns:
            Dictionnaire de statistiques
        """
        if not self.anomaly_detector:
            return {
                "enabled": False,
                "detected": 0,
                "recovered": 0
            }
            
        # Récupérer le résumé des anomalies pour la dernière heure
        anomalies_summary = self.anomaly_detector.get_anomaly_summary(window_seconds=3600)
        
        # Ajouter nos propres statistiques
        anomalies_summary.update({
            "enabled": True,
            "detected_total": self.stats['anomalies_detected'],
            "recovered_total": self.stats['anomalies_recovered'],
        })
        
        return anomalies_summary
    
    def detect_data_gap(self, 
                      symbol: str, 
                      stream_type: str, 
                      start_time: float, 
                      end_time: float, 
                      expected_interval: Optional[float] = None) -> bool:
        """
        Détecte si un gap existe dans les données entre deux timestamps.
        
        Cette méthode analyse les données de la fenêtre glissante pour déterminer
        s'il existe un gap significatif entre les timestamps fournis.
        
        Args:
            symbol: Symbole à vérifier
            stream_type: Type de stream (kline, trade, etc.)
            start_time: Timestamp de début
            end_time: Timestamp de fin
            expected_interval: Intervalle attendu entre deux points de données,
                              si None, estimé à partir des données
        
        Returns:
            True si un gap est détecté, False sinon
        """
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # Récupérer toutes les données dans l'intervalle
        data = self.sliding_window_buffer.get_between(stream_key, start_time, end_time)
        
        if not data:
            # Aucune donnée dans l'intervalle = gap
            return True
        
        # Extraire les timestamps
        timestamps = []
        for timestamp, item in data:
            timestamps.append(timestamp)
        
        if not timestamps:
            return True
            
        # Trier les timestamps
        timestamps.sort()
        
        # Si expected_interval n'est pas fourni, l'estimer
        if expected_interval is None:
            if len(timestamps) > 1:
                diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                expected_interval = sum(diffs) / len(diffs)
            else:
                # Impossible d'estimer, on retourne False par défaut
                return False
        
        # Calculer les différences successives
        diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        
        # Détecter des gaps en utilisant le seuil configuré
        threshold = expected_interval * self.config.data_gap_threshold
        
        for i, diff in enumerate(diffs):
            if diff > threshold:
                # Gap détecté
                logger.warning(f"Gap détecté dans {stream_key} entre "
                             f"{datetime.fromtimestamp(timestamps[i])} et "
                             f"{datetime.fromtimestamp(timestamps[i+1])}, "
                             f"durée: {diff:.2f}s (seuil: {threshold:.2f}s)")
                return True
        
        # Vérifier si le premier timestamp est trop éloigné du start_time
        if timestamps[0] - start_time > threshold:
            logger.warning(f"Gap détecté au début de l'intervalle {stream_key} entre "
                         f"{datetime.fromtimestamp(start_time)} et "
                         f"{datetime.fromtimestamp(timestamps[0])}, "
                         f"durée: {timestamps[0] - start_time:.2f}s (seuil: {threshold:.2f}s)")
            return True
            
        # Vérifier si le dernier timestamp est trop éloigné du end_time
        if end_time - timestamps[-1] > threshold:
            logger.warning(f"Gap détecté à la fin de l'intervalle {stream_key} entre "
                         f"{datetime.fromtimestamp(timestamps[-1])} et "
                         f"{datetime.fromtimestamp(end_time)}, "
                         f"durée: {end_time - timestamps[-1]:.2f}s (seuil: {threshold:.2f}s)")
            return True
        
        # Pas de gap détecté
        return False
    
    def fill_historical_gap(self, 
                          symbol: str, 
                          stream_type: str, 
                          start_time: float, 
                          end_time: float) -> bool:
        """
        Lance une requête de récupération de données historiques pour combler un gap.
        
        Args:
            symbol: Symbole à compléter
            stream_type: Type de stream (kline, trade, etc.)
            start_time: Timestamp de début
            end_time: Timestamp de fin
            
        Returns:
            True si la récupération a été lancée, False sinon
        """
        if not self.rest_client:
            logger.warning("Client REST non configuré, impossible de récupérer les données historiques")
            return False
            
        # Lancer la tâche de récupération
        asyncio.create_task(self._recover_from_gap(symbol, start_time, end_time))
        return True

    async def _handle_anomaly(self, anomaly: Anomaly):
        """
        Traite une anomalie détectée et tente de récupérer les données manquantes.
        
        Args:
            anomaly: Anomalie détectée
        """
        try:
            # Enregistrer l'anomalie dans les statistiques
            self.stats['anomalies'].append({
                'timestamp': anomaly.timestamp,
                'type': anomaly.anomaly_type.value,
                'symbol': anomaly.symbol,
                'severity': anomaly.severity,
                'details': anomaly.details
            })
            
            # Convertir la sévérité numérique en catégorie pour l'affichage
            severity_text = "FAIBLE"
            if anomaly.severity >= 0.7:
                severity_text = "ÉLEVÉE"
            elif anomaly.severity >= 0.4:
                severity_text = "MOYENNE"
                
            # Construire un message descriptif selon le type d'anomalie
            message = ""
            details = {
                "Sévérité": f"{anomaly.severity:.2f} ({severity_text})",
                "Timestamp": datetime.fromtimestamp(anomaly.timestamp).strftime('%H:%M:%S')
            }
            
            if anomaly.anomaly_type == AnomalyType.DATA_GAP:
                gap_duration = anomaly.details.get('gap_duration', 0)
                message = (f"Gap de données détecté sur {anomaly.symbol} "
                          f"(durée: {gap_duration:.1f}s)")
                details.update({
                    "Dernier timestamp": datetime.fromtimestamp(
                        anomaly.details.get('last_timestamp', 0)).strftime('%H:%M:%S'),
                    "Durée du gap": f"{gap_duration:.1f}s"
                })
                
                # Tenter de récupérer les données manquantes
                await self._recover_from_gap(
                    symbol=anomaly.symbol,
                    start_time=anomaly.details.get('last_timestamp'),
                    end_time=anomaly.timestamp
                )
                
            elif anomaly.anomaly_type == AnomalyType.VOLUME_SPIKE:
                volume = anomaly.details.get('volume', 0)
                avg_volume = anomaly.details.get('avg_volume', 0)
                ratio = volume / avg_volume if avg_volume else 0
                
                message = (f"Spike de volume détecté sur {anomaly.symbol} "
                          f"(×{ratio:.1f} la moyenne)")
                details.update({
                    "Volume": f"{volume:.8f}",
                    "Volume moyen": f"{avg_volume:.8f}",
                    "Ratio": f"{ratio:.1f}x"
                })
                
                # Valider le spike de volume pour les anomalies importantes
                if anomaly.severity >= 0.7:
                    await self._validate_volume_spike(anomaly)
                
            elif anomaly.anomaly_type == AnomalyType.PRICE_SPIKE:
                price = anomaly.details.get('price', 0)
                avg_price = anomaly.details.get('avg_price', 0)
                percentage = abs((price - avg_price) / avg_price * 100) if avg_price else 0
                
                message = (f"Spike de prix détecté sur {anomaly.symbol} "
                          f"(variation de {percentage:.1f}%)")
                details.update({
                    "Prix": f"{price:.8f}",
                    "Prix moyen": f"{avg_price:.8f}",
                    "Variation": f"{percentage:.1f}%"
                })
                
                # Valider le spike de prix pour les anomalies importantes
                if anomaly.severity >= 0.7:
                    await self._validate_price_spike(anomaly)
                
            elif anomaly.anomaly_type == AnomalyType.NEGATIVE_SPREAD:
                bid = anomaly.details.get('bid', 0)
                ask = anomaly.details.get('ask', 0)
                spread = bid - ask
                
                message = (f"Spread négatif détecté sur {anomaly.symbol} "
                          f"(spread: {spread:.8f})")
                details.update({
                    "Meilleur bid": f"{bid:.8f}",
                    "Meilleur ask": f"{ask:.8f}",
                    "Spread": f"{spread:.8f}"
                })
                
                # Pour un spread négatif, récupérer un nouveau snapshot du carnet d'ordres
                await self._recover_orderbook(anomaly.symbol)
            
            # Envoyer une notification Telegram pour l'anomalie
            await notification_manager.notify_anomaly(
                anomaly_type=anomaly.anomaly_type.value,
                symbol=anomaly.symbol,
                severity=severity_text,
                message=message,
                details=details
            )
            
            # Exécuter les callbacks enregistrés pour ce type d'anomalie
            for callback in self.anomaly_callbacks.get(anomaly.anomaly_type, []):
                try:
                    await callback(anomaly)
                except Exception as e:
                    logger.error(f"Erreur lors de l'exécution du callback d'anomalie: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'anomalie: {str(e)}")
