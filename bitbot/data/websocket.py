"""
Client WebSocket Binance optimisé pour le trading haute fréquence.
Gère la connexion, la reconnexion et le traitement des données en temps réel.
"""

import json
import asyncio
import websockets
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import time
from dataclasses import dataclass
import hmac
import hashlib
from collections import defaultdict
import pandas as pd

from bitbot.utils.logger import logger
from bitbot.models.market_data import Kline, Trade, OrderBook, Ticker
from bitbot.utils.rate_limiter import RateLimiter

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
        self.rate_limiter = RateLimiter(
            rate=self.config.message_rate_limit,
            per=1
        )
        
        # Buffers pour les données
        self.kline_buffer: Dict[str, List[Kline]] = defaultdict(list)
        self.trade_buffer: Dict[str, List[Trade]] = defaultdict(list)
        self.orderbook_buffer: Dict[str, OrderBook] = {}
        
        # Tâches asyncio
        self.tasks = []
        self.running = False
        
        # Statistiques
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'reconnections': 0,
            'latency': []
        }
    
    async def connect(self):
        """Établit la connexion WebSocket avec gestion des erreurs."""
        try:
            self.ws = await websockets.connect(
                self.config.base_endpoint + "/ws",
                ping_interval=None,  # On gère nous-mêmes les pings
                ping_timeout=None
            )
            self.connected = True
            self.connection_time = time.time()
            logger.info("Connexion WebSocket établie")
            
            # Démarrer les tâches de maintenance
            self.tasks = [
                asyncio.create_task(self._ping_loop()),
                asyncio.create_task(self._process_messages()),
                asyncio.create_task(self._monitor_connection())
            ]
            self.running = True
            
            # Réabonner aux streams précédents
            if self.subscriptions:
                await self._resubscribe()
            
        except Exception as e:
            logger.error(f"Erreur de connexion: {str(e)}")
            self.connected = False
            await self._handle_connection_error()
    
    async def _ping_loop(self):
        """Envoie des pings réguliers pour maintenir la connexion."""
        while self.running:
            try:
                if self.connected:
                    await self.ws.ping()
                    self.last_ping_time = time.time()
                    await asyncio.sleep(self.config.ping_interval)
            except Exception as e:
                logger.error(f"Erreur ping: {str(e)}")
                await self._handle_connection_error()
    
    async def _process_messages(self):
        """Traite les messages entrants avec gestion de la latence."""
        while self.running:
            try:
                if not self.connected:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self.ws.recv()
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
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connexion WebSocket fermée")
                await self._handle_connection_error()
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
            
            # Appeler les callbacks
            stream_key = f"{symbol}@{stream_type}"
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
    
    async def _handle_connection_error(self):
        """Gère les erreurs de connexion avec tentatives de reconnexion."""
        self.connected = False
        self.stats['reconnections'] += 1
        
        for attempt in range(self.config.max_reconnect_attempts):
            logger.info(f"Tentative de reconnexion {attempt + 1}/{self.config.max_reconnect_attempts}")
            try:
                await asyncio.sleep(self.config.reconnect_delay * (attempt + 1))
                await self.connect()
                if self.connected:
                    return
            except Exception as e:
                logger.error(f"Échec de la reconnexion: {str(e)}")
        
        logger.error("Échec de toutes les tentatives de reconnexion")
        raise ConnectionError("Impossible de rétablir la connexion WebSocket")
    
    async def _monitor_connection(self):
        """Surveille l'état de la connexion et la durée de vie."""
        while self.running:
            try:
                # Vérifier la durée de vie de 24h
                if self.connection_time and time.time() - self.connection_time >= 86400:
                    logger.info("Reconnexion programmée (24h)")
                    await self.close()
                    await self.connect()
                
                # Vérifier le dernier ping
                if (self.last_ping_time and 
                    time.time() - self.last_ping_time > self.config.ping_interval + self.config.ping_timeout):
                    logger.warning("Ping timeout")
                    await self._handle_connection_error()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur monitoring: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques du client.
        
        Returns:
            Statistiques de performance
        """
        stats = self.stats.copy()
        if self.stats['latency']:
            stats['avg_latency'] = sum(self.stats['latency']) / len(self.stats['latency'])
            stats['max_latency'] = max(self.stats['latency'])
            stats['min_latency'] = min(self.stats['latency'])
        return stats
    
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
