"""
Exemple d'utilisation du client WebSocket pour le trading en temps réel.
"""

import asyncio
import pandas as pd
from decimal import Decimal
from typing import Dict, Optional
import signal
import sys

from bitbot.data.websocket import BinanceWebSocket, WebSocketConfig, StreamType
from bitbot.models.market_data import Kline, Trade, OrderBook, Ticker
from bitbot.utils.logger import logger

class RealTimeStrategy:
    """Stratégie de trading temps réel utilisant les WebSockets."""
    
    def __init__(self, symbol: str):
        """
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDT')
        """
        self.symbol = symbol
        self.ws_config = WebSocketConfig(
            ping_interval=20,
            ping_timeout=10,
            reconnect_delay=5,
            max_reconnect_attempts=5,
            message_rate_limit=5
        )
        self.ws_client = BinanceWebSocket(self.ws_config)
        
        # État du marché
        self.current_price: Optional[Decimal] = None
        self.orderbook: Optional[OrderBook] = None
        self.last_kline: Optional[Kline] = None
        self.trades: list[Trade] = []
        self.ticker: Optional[Ticker] = None
        
        # Métriques de performance
        self.message_count = 0
        self.start_time = None
        
        # Contrôle d'exécution
        self.running = False
    
    async def start(self):
        """Démarre la stratégie."""
        logger.info(f"Démarrage de la stratégie sur {self.symbol}")
        
        # Gérer l'arrêt propre
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda: asyncio.create_task(self.stop())
            )
        
        try:
            # Connexion WebSocket
            await self.ws_client.connect()
            
            # Abonnement aux streams
            await self._subscribe_to_streams()
            
            self.running = True
            self.start_time = pd.Timestamp.now()
            
            # Boucle principale
            while self.running:
                # Afficher les statistiques périodiquement
                await self._display_stats()
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Erreur dans la stratégie: {str(e)}")
            await self.stop()
    
    async def stop(self):
        """Arrête proprement la stratégie."""
        logger.info("Arrêt de la stratégie")
        self.running = False
        await self.ws_client.close()
    
    async def _subscribe_to_streams(self):
        """S'abonne à tous les streams nécessaires."""
        # Stream de trades
        await self.ws_client.subscribe(
            self.symbol,
            StreamType.TRADE,
            self._handle_trade
        )
        
        # Stream de klines 1m
        await self.ws_client.subscribe(
            self.symbol,
            StreamType.KLINE,
            self._handle_kline
        )
        
        # Stream du carnet d'ordres
        await self.ws_client.subscribe(
            self.symbol,
            StreamType.DEPTH,
            self._handle_orderbook
        )
        
        # Stream du ticker
        await self.ws_client.subscribe(
            self.symbol,
            StreamType.TICKER,
            self._handle_ticker
        )
    
    async def _handle_trade(self, data: Dict):
        """
        Traite un nouveau trade.
        
        Args:
            data: Données du trade
        """
        trade = Trade.from_ws_message(data)
        self.trades.append(trade)
        self.current_price = trade.price
        
        # Garder uniquement les 1000 derniers trades
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]
        
        self.message_count += 1
        await self._analyze_trade(trade)
    
    async def _handle_kline(self, data: Dict):
        """
        Traite une nouvelle kline.
        
        Args:
            data: Données de la kline
        """
        kline = Kline.from_ws_message(data)
        self.last_kline = kline
        
        self.message_count += 1
        await self._analyze_kline(kline)
    
    async def _handle_orderbook(self, data: Dict):
        """
        Traite une mise à jour du carnet d'ordres.
        
        Args:
            data: Données du carnet
        """
        if not self.orderbook:
            self.orderbook = OrderBook(self.symbol)
        
        # Mettre à jour le carnet
        if 'b' in data:  # Bids
            for bid in data['b']:
                price, quantity = float(bid[0]), float(bid[1])
                if quantity > 0:
                    self.orderbook.bids[price] = quantity
                else:
                    self.orderbook.bids.pop(price, None)
        
        if 'a' in data:  # Asks
            for ask in data['a']:
                price, quantity = float(ask[0]), float(ask[1])
                if quantity > 0:
                    self.orderbook.asks[price] = quantity
                else:
                    self.orderbook.asks.pop(price, None)
        
        self.message_count += 1
        await self._analyze_orderbook()
    
    async def _handle_ticker(self, data: Dict):
        """
        Traite une mise à jour du ticker.
        
        Args:
            data: Données du ticker
        """
        self.ticker = Ticker.from_ws_message(data)
        self.message_count += 1
        await self._analyze_ticker()
    
    async def _analyze_trade(self, trade: Trade):
        """
        Analyse un nouveau trade.
        
        Args:
            trade: Trade à analyser
        """
        # Exemple d'analyse
        if trade.quantity > Decimal('1.0'):  # Trade important
            logger.info(
                f"Trade important détecté: {trade.quantity} {self.symbol} "
                f"à {trade.price} ({trade.timestamp})"
            )
    
    async def _analyze_kline(self, kline: Kline):
        """
        Analyse une nouvelle kline.
        
        Args:
            kline: Kline à analyser
        """
        # Exemple d'analyse
        price_change = ((kline.close - kline.open) / kline.open) * Decimal('100')
        if abs(price_change) > Decimal('0.5'):  # Mouvement de prix significatif
            logger.info(
                f"Mouvement de prix significatif: {price_change:.2f}% "
                f"sur {kline.interval}"
            )
    
    async def _analyze_orderbook(self):
        """Analyse le carnet d'ordres."""
        if not self.orderbook:
            return
        
        # Exemple d'analyse
        spread = self.orderbook.get_spread()
        if spread > 0.1:  # Spread important
            logger.info(f"Spread important détecté: {spread:.2f}%")
    
    async def _analyze_ticker(self):
        """Analyse le ticker."""
        if not self.ticker:
            return
        
        # Exemple d'analyse
        if abs(float(self.ticker.price_change_percent)) > 1.0:
            logger.info(
                f"Variation 24h significative: {self.ticker.price_change_percent}% "
                f"sur {self.symbol}"
            )
    
    async def _display_stats(self):
        """Affiche les statistiques de performance."""
        if not self.start_time:
            return
        
        elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return
        
        messages_per_second = self.message_count / elapsed
        
        # Obtenir les stats WebSocket
        ws_stats = self.ws_client.get_statistics()
        avg_latency = ws_stats.get('avg_latency', 0)
        
        logger.info(
            f"Stats: {messages_per_second:.1f} msg/s, "
            f"Latence moyenne: {avg_latency:.1f}ms, "
            f"Reconnexions: {ws_stats['reconnections']}"
        )

async def main():
    """Point d'entrée principal."""
    # Créer et démarrer la stratégie
    strategy = RealTimeStrategy("BTCUSDT")
    await strategy.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt manuel")
