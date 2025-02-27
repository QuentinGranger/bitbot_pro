"""
Modèles de données pour les informations de marché.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from decimal import Decimal

@dataclass
class Kline:
    """Chandelier (OHLCV)."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal
    trades: int
    taker_buy_volume: Decimal
    taker_buy_quote_volume: Decimal
    interval: str
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Kline':
        """
        Crée un Kline depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Kline
        """
        k = message['k']
        return cls(
            timestamp=datetime.fromtimestamp(k['t'] / 1000),
            open=Decimal(str(k['o'])),
            high=Decimal(str(k['h'])),
            low=Decimal(str(k['l'])),
            close=Decimal(str(k['c'])),
            volume=Decimal(str(k['v'])),
            close_time=datetime.fromtimestamp(k['T'] / 1000),
            quote_volume=Decimal(str(k['q'])),
            trades=k['n'],
            taker_buy_volume=Decimal(str(k['V'])),
            taker_buy_quote_volume=Decimal(str(k['Q'])),
            interval=k['i']
        )

@dataclass
class Trade:
    """Transaction."""
    timestamp: datetime
    symbol: str
    id: int
    price: Decimal
    quantity: Decimal
    buyer_maker: bool
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Trade':
        """
        Crée un Trade depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Trade
        """
        return cls(
            timestamp=datetime.fromtimestamp(message['T'] / 1000),
            symbol=message['s'],
            id=message['t'],
            price=Decimal(str(message['p'])),
            quantity=Decimal(str(message['q'])),
            buyer_maker=message['m']
        )

@dataclass
class OrderBook:
    """Carnet d'ordres."""
    symbol: str
    bids: Dict[float, float]  # prix -> quantité
    asks: Dict[float, float]  # prix -> quantité
    update_id: Optional[int] = None
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = {}
        self.asks = {}
        self.update_id = None
    
    def get_best_bid(self) -> tuple[float, float]:
        """
        Retourne le meilleur bid.
        
        Returns:
            (prix, quantité)
        """
        if not self.bids:
            return (0, 0)
        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])
    
    def get_best_ask(self) -> tuple[float, float]:
        """
        Retourne le meilleur ask.
        
        Returns:
            (prix, quantité)
        """
        if not self.asks:
            return (float('inf'), 0)
        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])
    
    def get_spread(self) -> float:
        """
        Calcule le spread.
        
        Returns:
            Spread en pourcentage
        """
        best_bid = self.get_best_bid()[0]
        best_ask = self.get_best_ask()[0]
        
        if best_bid == 0 or best_ask == float('inf'):
            return float('inf')
        
        return (best_ask - best_bid) / best_bid * 100

@dataclass
class Ticker:
    """Ticker 24h."""
    symbol: str
    price_change: Decimal
    price_change_percent: Decimal
    weighted_avg_price: Decimal
    prev_close_price: Decimal
    last_price: Decimal
    last_qty: Decimal
    bid_price: Decimal
    ask_price: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    open_time: datetime
    close_time: datetime
    first_id: int
    last_id: int
    count: int
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Ticker':
        """
        Crée un Ticker depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Ticker
        """
        return cls(
            symbol=message['s'],
            price_change=Decimal(str(message['p'])),
            price_change_percent=Decimal(str(message['P'])),
            weighted_avg_price=Decimal(str(message['w'])),
            prev_close_price=Decimal(str(message['x'])),
            last_price=Decimal(str(message['c'])),
            last_qty=Decimal(str(message['Q'])),
            bid_price=Decimal(str(message['b'])),
            ask_price=Decimal(str(message['a'])),
            open_price=Decimal(str(message['o'])),
            high_price=Decimal(str(message['h'])),
            low_price=Decimal(str(message['l'])),
            volume=Decimal(str(message['v'])),
            quote_volume=Decimal(str(message['q'])),
            open_time=datetime.fromtimestamp(message['O'] / 1000),
            close_time=datetime.fromtimestamp(message['C'] / 1000),
            first_id=message['F'],
            last_id=message['L'],
            count=message['n']
        )
