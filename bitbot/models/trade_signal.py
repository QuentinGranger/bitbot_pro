"""
Module pour les signaux de trading.

Ce module fournit des classes pour représenter les signaux de trading
générés par les stratégies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from enum import Enum
from datetime import datetime
import uuid

class SignalType(Enum):
    """Types de signaux de trading."""
    BUY = "Achat"
    SELL = "Vente"
    STRONG_BUY = "Achat fort"
    STRONG_SELL = "Vente forte"
    NEUTRAL = "Neutre"
    EXIT = "Sortie"
    STOP_LOSS = "Stop loss"
    TAKE_PROFIT = "Prise de profit"

class TradeSignal:
    """
    Classe représentant un signal de trading.
    """
    
    def __init__(self, symbol: str, timeframe: str, timestamp: Union[datetime, pd.Timestamp],
                signal_type: SignalType, price: float, confidence: float = 0.5,
                source: str = "unknown", metadata: Dict[str, Any] = None):
        """
        Initialise un signal de trading.
        
        Args:
            symbol: Symbole du marché (par exemple, "BTCUSDT")
            timeframe: Période temporelle (par exemple, "1h", "4h", "1d")
            timestamp: Horodatage du signal
            signal_type: Type de signal (achat, vente, etc.)
            price: Prix au moment du signal
            confidence: Niveau de confiance du signal (entre 0 et 1)
            source: Source du signal (nom de la stratégie, indicateur, etc.)
            metadata: Métadonnées supplémentaires associées au signal
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.price = price
        self.confidence = max(0.0, min(1.0, confidence))  # Limiter entre 0 et 1
        self.source = source
        self.metadata = metadata or {}
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """
        Convertit le signal en dictionnaire.
        
        Returns:
            Dictionnaire représentant le signal
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.name,
            "signal_value": self.signal_type.value,
            "price": self.price,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeSignal':
        """
        Crée un signal à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire contenant les données du signal
            
        Returns:
            Instance de TradeSignal
        """
        signal = cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            timestamp=pd.to_datetime(data["timestamp"]),
            signal_type=SignalType[data["signal_type"]],
            price=data["price"],
            confidence=data["confidence"],
            source=data["source"],
            metadata=data.get("metadata", {})
        )
        signal.id = data["id"]
        signal.created_at = pd.to_datetime(data["created_at"])
        return signal
    
    def __str__(self) -> str:
        """
        Retourne une représentation sous forme de chaîne du signal.
        
        Returns:
            Chaîne de caractères représentant le signal
        """
        return (f"Signal {self.signal_type.value} pour {self.symbol} ({self.timeframe}) "
                f"à {self.timestamp} - Prix: {self.price}, Confiance: {self.confidence:.2f}")
    
    def __repr__(self) -> str:
        """
        Retourne une représentation sous forme de chaîne du signal.
        
        Returns:
            Chaîne de caractères représentant le signal
        """
        return self.__str__()
