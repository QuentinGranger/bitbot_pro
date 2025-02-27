"""
Module de base pour les stratégies de trading.

Ce module fournit une classe de base pour toutes les stratégies de trading
dans le système BitBotPro.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from abc import ABC, abstractmethod

from bitbot.models.market_data import MarketData
from bitbot.models.trade_signal import TradeSignal
from bitbot.utils.logger import logger

class StrategyBase(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    """
    
    def __init__(self):
        """
        Initialise la classe de base de stratégie.
        """
        self.name = "BaseStrategy"
        self.description = "Stratégie de base"
        
    def set_parameters(self, **kwargs) -> None:
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Paramètres de la stratégie {self.name} mis à jour: {kwargs}")
    
    @abstractmethod
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur les données fournies.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Liste de signaux de trading
        """
        pass
    
    def backtest(self, data: Union[pd.DataFrame, MarketData], initial_capital: float = 10000.0) -> Dict:
        """
        Effectue un backtest simple de la stratégie sur les données historiques.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            initial_capital: Capital initial pour le backtest
            
        Returns:
            Dictionnaire contenant les résultats du backtest
        """
        # Méthode à implémenter par les classes dérivées
        raise NotImplementedError("La méthode backtest doit être implémentée par les classes dérivées")
    
    def validate(self, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Valide si les données sont suffisantes pour cette stratégie.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            True si les données sont valides, False sinon
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv
        else:
            df = data
        
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"La colonne '{col}' est requise pour la stratégie {self.name}")
                return False
        
        # Vérifier qu'il y a suffisamment de données
        min_data_points = 30  # Valeur par défaut, à ajuster par les classes dérivées
        if len(df) < min_data_points:
            logger.warning(f"Pas assez de données pour la stratégie {self.name}. "
                          f"Minimum requis: {min_data_points}, fourni: {len(df)}")
            return False
        
        return True
    
    def __str__(self) -> str:
        """
        Retourne une représentation sous forme de chaîne de la stratégie.
        
        Returns:
            Chaîne de caractères représentant la stratégie
        """
        return f"{self.name}: {self.description}"
