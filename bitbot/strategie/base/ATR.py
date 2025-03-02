"""
Module pour calculer et analyser l'Average True Range (ATR).

Ce module fournit des fonctions pour calculer l'ATR,
un indicateur technique qui mesure la volatilité du marché
en prenant en compte les gaps entre les sessions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class VolatilityLevel(Enum):
    """Niveaux de volatilité identifiables."""
    VERY_LOW = "Volatilité très faible"
    LOW = "Volatilité faible"
    MODERATE = "Volatilité modérée"
    HIGH = "Volatilité élevée"
    VERY_HIGH = "Volatilité très élevée"
    EXTREME = "Volatilité extrême"

class ATRIndicator:
    """
    Classe pour calculer et analyser l'Average True Range (ATR).
    """
    
    def __init__(self, period: int = 14, clean_data: bool = True):
        """
        Initialise la classe des indicateurs ATR.
        
        Args:
            period: Période pour le calcul de l'ATR (par défaut 14)
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.period = period
        self.data_cleaner = DataCleaner() if clean_data else None
        
    def set_period(self, period: int) -> None:
        """
        Définit la période pour le calcul de l'ATR.
        
        Args:
            period: Période pour le calcul de l'ATR
        """
        self.period = period
        logger.info(f"Période ATR définie: {self.period}")
    
    def calculate_atr(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule l'Average True Range (ATR).
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            DataFrame avec les colonnes TR (True Range) et ATR ajoutées
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La colonne '{col}' est requise pour calculer l'ATR")
        
        # Calculer le True Range (TR)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Supprimer les colonnes temporaires
        df.drop(['tr0', 'tr1', 'tr2'], axis=1, inplace=True)
        
        # Calculer l'ATR (moyenne mobile exponentielle du TR)
        df['atr'] = df['tr'].ewm(span=self.period, adjust=False).mean()
        
        # Calculer l'ATR en pourcentage du prix
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        return df
    
    def get_volatility_level(self, data: Union[pd.DataFrame, MarketData], 
                            lookback_period: int = 100) -> VolatilityLevel:
        """
        Détermine le niveau de volatilité actuel par rapport à l'historique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            lookback_period: Période historique pour comparer la volatilité
            
        Returns:
            Niveau de volatilité actuel
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            lookback_period = len(df)
            logger.warning(f"Période de lookback ajustée à {lookback_period} en raison du manque de données")
        
        # Obtenir l'ATR actuel et historique
        current_atr_pct = df['atr_pct'].iloc[-1]
        historical_atr_pct = df['atr_pct'].iloc[-lookback_period:-1]
        
        # Calculer les percentiles
        p20 = np.percentile(historical_atr_pct, 20)
        p40 = np.percentile(historical_atr_pct, 40)
        p60 = np.percentile(historical_atr_pct, 60)
        p80 = np.percentile(historical_atr_pct, 80)
        p95 = np.percentile(historical_atr_pct, 95)
        
        # Déterminer le niveau de volatilité
        if current_atr_pct < p20:
            return VolatilityLevel.VERY_LOW
        elif current_atr_pct < p40:
            return VolatilityLevel.LOW
        elif current_atr_pct < p60:
            return VolatilityLevel.MODERATE
        elif current_atr_pct < p80:
            return VolatilityLevel.HIGH
        elif current_atr_pct < p95:
            return VolatilityLevel.VERY_HIGH
        else:
            return VolatilityLevel.EXTREME
    
    def calculate_volatility_change(self, data: Union[pd.DataFrame, MarketData], 
                                  lookback_period: int = 5) -> float:
        """
        Calcule le changement de volatilité sur une période donnée.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            lookback_period: Période pour calculer le changement de volatilité
            
        Returns:
            Pourcentage de changement de l'ATR
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period + 1:
            lookback_period = len(df) - 1
            logger.warning(f"Période de lookback ajustée à {lookback_period} en raison du manque de données")
        
        # Calculer le changement de volatilité
        current_atr = df['atr'].iloc[-1]
        past_atr = df['atr'].iloc[-(lookback_period+1)]
        
        volatility_change_pct = ((current_atr - past_atr) / past_atr) * 100
        
        return volatility_change_pct
    
    def calculate_atr_bands(self, data: Union[pd.DataFrame, MarketData], 
                           multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calcule les bandes ATR autour du prix.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            multiplier: Multiplicateur pour les bandes ATR
            
        Returns:
            DataFrame avec les colonnes upper_band et lower_band ajoutées
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Calculer les bandes
        df['upper_band'] = df['close'] + (df['atr'] * multiplier)
        df['lower_band'] = df['close'] - (df['atr'] * multiplier)
        
        return df
    
    def is_volatility_breakout(self, data: Union[pd.DataFrame, MarketData], 
                              threshold_pct: float = 50.0,
                              lookback_period: int = 20) -> bool:
        """
        Détecte si une augmentation significative de la volatilité s'est produite.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            threshold_pct: Seuil de pourcentage d'augmentation pour considérer un breakout
            lookback_period: Période pour calculer la volatilité moyenne de référence
            
        Returns:
            True si un breakout de volatilité est détecté, False sinon
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period + 1:
            return False
        
        # Calculer la moyenne de l'ATR sur la période de lookback
        avg_atr = df['atr'].iloc[-lookback_period-1:-1].mean()
        
        # Obtenir l'ATR actuel
        current_atr = df['atr'].iloc[-1]
        
        # Vérifier si l'ATR actuel dépasse le seuil
        return (current_atr - avg_atr) / avg_atr * 100 > threshold_pct
    
    def calculate_atr_stop_loss(self, data: Union[pd.DataFrame, MarketData], 
                               multiplier: float = 3.0, 
                               is_long: bool = True) -> float:
        """
        Calcule un niveau de stop loss basé sur l'ATR.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            multiplier: Multiplicateur pour l'ATR
            is_long: True pour une position longue, False pour une position courte
            
        Returns:
            Niveau de prix pour le stop loss
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Obtenir le dernier prix et ATR
        last_close = df['close'].iloc[-1]
        last_atr = df['atr'].iloc[-1]
        
        # Calculer le stop loss
        if is_long:
            stop_loss = last_close - (last_atr * multiplier)
        else:
            stop_loss = last_close + (last_atr * multiplier)
        
        return stop_loss
    
    def calculate_position_size(self, data: Union[pd.DataFrame, MarketData], 
                              risk_pct: float = 1.0, 
                              account_size: float = 10000.0,
                              stop_multiplier: float = 2.0,
                              is_long: bool = True) -> Tuple[float, float]:
        """
        Calcule la taille de position optimale basée sur l'ATR et le risque.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            risk_pct: Pourcentage du capital à risquer par trade
            account_size: Taille du compte/capital disponible
            stop_multiplier: Multiplicateur pour l'ATR pour le stop loss
            is_long: True pour une position longue, False pour une position courte
            
        Returns:
            Tuple (position_size, stop_loss_price)
        """
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Obtenir le dernier prix et ATR
        last_close = df['close'].iloc[-1]
        last_atr = df['atr'].iloc[-1]
        
        # Calculer le stop loss
        if is_long:
            stop_loss = last_close - (last_atr * stop_multiplier)
        else:
            stop_loss = last_close + (last_atr * stop_multiplier)
        
        # Calculer le risque par unité
        risk_per_unit = abs(last_close - stop_loss)
        
        # Calculer le montant de risque
        risk_amount = account_size * (risk_pct / 100)
        
        # Calculer la taille de position
        position_size = risk_amount / risk_per_unit
        
        return position_size, stop_loss
    
    def analyze(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Analyse complète de l'ATR.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        results = {}
        
        # Calculer l'ATR
        df = self.calculate_atr(data)
        
        # Obtenir les dernières valeurs
        results['atr'] = df['atr'].iloc[-1]
        results['atr_pct'] = df['atr_pct'].iloc[-1]
        
        # Déterminer le niveau de volatilité
        results['volatility_level'] = self.get_volatility_level(data)
        
        # Calculer le changement de volatilité
        results['volatility_change_pct'] = self.calculate_volatility_change(data)
        
        # Vérifier s'il y a un breakout de volatilité
        results['is_volatility_breakout'] = self.is_volatility_breakout(data)
        
        # Calculer les niveaux de stop loss
        results['long_stop_loss'] = self.calculate_atr_stop_loss(data, is_long=True)
        results['short_stop_loss'] = self.calculate_atr_stop_loss(data, is_long=False)
        
        # Calculer la taille de position recommandée (pour un compte de 10000)
        position_size, _ = self.calculate_position_size(data)
        results['recommended_position_size'] = position_size
        
        return results

# Importer la stratégie ATR Stop Loss à la fin du module
# Cela permet d'accéder à toutes les fonctionnalités via le module ATR sans duplication de code
from bitbot.strategie.indicators.atr_stop_loss_strategy import (
    ATRStopLossStrategy,
    StopLossType,
    TrailingSLMode
)

# Exposer les classes directement dans le module ATR pour une utilisation plus facile
__all__ = [
    'VolatilityLevel',
    'ATRIndicator',
    'ATRStopLossStrategy',
    'StopLossType',
    'TrailingSLMode'
]
