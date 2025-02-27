"""
Module pour les stratégies basées sur le MACD (Moving Average Convergence Divergence).

Ce module fournit des fonctions pour calculer et analyser le MACD,
un indicateur technique populaire basé sur les moyennes mobiles exponentielles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger
from bitbot.strategie.base.EMA import EMAIndicator

class MACDSignalType(Enum):
    """Types de signaux MACD."""
    BUY = "Signal d'achat"
    SELL = "Signal de vente"
    NEUTRAL = "Signal neutre"
    BULLISH_DIVERGENCE = "Divergence haussière"
    BEARISH_DIVERGENCE = "Divergence baissière"

class MACDIndicator:
    """
    Classe pour calculer et analyser l'indicateur MACD
    (Moving Average Convergence Divergence).
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                signal_period: int = 9, clean_data: bool = True):
        """
        Initialise la classe des indicateurs MACD.
        
        Args:
            fast_period: Période pour l'EMA rapide
            slow_period: Période pour l'EMA lente
            signal_period: Période pour la ligne de signal
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.data_cleaner = DataCleaner() if clean_data else None
        self.ema_indicator = EMAIndicator(clean_data=clean_data)
        
    def set_parameters(self, fast_period: int = None, slow_period: int = None, 
                      signal_period: int = None) -> None:
        """
        Définit les paramètres du MACD.
        
        Args:
            fast_period: Période pour l'EMA rapide
            slow_period: Période pour l'EMA lente
            signal_period: Période pour la ligne de signal
        """
        if fast_period:
            self.fast_period = fast_period
        if slow_period:
            self.slow_period = slow_period
        if signal_period:
            self.signal_period = signal_period
            
        logger.info(f"Paramètres MACD définis: fast={self.fast_period}, "
                   f"slow={self.slow_period}, signal={self.signal_period}")
    
    def calculate_macd(self, data: Union[pd.DataFrame, MarketData],
                      column: str = 'close') -> pd.DataFrame:
        """
        Calcule l'indicateur MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            
        Returns:
            DataFrame avec les colonnes MACD, Signal et Histogramme ajoutées
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Calculer les EMA
        df = self.ema_indicator.calculate_ema(
            df, column, periods=[self.fast_period, self.slow_period]
        )
        
        # Calculer le MACD
        df['macd'] = df[f'ema_{self.fast_period}'] - df[f'ema_{self.slow_period}']
        
        # Calculer la ligne de signal (EMA du MACD)
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculer l'histogramme
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def detect_signal(self, data: Union[pd.DataFrame, MarketData]) -> MACDSignalType:
        """
        Détecte les signaux d'achat et de vente basés sur le MACD.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Type de signal détecté
        """
        # Calculer le MACD
        df = self.calculate_macd(data)
        
        # Vérifier s'il y a au moins 2 points de données
        if len(df) < 2:
            return MACDSignalType.NEUTRAL
        
        # Détecter les signaux
        if df['macd_hist'].iloc[-2] < 0 and df['macd_hist'].iloc[-1] > 0:
            return MACDSignalType.BUY
        elif df['macd_hist'].iloc[-2] > 0 and df['macd_hist'].iloc[-1] < 0:
            return MACDSignalType.SELL
        else:
            return MACDSignalType.NEUTRAL
    
    def detect_crossover(self, data: Union[pd.DataFrame, MarketData]) -> Tuple[bool, bool]:
        """
        Détecte les croisements entre la ligne MACD et la ligne de signal.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Tuple (bullish_crossover, bearish_crossover) indiquant si un croisement a été détecté
        """
        # Calculer le MACD
        df = self.calculate_macd(data)
        
        # Vérifier s'il y a au moins 2 points de données
        if len(df) < 2:
            return False, False
        
        # Détecter les croisements
        bullish_crossover = (df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and 
                           df['macd'].iloc[-1] > df['macd_signal'].iloc[-1])
        
        bearish_crossover = (df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and 
                           df['macd'].iloc[-1] < df['macd_signal'].iloc[-1])
        
        return bullish_crossover, bearish_crossover
    
    def detect_divergence(self, data: Union[pd.DataFrame, MarketData], 
                         lookback_period: int = 20) -> MACDSignalType:
        """
        Détecte les divergences entre le prix et le MACD.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            lookback_period: Période de recherche pour les divergences
            
        Returns:
            Type de divergence détecté
        """
        # Calculer le MACD
        df = self.calculate_macd(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            return MACDSignalType.NEUTRAL
        
        # Extraire la période de recherche
        df_period = df.iloc[-lookback_period:]
        
        # Trouver les sommets et creux locaux du prix
        price_highs = []
        price_lows = []
        
        for i in range(1, len(df_period) - 1):
            if df_period['close'].iloc[i] > df_period['close'].iloc[i-1] and df_period['close'].iloc[i] > df_period['close'].iloc[i+1]:
                price_highs.append((i, df_period['close'].iloc[i]))
            elif df_period['close'].iloc[i] < df_period['close'].iloc[i-1] and df_period['close'].iloc[i] < df_period['close'].iloc[i+1]:
                price_lows.append((i, df_period['close'].iloc[i]))
        
        # Trouver les sommets et creux locaux du MACD
        macd_highs = []
        macd_lows = []
        
        for i in range(1, len(df_period) - 1):
            if df_period['macd'].iloc[i] > df_period['macd'].iloc[i-1] and df_period['macd'].iloc[i] > df_period['macd'].iloc[i+1]:
                macd_highs.append((i, df_period['macd'].iloc[i]))
            elif df_period['macd'].iloc[i] < df_period['macd'].iloc[i-1] and df_period['macd'].iloc[i] < df_period['macd'].iloc[i+1]:
                macd_lows.append((i, df_period['macd'].iloc[i]))
        
        # Vérifier les divergences haussières (prix fait des creux plus bas, MACD fait des creux plus hauts)
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and macd_lows[-1][1] > macd_lows[-2][1]:
                return MACDSignalType.BULLISH_DIVERGENCE
        
        # Vérifier les divergences baissières (prix fait des sommets plus hauts, MACD fait des sommets plus bas)
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and macd_highs[-1][1] < macd_highs[-2][1]:
                return MACDSignalType.BEARISH_DIVERGENCE
        
        return MACDSignalType.NEUTRAL
    
    def get_histogram_strength(self, data: Union[pd.DataFrame, MarketData], 
                              lookback_period: int = 5) -> float:
        """
        Calcule la force de l'histogramme MACD (tendance et momentum).
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            lookback_period: Période de recherche pour la force
            
        Returns:
            Valeur de force (positive = haussier, négative = baissier)
        """
        # Calculer le MACD
        df = self.calculate_macd(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            return 0.0
        
        # Calculer la pente de l'histogramme
        hist_values = df['macd_hist'].iloc[-lookback_period:].values
        hist_slope = np.polyfit(range(lookback_period), hist_values, 1)[0]
        
        # Calculer la moyenne de l'histogramme
        hist_mean = hist_values.mean()
        
        # Combiner la pente et la moyenne pour obtenir la force
        strength = hist_slope * 10 + hist_mean
        
        return strength
    
    def analyze(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Analyse complète du MACD.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        results = {}
        
        # Calculer le MACD
        df = self.calculate_macd(data)
        
        # Obtenir les dernières valeurs
        results['macd'] = df['macd'].iloc[-1]
        results['signal'] = df['macd_signal'].iloc[-1]
        results['histogram'] = df['macd_hist'].iloc[-1]
        
        # Détecter les signaux
        results['signal_type'] = self.detect_signal(data)
        
        # Détecter les croisements
        bullish_crossover, bearish_crossover = self.detect_crossover(data)
        results['bullish_crossover'] = bullish_crossover
        results['bearish_crossover'] = bearish_crossover
        
        # Détecter les divergences
        results['divergence'] = self.detect_divergence(data)
        
        # Calculer la force
        results['strength'] = self.get_histogram_strength(data)
        
        return results
