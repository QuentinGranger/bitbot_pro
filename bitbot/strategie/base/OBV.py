"""
Module pour calculer et analyser l'indicateur On-Balance Volume (OBV).

Ce module fournit des fonctions pour calculer l'OBV, un indicateur technique qui utilise
le flux de volume pour prédire les changements de prix d'un actif. L'OBV est basé sur
l'idée que les variations de volume peuvent précéder les mouvements de prix.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class OBVSignal(Enum):
    """Signaux générés par l'indicateur OBV."""
    STRONG_BUY = "Forte pression acheteuse"
    BUY = "Pression acheteuse"
    NEUTRAL = "Pression neutre"
    SELL = "Pression vendeuse"
    STRONG_SELL = "Forte pression vendeuse"

class OBVIndicator:
    """
    Classe pour calculer et analyser l'indicateur On-Balance Volume (OBV).
    """
    
    def __init__(self, ema_period: int = 20, signal_period: int = 9, clean_data: bool = True):
        """
        Initialise la classe des indicateurs OBV.
        
        Args:
            ema_period: Période pour le calcul de l'EMA de l'OBV (par défaut 20)
            signal_period: Période pour le calcul de la ligne de signal (par défaut 9)
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.ema_period = ema_period
        self.signal_period = signal_period
        self.data_cleaner = DataCleaner() if clean_data else None
        
    def set_parameters(self, ema_period: int = None, signal_period: int = None) -> None:
        """
        Définit les paramètres pour le calcul de l'OBV.
        
        Args:
            ema_period: Période pour le calcul de l'EMA de l'OBV
            signal_period: Période pour le calcul de la ligne de signal
        """
        if ema_period is not None:
            self.ema_period = ema_period
        if signal_period is not None:
            self.signal_period = signal_period
        
        logger.info(f"Paramètres de l'indicateur OBV définis: "
                   f"ema_period={self.ema_period}, signal_period={self.signal_period}")
    
    def calculate_obv(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule l'indicateur On-Balance Volume (OBV).
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec les colonnes OBV, OBV_EMA et OBV_Signal ajoutées
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
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La colonne '{col}' est requise pour calculer l'OBV")
        
        # Calculer l'OBV
        df['price_change'] = df['close'].diff()
        df['obv_change'] = np.where(df['price_change'] > 0, df['volume'],
                                   np.where(df['price_change'] < 0, -df['volume'], 0))
        df['OBV'] = df['obv_change'].cumsum()
        
        # Calculer l'EMA de l'OBV
        df['OBV_EMA'] = df['OBV'].ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculer la ligne de signal (EMA de l'OBV)
        df['OBV_Signal'] = df['OBV'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculer l'histogramme (différence entre OBV et Signal)
        df['OBV_Histogram'] = df['OBV'] - df['OBV_Signal']
        
        # Supprimer les colonnes temporaires
        df.drop(['price_change', 'obv_change'], axis=1, inplace=True)
        
        return df
    
    def get_signal(self, data: Union[pd.DataFrame, MarketData]) -> OBVSignal:
        """
        Détermine le signal généré par l'indicateur OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            Signal généré (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
        """
        # Calculer l'OBV
        df = self.calculate_obv(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < 2:
            return OBVSignal.NEUTRAL
        
        # Obtenir les dernières valeurs
        last_obv = df['OBV'].iloc[-1]
        last_ema = df['OBV_EMA'].iloc[-1]
        last_signal = df['OBV_Signal'].iloc[-1]
        last_histogram = df['OBV_Histogram'].iloc[-1]
        
        prev_obv = df['OBV'].iloc[-2]
        prev_ema = df['OBV_EMA'].iloc[-2]
        prev_signal = df['OBV_Signal'].iloc[-2]
        prev_histogram = df['OBV_Histogram'].iloc[-2]
        
        # Calculer la tendance de l'OBV (sur les 10 dernières périodes)
        obv_trend = df['OBV'].iloc[-10:].diff().mean()
        
        # Calculer le croisement de l'OBV et de la ligne de signal
        obv_crosses_signal_up = prev_obv < prev_signal and last_obv > last_signal
        obv_crosses_signal_down = prev_obv > prev_signal and last_obv < last_signal
        
        # Calculer la divergence entre l'OBV et le prix
        price_trend = df['close'].iloc[-10:].diff().mean()
        
        # Divergence haussière: prix baisse mais OBV monte
        bullish_divergence = price_trend < 0 and obv_trend > 0
        
        # Divergence baissière: prix monte mais OBV baisse
        bearish_divergence = price_trend > 0 and obv_trend < 0
        
        # Déterminer le signal
        if obv_crosses_signal_up and last_obv > last_ema and bullish_divergence:
            return OBVSignal.STRONG_BUY
        elif obv_crosses_signal_up or (last_obv > last_ema and last_histogram > 0 and last_histogram > prev_histogram):
            return OBVSignal.BUY
        elif obv_crosses_signal_down and last_obv < last_ema and bearish_divergence:
            return OBVSignal.STRONG_SELL
        elif obv_crosses_signal_down or (last_obv < last_ema and last_histogram < 0 and last_histogram < prev_histogram):
            return OBVSignal.SELL
        else:
            return OBVSignal.NEUTRAL
    
    def is_increasing(self, data: Union[pd.DataFrame, MarketData], lookback: int = 5) -> bool:
        """
        Détermine si l'OBV est en augmentation sur la période spécifiée.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            lookback: Nombre de périodes à considérer
            
        Returns:
            True si l'OBV est en augmentation, False sinon
        """
        # Calculer l'OBV
        df = self.calculate_obv(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback + 1:
            return False
        
        # Calculer la tendance de l'OBV sur la période spécifiée
        obv_values = df['OBV'].iloc[-lookback-1:]
        obv_trend = obv_values.diff().mean()
        
        return obv_trend > 0
    
    def is_decreasing(self, data: Union[pd.DataFrame, MarketData], lookback: int = 5) -> bool:
        """
        Détermine si l'OBV est en diminution sur la période spécifiée.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            lookback: Nombre de périodes à considérer
            
        Returns:
            True si l'OBV est en diminution, False sinon
        """
        # Calculer l'OBV
        df = self.calculate_obv(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback + 1:
            return False
        
        # Calculer la tendance de l'OBV sur la période spécifiée
        obv_values = df['OBV'].iloc[-lookback-1:]
        obv_trend = obv_values.diff().mean()
        
        return obv_trend < 0
    
    def detect_divergence(self, data: Union[pd.DataFrame, MarketData], 
                         lookback_period: int = 20) -> Tuple[bool, bool]:
        """
        Détecte les divergences entre le prix et l'OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            lookback_period: Période pour rechercher des divergences
            
        Returns:
            Tuple (bullish_divergence, bearish_divergence)
        """
        # Calculer l'OBV
        df = self.calculate_obv(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            return False, False
        
        # Extraire les données récentes pour l'analyse
        recent_data = df.iloc[-lookback_period:].copy()
        
        # Trouver les minimums locaux du prix et de l'OBV
        price_mins = []
        obv_mins = []
        
        for i in range(1, len(recent_data) - 1):
            # Minimum local du prix
            if (recent_data['close'].iloc[i] < recent_data['close'].iloc[i-1] and 
                recent_data['close'].iloc[i] < recent_data['close'].iloc[i+1]):
                price_mins.append((i, recent_data['close'].iloc[i]))
            
            # Minimum local de l'OBV
            if (recent_data['OBV'].iloc[i] < recent_data['OBV'].iloc[i-1] and 
                recent_data['OBV'].iloc[i] < recent_data['OBV'].iloc[i+1]):
                obv_mins.append((i, recent_data['OBV'].iloc[i]))
        
        # Trouver les maximums locaux du prix et de l'OBV
        price_maxs = []
        obv_maxs = []
        
        for i in range(1, len(recent_data) - 1):
            # Maximum local du prix
            if (recent_data['close'].iloc[i] > recent_data['close'].iloc[i-1] and 
                recent_data['close'].iloc[i] > recent_data['close'].iloc[i+1]):
                price_maxs.append((i, recent_data['close'].iloc[i]))
            
            # Maximum local de l'OBV
            if (recent_data['OBV'].iloc[i] > recent_data['OBV'].iloc[i-1] and 
                recent_data['OBV'].iloc[i] > recent_data['OBV'].iloc[i+1]):
                obv_maxs.append((i, recent_data['OBV'].iloc[i]))
        
        # Vérifier s'il y a au moins deux minimums/maximums
        if len(price_mins) < 2 or len(obv_mins) < 2 or len(price_maxs) < 2 or len(obv_maxs) < 2:
            return False, False
        
        # Vérifier la divergence haussière (prix fait des minimums plus bas, mais OBV fait des minimums plus hauts)
        bullish_divergence = False
        if (price_mins[-1][1] < price_mins[-2][1] and obv_mins[-1][1] > obv_mins[-2][1]):
            bullish_divergence = True
        
        # Vérifier la divergence baissière (prix fait des maximums plus hauts, mais OBV fait des maximums plus bas)
        bearish_divergence = False
        if (price_maxs[-1][1] > price_maxs[-2][1] and obv_maxs[-1][1] < obv_maxs[-2][1]):
            bearish_divergence = True
        
        return bullish_divergence, bearish_divergence
    
    def calculate_obv_momentum(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule le momentum de l'OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec la colonne OBV_Momentum ajoutée
        """
        # Calculer l'OBV
        df = self.calculate_obv(data)
        
        # Calculer le momentum comme le taux de variation de l'OBV
        df['OBV_Momentum'] = df['OBV'].pct_change(periods=5) * 100
        
        return df
    
    def calculate_volume_price_trend(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule l'indicateur Volume Price Trend (VPT), une variante de l'OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec la colonne VPT ajoutée
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Calculer le VPT
        df['percent_change'] = df['close'].pct_change()
        df['VPT_change'] = df['volume'] * df['percent_change']
        df['VPT'] = df['VPT_change'].cumsum()
        
        # Calculer l'EMA du VPT
        df['VPT_EMA'] = df['VPT'].ewm(span=self.ema_period, adjust=False).mean()
        
        # Supprimer les colonnes temporaires
        df.drop(['percent_change', 'VPT_change'], axis=1, inplace=True)
        
        return df
    
    def analyze(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Analyse complète de l'indicateur OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        results = {}
        
        # Calculer l'OBV
        df_obv = self.calculate_obv(data)
        
        # Obtenir les dernières valeurs
        results['OBV'] = df_obv['OBV'].iloc[-1]
        results['OBV_EMA'] = df_obv['OBV_EMA'].iloc[-1]
        results['OBV_Signal'] = df_obv['OBV_Signal'].iloc[-1]
        results['OBV_Histogram'] = df_obv['OBV_Histogram'].iloc[-1]
        
        # Déterminer le signal
        results['signal'] = self.get_signal(data)
        
        # Vérifier si l'OBV est en augmentation ou en diminution
        results['is_increasing'] = self.is_increasing(data)
        results['is_decreasing'] = self.is_decreasing(data)
        
        # Détecter les divergences
        bullish_divergence, bearish_divergence = self.detect_divergence(data)
        results['bullish_divergence'] = bullish_divergence
        results['bearish_divergence'] = bearish_divergence
        
        # Calculer le momentum
        df_momentum = self.calculate_obv_momentum(data)
        results['OBV_Momentum'] = df_momentum['OBV_Momentum'].iloc[-1]
        
        # Calculer le VPT
        df_vpt = self.calculate_volume_price_trend(data)
        results['VPT'] = df_vpt['VPT'].iloc[-1]
        results['VPT_EMA'] = df_vpt['VPT_EMA'].iloc[-1]
        
        # Calculer la force relative de l'OBV (comparaison avec sa moyenne)
        obv_values = df_obv['OBV'].iloc[-20:]
        obv_mean = obv_values.mean()
        results['OBV_Strength'] = (results['OBV'] / obv_mean - 1) * 100  # en pourcentage
        
        return results
