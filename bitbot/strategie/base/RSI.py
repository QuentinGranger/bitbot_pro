"""
Module pour calculer et analyser le Relative Strength Index (RSI).

Ce module fournit des fonctions pour calculer et analyser le RSI,
un indicateur technique qui mesure la vitesse et le changement des mouvements de prix.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class RSISignalType(Enum):
    """Types de signaux RSI."""
    BUY = "Signal d'achat"
    SELL = "Signal de vente"
    NEUTRAL = "Signal neutre"
    STRONG_BUY = "Signal d'achat fort"
    STRONG_SELL = "Signal de vente fort"

class TrendType(Enum):
    """Types de tendances du marché."""
    BULL = "Marché haussier"
    BEAR = "Marché baissier"
    RANGE = "Marché sans tendance forte"
    UNKNOWN = "Tendance inconnue"

class RSIIndicator:
    """
    Classe pour calculer et analyser l'indicateur RSI
    (Relative Strength Index).
    """
    
    def __init__(self, period: int = 14, 
                overbought_threshold: int = 70, 
                oversold_threshold: int = 30,
                strong_overbought_threshold: int = 80,
                strong_oversold_threshold: int = 20,
                clean_data: bool = True):
        """
        Initialise la classe des indicateurs RSI.
        
        Args:
            period: Période pour le calcul du RSI (par défaut 14)
            overbought_threshold: Niveau de surachat standard (par défaut 70)
            oversold_threshold: Niveau de survente standard (par défaut 30)
            strong_overbought_threshold: Niveau de surachat fort (par défaut 80)
            strong_oversold_threshold: Niveau de survente fort (par défaut 20)
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.period = period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.strong_overbought_threshold = strong_overbought_threshold
        self.strong_oversold_threshold = strong_oversold_threshold
        self.data_cleaner = DataCleaner() if clean_data else None
        
    def set_parameters(self, period: int = None, 
                     overbought_threshold: int = None, 
                     oversold_threshold: int = None,
                     strong_overbought_threshold: int = None, 
                     strong_oversold_threshold: int = None) -> None:
        """
        Définit les paramètres pour le calcul du RSI.
        
        Args:
            period: Période pour le calcul du RSI
            overbought_threshold: Niveau de surachat standard
            oversold_threshold: Niveau de survente standard
            strong_overbought_threshold: Niveau de surachat fort
            strong_oversold_threshold: Niveau de survente fort
        """
        if period is not None:
            self.period = period
        if overbought_threshold is not None:
            self.overbought_threshold = overbought_threshold
        if oversold_threshold is not None:
            self.oversold_threshold = oversold_threshold
        if strong_overbought_threshold is not None:
            self.strong_overbought_threshold = strong_overbought_threshold
        if strong_oversold_threshold is not None:
            self.strong_oversold_threshold = strong_oversold_threshold
            
    def calculate_rsi(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule le RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            DataFrame avec la colonne RSI ajoutée
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # S'assurer que le DataFrame a une colonne 'close'
        if 'close' not in df.columns:
            raise ValueError("Les données doivent contenir une colonne 'close'")
        
        # Calculer le RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def detect_trend(self, data: Union[pd.DataFrame, MarketData], 
                  lookback_period: int = 50, 
                  ema_short: int = 20, 
                  ema_long: int = 50) -> TrendType:
        """
        Détecte la tendance actuelle du marché.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            lookback_period: Période pour analyser la tendance
            ema_short: Période pour l'EMA courte
            ema_long: Période pour l'EMA longue
            
        Returns:
            Type de tendance (BULL, BEAR, RANGE)
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # S'assurer que nous avons assez de données
        if len(df) < lookback_period:
            return TrendType.UNKNOWN
        
        # Calculer les EMA pour détecter la tendance
        df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        # Analyser uniquement la période récente
        recent_df = df.iloc[-lookback_period:]
        
        # Calculer la pente des EMAs
        short_slope = (recent_df['ema_short'].iloc[-1] - recent_df['ema_short'].iloc[0]) / lookback_period
        long_slope = (recent_df['ema_long'].iloc[-1] - recent_df['ema_long'].iloc[0]) / lookback_period
        
        # Déterminer la tendance
        if short_slope > 0.001 and long_slope > 0.0005 and recent_df['ema_short'].iloc[-1] > recent_df['ema_long'].iloc[-1]:
            return TrendType.BULL
        elif short_slope < -0.001 and long_slope < -0.0005 and recent_df['ema_short'].iloc[-1] < recent_df['ema_long'].iloc[-1]:
            return TrendType.BEAR
        else:
            # Calculer la volatilité directionnelle pour confirmer le range
            price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()
            if price_range < 0.05:  # Moins de 5% de mouvement
                return TrendType.RANGE
            
            # Si la volatilité est plus élevée mais pas de tendance claire
            return TrendType.RANGE
    
    def get_dynamic_thresholds(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Ajuste dynamiquement les seuils de surachat/survente en fonction de la tendance.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Dictionnaire avec les seuils ajustés
        """
        trend = self.detect_trend(data)
        
        # Valeurs par défaut
        thresholds = {
            'overbought': self.overbought_threshold,
            'oversold': self.oversold_threshold,
            'strong_overbought': self.strong_overbought_threshold,
            'strong_oversold': self.strong_oversold_threshold,
            'trend': trend
        }
        
        # Ajuster les seuils en fonction de la tendance
        if trend == TrendType.BULL:
            # En marché haussier, le RSI reste plus élevé, ajuster les seuils à la hausse
            thresholds['overbought'] = min(80, self.overbought_threshold + 10)
            thresholds['oversold'] = min(40, self.oversold_threshold + 10)
            thresholds['strong_overbought'] = min(90, self.strong_overbought_threshold + 5)
            thresholds['strong_oversold'] = min(30, self.strong_oversold_threshold + 10)
        elif trend == TrendType.BEAR:
            # En marché baissier, le RSI reste plus bas, ajuster les seuils à la baisse
            thresholds['overbought'] = max(60, self.overbought_threshold - 10)
            thresholds['oversold'] = max(20, self.oversold_threshold - 10)
            thresholds['strong_overbought'] = max(70, self.strong_overbought_threshold - 5)
            thresholds['strong_oversold'] = max(10, self.strong_oversold_threshold - 5)
        
        return thresholds
    
    def is_overbought(self, data: Union[pd.DataFrame, MarketData], 
                    use_dynamic_thresholds: bool = True) -> Dict:
        """
        Détermine si le marché est en condition de surachat.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            use_dynamic_thresholds: Si True, utilise des seuils ajustés selon la tendance
            
        Returns:
            Dictionnaire avec les informations de surachat
        """
        df = self.calculate_rsi(data)
        current_rsi = df['rsi'].iloc[-1]
        
        if use_dynamic_thresholds:
            thresholds = self.get_dynamic_thresholds(data)
            overbought_threshold = thresholds['overbought']
            strong_overbought_threshold = thresholds['strong_overbought']
            trend = thresholds['trend']
        else:
            overbought_threshold = self.overbought_threshold
            strong_overbought_threshold = self.strong_overbought_threshold
            trend = self.detect_trend(data)
        
        is_overbought = current_rsi > overbought_threshold
        is_strong_overbought = current_rsi > strong_overbought_threshold
        
        return {
            'is_overbought': is_overbought,
            'is_strong_overbought': is_strong_overbought,
            'current_rsi': current_rsi,
            'overbought_threshold': overbought_threshold,
            'strong_overbought_threshold': strong_overbought_threshold,
            'trend': trend
        }
    
    def is_oversold(self, data: Union[pd.DataFrame, MarketData], 
                  use_dynamic_thresholds: bool = True) -> Dict:
        """
        Détermine si le marché est en condition de survente.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            use_dynamic_thresholds: Si True, utilise des seuils ajustés selon la tendance
            
        Returns:
            Dictionnaire avec les informations de survente
        """
        df = self.calculate_rsi(data)
        current_rsi = df['rsi'].iloc[-1]
        
        if use_dynamic_thresholds:
            thresholds = self.get_dynamic_thresholds(data)
            oversold_threshold = thresholds['oversold']
            strong_oversold_threshold = thresholds['strong_oversold']
            trend = thresholds['trend']
        else:
            oversold_threshold = self.oversold_threshold
            strong_oversold_threshold = self.strong_oversold_threshold
            trend = self.detect_trend(data)
        
        is_oversold = current_rsi < oversold_threshold
        is_strong_oversold = current_rsi < strong_oversold_threshold
        
        return {
            'is_oversold': is_oversold,
            'is_strong_oversold': is_strong_oversold,
            'current_rsi': current_rsi,
            'oversold_threshold': oversold_threshold,
            'strong_oversold_threshold': strong_oversold_threshold,
            'trend': trend
        }
    
    def get_signal(self, data: Union[pd.DataFrame, MarketData], 
                 use_dynamic_thresholds: bool = True) -> Dict:
        """
        Détermine le signal généré par le RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            use_dynamic_thresholds: Si True, utilise des seuils ajustés selon la tendance
            
        Returns:
            Dictionnaire avec le signal et les informations associées
        """
        overbought_info = self.is_overbought(data, use_dynamic_thresholds)
        oversold_info = self.is_oversold(data, use_dynamic_thresholds)
        
        current_rsi = overbought_info['current_rsi']  # Même valeur dans les deux dictionnaires
        trend = overbought_info['trend']  # Même valeur dans les deux dictionnaires
        
        # Déterminer le signal
        if oversold_info['is_strong_oversold']:
            signal = RSISignalType.STRONG_BUY
            message = f"RSI fortement survendu: {current_rsi:.2f} < {oversold_info['strong_oversold_threshold']}"
        elif oversold_info['is_oversold']:
            signal = RSISignalType.BUY
            message = f"RSI survendu: {current_rsi:.2f} < {oversold_info['oversold_threshold']}"
        elif overbought_info['is_strong_overbought']:
            signal = RSISignalType.STRONG_SELL
            message = f"RSI fortement suracheté: {current_rsi:.2f} > {overbought_info['strong_overbought_threshold']}"
        elif overbought_info['is_overbought']:
            signal = RSISignalType.SELL
            message = f"RSI suracheté: {current_rsi:.2f} > {overbought_info['overbought_threshold']}"
        else:
            signal = RSISignalType.NEUTRAL
            message = f"RSI neutre: {current_rsi:.2f}"
        
        # Ajouter les informations sur la tendance au message
        if trend == TrendType.BULL:
            trend_info = "(Marché haussier, seuils ajustés à la hausse)"
        elif trend == TrendType.BEAR:
            trend_info = "(Marché baissier, seuils ajustés à la baisse)"
        elif trend == TrendType.RANGE:
            trend_info = "(Marché sans tendance forte)"
        else:
            trend_info = ""
            
        if use_dynamic_thresholds and trend_info:
            message += f" {trend_info}"
        
        return {
            'signal': signal,
            'message': message,
            'current_rsi': current_rsi,
            'trend': trend,
            'overbought_info': overbought_info,
            'oversold_info': oversold_info
        }
    
    def analyze(self, data: Union[pd.DataFrame, MarketData], 
              use_dynamic_thresholds: bool = True) -> Dict:
        """
        Analyse complète du RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            use_dynamic_thresholds: Si True, utilise des seuils ajustés selon la tendance
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        # Calculer le RSI
        df = self.calculate_rsi(data)
        
        # Obtenir le signal
        signal_info = self.get_signal(data, use_dynamic_thresholds)
        
        # Calculer des statistiques sur le RSI
        rsi_values = df['rsi'].dropna()
        
        if len(rsi_values) > 0:
            rsi_mean = rsi_values.mean()
            rsi_std = rsi_values.std()
            rsi_min = rsi_values.min()
            rsi_max = rsi_values.max()
        else:
            rsi_mean = rsi_std = rsi_min = rsi_max = 0
        
        # Construire le résultat
        results = {
            'signal': signal_info['signal'],
            'message': signal_info['message'],
            'current_rsi': signal_info['current_rsi'],
            'trend': signal_info['trend'],
            'statistics': {
                'rsi_mean': rsi_mean,
                'rsi_std': rsi_std,
                'rsi_min': rsi_min,
                'rsi_max': rsi_max
            },
            'thresholds': {
                'overbought': signal_info['overbought_info']['overbought_threshold'],
                'oversold': signal_info['oversold_info']['oversold_threshold'],
                'strong_overbought': signal_info['overbought_info']['strong_overbought_threshold'],
                'strong_oversold': signal_info['oversold_info']['strong_oversold_threshold'],
                'is_dynamic': use_dynamic_thresholds
            }
        }
        
        return results
