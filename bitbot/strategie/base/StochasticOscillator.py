"""
Module pour calculer et analyser l'oscillateur stochastique.

Ce module fournit des fonctions pour calculer l'oscillateur stochastique,
un indicateur technique qui permet d'identifier les retournements de tendance
à court terme et les zones de surachat et de survente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class StochasticSignal(Enum):
    """Signaux générés par l'oscillateur stochastique."""
    BUY = "Signal d'achat"
    SELL = "Signal de vente"
    NEUTRAL = "Signal neutre"
    STRONG_BUY = "Signal d'achat fort"
    STRONG_SELL = "Signal de vente fort"

class StochasticOscillatorIndicator:
    """
    Classe pour calculer et analyser l'oscillateur stochastique.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3, 
                overbought: int = 80, oversold: int = 20, clean_data: bool = True):
        """
        Initialise la classe des indicateurs de l'oscillateur stochastique.
        
        Args:
            k_period: Période pour le calcul de %K (par défaut 14)
            d_period: Période pour le calcul de %D (par défaut 3)
            slowing: Période de ralentissement (par défaut 3)
            overbought: Niveau de surachat (par défaut 80)
            oversold: Niveau de survente (par défaut 20)
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.overbought = overbought
        self.oversold = oversold
        self.data_cleaner = DataCleaner() if clean_data else None
        
    def set_parameters(self, k_period: int = None, d_period: int = None, 
                      slowing: int = None, overbought: int = None, 
                      oversold: int = None) -> None:
        """
        Définit les paramètres pour le calcul de l'oscillateur stochastique.
        
        Args:
            k_period: Période pour le calcul de %K
            d_period: Période pour le calcul de %D
            slowing: Période de ralentissement
            overbought: Niveau de surachat
            oversold: Niveau de survente
        """
        if k_period is not None:
            self.k_period = k_period
        if d_period is not None:
            self.d_period = d_period
        if slowing is not None:
            self.slowing = slowing
        if overbought is not None:
            self.overbought = overbought
        if oversold is not None:
            self.oversold = oversold
        
        logger.info(f"Paramètres de l'oscillateur stochastique définis: %K={self.k_period}, "
                   f"%D={self.d_period}, slowing={self.slowing}, "
                   f"overbought={self.overbought}, oversold={self.oversold}")
    
    def calculate_stochastic(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule l'oscillateur stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            DataFrame avec les colonnes %K et %D ajoutées
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
                raise ValueError(f"La colonne '{col}' est requise pour calculer l'oscillateur stochastique")
        
        # Calculer le plus haut et le plus bas sur la période k_period
        df['lowest_low'] = df['low'].rolling(window=self.k_period).min()
        df['highest_high'] = df['high'].rolling(window=self.k_period).max()
        
        # Calculer %K (stochastique rapide)
        df['%K_fast'] = 100 * ((df['close'] - df['lowest_low']) / 
                              (df['highest_high'] - df['lowest_low']))
        
        # Appliquer le ralentissement (slowing) pour obtenir %K (stochastique lent)
        df['%K'] = df['%K_fast'].rolling(window=self.slowing).mean()
        
        # Calculer %D (moyenne mobile de %K)
        df['%D'] = df['%K'].rolling(window=self.d_period).mean()
        
        # Supprimer les colonnes temporaires
        df.drop(['lowest_low', 'highest_high', '%K_fast'], axis=1, inplace=True)
        
        return df
    
    def get_signal(self, data: Union[pd.DataFrame, MarketData]) -> StochasticSignal:
        """
        Détermine le signal généré par l'oscillateur stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Signal généré (BUY, SELL, NEUTRAL, STRONG_BUY, STRONG_SELL)
        """
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < 2:
            return StochasticSignal.NEUTRAL
        
        # Obtenir les dernières valeurs
        last_k = df['%K'].iloc[-1]
        last_d = df['%D'].iloc[-1]
        prev_k = df['%K'].iloc[-2]
        prev_d = df['%D'].iloc[-2]
        
        # Signal de croisement de %K et %D
        k_crosses_d_up = prev_k < prev_d and last_k > last_d
        k_crosses_d_down = prev_k > prev_d and last_k < last_d
        
        # Signal basé sur les niveaux de surachat/survente
        if last_k < self.oversold and last_d < self.oversold:
            if k_crosses_d_up:
                return StochasticSignal.STRONG_BUY
            return StochasticSignal.BUY
        elif last_k > self.overbought and last_d > self.overbought:
            if k_crosses_d_down:
                return StochasticSignal.STRONG_SELL
            return StochasticSignal.SELL
        elif k_crosses_d_up:
            return StochasticSignal.BUY
        elif k_crosses_d_down:
            return StochasticSignal.SELL
        else:
            return StochasticSignal.NEUTRAL
    
    def is_overbought(self, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Détermine si le marché est en condition de surachat.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            True si le marché est en surachat, False sinon
        """
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Vérifier s'il y a assez de données
        if df['%K'].isna().iloc[-1] or df['%D'].isna().iloc[-1]:
            return False
        
        # Vérifier si %K et %D sont au-dessus du niveau de surachat
        return df['%K'].iloc[-1] > self.overbought and df['%D'].iloc[-1] > self.overbought
    
    def is_oversold(self, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Détermine si le marché est en condition de survente.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            True si le marché est en survente, False sinon
        """
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Vérifier s'il y a assez de données
        if df['%K'].isna().iloc[-1] or df['%D'].isna().iloc[-1]:
            return False
        
        # Vérifier si %K et %D sont en-dessous du niveau de survente
        return df['%K'].iloc[-1] < self.oversold and df['%D'].iloc[-1] < self.oversold
    
    def detect_divergence(self, data: Union[pd.DataFrame, MarketData], 
                         lookback_period: int = 20) -> Tuple[bool, bool]:
        """
        Détecte les divergences entre le prix et l'oscillateur stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            lookback_period: Période pour rechercher des divergences
            
        Returns:
            Tuple (bullish_divergence, bearish_divergence)
        """
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            return False, False
        
        # Extraire les données récentes pour l'analyse
        recent_data = df.iloc[-lookback_period:].copy()
        
        # Trouver les minimums locaux du prix et de %K
        price_mins = []
        k_mins = []
        
        for i in range(1, len(recent_data) - 1):
            # Minimum local du prix
            if (recent_data['close'].iloc[i] < recent_data['close'].iloc[i-1] and 
                recent_data['close'].iloc[i] < recent_data['close'].iloc[i+1]):
                price_mins.append((i, recent_data['close'].iloc[i]))
            
            # Minimum local de %K
            if (recent_data['%K'].iloc[i] < recent_data['%K'].iloc[i-1] and 
                recent_data['%K'].iloc[i] < recent_data['%K'].iloc[i+1]):
                k_mins.append((i, recent_data['%K'].iloc[i]))
        
        # Trouver les maximums locaux du prix et de %K
        price_maxs = []
        k_maxs = []
        
        for i in range(1, len(recent_data) - 1):
            # Maximum local du prix
            if (recent_data['close'].iloc[i] > recent_data['close'].iloc[i-1] and 
                recent_data['close'].iloc[i] > recent_data['close'].iloc[i+1]):
                price_maxs.append((i, recent_data['close'].iloc[i]))
            
            # Maximum local de %K
            if (recent_data['%K'].iloc[i] > recent_data['%K'].iloc[i-1] and 
                recent_data['%K'].iloc[i] > recent_data['%K'].iloc[i+1]):
                k_maxs.append((i, recent_data['%K'].iloc[i]))
        
        # Vérifier s'il y a au moins deux minimums/maximums
        if len(price_mins) < 2 or len(k_mins) < 2 or len(price_maxs) < 2 or len(k_maxs) < 2:
            return False, False
        
        # Vérifier la divergence haussière (prix fait des minimums plus bas, mais %K fait des minimums plus hauts)
        bullish_divergence = False
        if (price_mins[-1][1] < price_mins[-2][1] and k_mins[-1][1] > k_mins[-2][1]):
            bullish_divergence = True
        
        # Vérifier la divergence baissière (prix fait des maximums plus hauts, mais %K fait des maximums plus bas)
        bearish_divergence = False
        if (price_maxs[-1][1] > price_maxs[-2][1] and k_maxs[-1][1] < k_maxs[-2][1]):
            bearish_divergence = True
        
        return bullish_divergence, bearish_divergence
    
    def calculate_stochastic_momentum(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule l'indice de momentum stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            DataFrame avec la colonne stoch_momentum ajoutée
        """
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Calculer le momentum comme la différence entre %K et %D
        df['stoch_momentum'] = df['%K'] - df['%D']
        
        return df
    
    def calculate_stochastic_rsi(self, data: Union[pd.DataFrame, MarketData], 
                               rsi_period: int = 14) -> pd.DataFrame:
        """
        Calcule le Stochastic RSI, une combinaison de l'oscillateur stochastique et du RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            rsi_period: Période pour le calcul du RSI
            
        Returns:
            DataFrame avec les colonnes RSI, StochRSI_K et StochRSI_D ajoutées
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Calculer le RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculer le Stochastic RSI
        df['lowest_rsi'] = df['RSI'].rolling(window=self.k_period).min()
        df['highest_rsi'] = df['RSI'].rolling(window=self.k_period).max()
        
        df['StochRSI_K_fast'] = 100 * ((df['RSI'] - df['lowest_rsi']) / 
                                      (df['highest_rsi'] - df['lowest_rsi']))
        
        # Appliquer le ralentissement (slowing)
        df['StochRSI_K'] = df['StochRSI_K_fast'].rolling(window=self.slowing).mean()
        
        # Calculer %D pour le Stochastic RSI
        df['StochRSI_D'] = df['StochRSI_K'].rolling(window=self.d_period).mean()
        
        # Supprimer les colonnes temporaires
        df.drop(['lowest_rsi', 'highest_rsi', 'StochRSI_K_fast'], axis=1, inplace=True)
        
        return df
    
    def analyze(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Analyse complète de l'oscillateur stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        results = {}
        
        # Calculer l'oscillateur stochastique
        df = self.calculate_stochastic(data)
        
        # Obtenir les dernières valeurs
        results['%K'] = df['%K'].iloc[-1]
        results['%D'] = df['%D'].iloc[-1]
        
        # Déterminer le signal
        results['signal'] = self.get_signal(data)
        
        # Vérifier les conditions de surachat/survente
        results['is_overbought'] = self.is_overbought(data)
        results['is_oversold'] = self.is_oversold(data)
        
        # Détecter les divergences
        bullish_divergence, bearish_divergence = self.detect_divergence(data)
        results['bullish_divergence'] = bullish_divergence
        results['bearish_divergence'] = bearish_divergence
        
        # Calculer le momentum
        df_momentum = self.calculate_stochastic_momentum(data)
        results['stoch_momentum'] = df_momentum['stoch_momentum'].iloc[-1]
        
        # Calculer le Stochastic RSI
        df_stoch_rsi = self.calculate_stochastic_rsi(data)
        results['RSI'] = df_stoch_rsi['RSI'].iloc[-1]
        results['StochRSI_K'] = df_stoch_rsi['StochRSI_K'].iloc[-1]
        results['StochRSI_D'] = df_stoch_rsi['StochRSI_D'].iloc[-1]
        
        return results
