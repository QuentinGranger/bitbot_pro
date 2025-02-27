"""
Module pour les stratégies basées sur les moyennes mobiles simples (SMA).

Ce module fournit des fonctions pour calculer les moyennes mobiles simples (SMA)
sur différentes périodes, permettant d'identifier les micro et macro-tendances
dans les données de marché.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_merger import DataMerger
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class TrendType(Enum):
    """Types de tendances identifiables."""
    STRONG_UPTREND = "Tendance haussière forte"
    UPTREND = "Tendance haussière"
    WEAK_UPTREND = "Tendance haussière faible"
    SIDEWAYS = "Tendance latérale"
    WEAK_DOWNTREND = "Tendance baissière faible"
    DOWNTREND = "Tendance baissière"
    STRONG_DOWNTREND = "Tendance baissière forte"
    UNKNOWN = "Tendance inconnue"

class TimeFrame(Enum):
    """Échelles temporelles pour l'analyse des tendances."""
    MICRO = "micro"  # Court terme (minutes, heures)
    MESO = "meso"    # Moyen terme (jours)
    MACRO = "macro"  # Long terme (semaines, mois)

class SMAIndicator:
    """
    Classe pour calculer et analyser les moyennes mobiles simples (SMA)
    sur différentes périodes.
    """
    
    def __init__(self, clean_data: bool = True):
        """
        Initialise la classe des indicateurs de moyennes mobiles simples.
        
        Args:
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.data_cleaner = DataCleaner() if clean_data else None
        self.data_merger = DataMerger()
        self.sma_periods = [9, 20, 50, 100, 200]  # Périodes standard pour SMA
        
    def set_custom_periods(self, sma_periods: List[int] = None) -> None:
        """
        Définit des périodes personnalisées pour les moyennes mobiles.
        
        Args:
            sma_periods: Liste des périodes pour les SMA
        """
        if sma_periods:
            self.sma_periods = sorted(sma_periods)
            logger.info(f"Périodes SMA personnalisées définies: {self.sma_periods}")
    
    def calculate_sma(self, data: Union[pd.DataFrame, MarketData], 
                     column: str = 'close', 
                     periods: List[int] = None) -> pd.DataFrame:
        """
        Calcule les moyennes mobiles simples (SMA) pour les périodes spécifiées.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            periods: Liste des périodes pour lesquelles calculer les SMA
                    (utilise self.sma_periods si None)
        
        Returns:
            DataFrame avec les colonnes SMA ajoutées
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Utiliser les périodes par défaut si non spécifiées
        periods = periods or self.sma_periods
        
        # Calculer les SMA pour chaque période
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        
        return df
    
    def calculate_all(self, data: Union[pd.DataFrame, MarketData], 
                     column: str = 'close') -> pd.DataFrame:
        """
        Calcule les SMA pour toutes les périodes configurées.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
        
        Returns:
            DataFrame avec toutes les colonnes SMA ajoutées
        """
        return self.calculate_sma(data, column)
    
    def identify_trend(self, data: Union[pd.DataFrame, MarketData], 
                      timeframe: TimeFrame = TimeFrame.MESO) -> TrendType:
        """
        Identifie la tendance actuelle en fonction des moyennes mobiles.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            timeframe: Échelle temporelle pour l'analyse (micro, meso, macro)
        
        Returns:
            Type de tendance identifié
        """
        # Calculer toutes les moyennes mobiles
        df = self.calculate_all(data)
        
        # Sélectionner les périodes en fonction de l'échelle temporelle
        if timeframe == TimeFrame.MICRO:
            sma_short, sma_long = 'sma_9', 'sma_20'
        elif timeframe == TimeFrame.MESO:
            sma_short, sma_long = 'sma_20', 'sma_50'
        else:  # MACRO
            sma_short, sma_long = 'sma_50', 'sma_200'
        
        # Obtenir les dernières valeurs
        last_close = df['close'].iloc[-1]
        last_sma_short = df[sma_short].iloc[-1]
        last_sma_long = df[sma_long].iloc[-1]
        
        # Calculer les pentes des moyennes mobiles (sur les 5 dernières périodes)
        sma_short_slope = (df[sma_short].iloc[-1] - df[sma_short].iloc[-6]) / 5
        sma_long_slope = (df[sma_long].iloc[-1] - df[sma_long].iloc[-6]) / 5
        
        # Déterminer la tendance
        # Tendance haussière forte
        if (last_close > last_sma_short > last_sma_long and 
            sma_short_slope > 0 and sma_long_slope > 0):
            return TrendType.STRONG_UPTREND
        
        # Tendance haussière
        elif (last_close > last_sma_short > last_sma_long and 
              sma_short_slope > 0):
            return TrendType.UPTREND
        
        # Tendance haussière faible
        elif (last_close > last_sma_short and sma_short_slope > 0):
            return TrendType.WEAK_UPTREND
        
        # Tendance baissière forte
        elif (last_close < last_sma_short < last_sma_long and 
              sma_short_slope < 0 and sma_long_slope < 0):
            return TrendType.STRONG_DOWNTREND
        
        # Tendance baissière
        elif (last_close < last_sma_short < last_sma_long and 
              sma_short_slope < 0):
            return TrendType.DOWNTREND
        
        # Tendance baissière faible
        elif (last_close < last_sma_short and sma_short_slope < 0):
            return TrendType.WEAK_DOWNTREND
        
        # Tendance latérale
        elif abs(sma_short_slope) < 0.001:
            return TrendType.SIDEWAYS
        
        # Tendance indéterminée
        else:
            return TrendType.UNKNOWN
    
    def analyze_all_timeframes(self, data: Union[pd.DataFrame, MarketData]) -> Dict[TimeFrame, TrendType]:
        """
        Analyse les tendances sur toutes les échelles temporelles.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
        
        Returns:
            Dictionnaire des tendances identifiées pour chaque échelle temporelle
        """
        results = {}
        
        for timeframe in TimeFrame:
            trend = self.identify_trend(data, timeframe)
            results[timeframe] = trend
            
        return results
    
    def detect_crossover(self, data: Union[pd.DataFrame, MarketData], 
                        fast_period: int = 9, 
                        slow_period: int = 20) -> Tuple[bool, bool]:
        """
        Détecte les croisements de moyennes mobiles (golden cross et death cross).
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            fast_period: Période pour la moyenne mobile rapide
            slow_period: Période pour la moyenne mobile lente
        
        Returns:
            Tuple (golden_cross, death_cross) indiquant si un croisement a été détecté
        """
        # Calculer les moyennes mobiles
        df = self.calculate_sma(data, periods=[fast_period, slow_period])
        fast_ma = f'sma_{fast_period}'
        slow_ma = f'sma_{slow_period}'
        
        # Vérifier s'il y a au moins 2 points de données
        if len(df) < 2:
            return False, False
        
        # Vérifier les croisements
        golden_cross = (df[fast_ma].iloc[-2] <= df[slow_ma].iloc[-2] and 
                       df[fast_ma].iloc[-1] > df[slow_ma].iloc[-1])
        
        death_cross = (df[fast_ma].iloc[-2] >= df[slow_ma].iloc[-2] and 
                      df[fast_ma].iloc[-1] < df[slow_ma].iloc[-1])
        
        return golden_cross, death_cross

    def get_ma_distance(self, data: Union[pd.DataFrame, MarketData], 
                       period: int = 20) -> float:
        """
        Calcule la distance entre le prix actuel et une moyenne mobile.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            period: Période pour la moyenne mobile
        
        Returns:
            Distance en pourcentage entre le prix actuel et la moyenne mobile
        """
        # Calculer la moyenne mobile
        df = self.calculate_sma(data, periods=[period])
        ma_col = f'sma_{period}'
        
        # Calculer la distance
        last_close = df['close'].iloc[-1]
        last_ma = df[ma_col].iloc[-1]
        
        distance_pct = ((last_close - last_ma) / last_ma) * 100
        
        return distance_pct


class MultiTimeFrameSMA:
    """
    Classe pour l'analyse multi-timeframe utilisant les moyennes mobiles simples.
    """
    
    def __init__(self, clean_data: bool = True):
        """
        Initialise l'analyseur multi-timeframe.
        
        Args:
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.sma_indicator = SMAIndicator(clean_data=clean_data)
        self.data_merger = DataMerger()
    
    def analyze(self, market_data_dict: Dict[str, MarketData]) -> Dict[str, Dict[TimeFrame, TrendType]]:
        """
        Analyse les tendances sur plusieurs timeframes.
        
        Args:
            market_data_dict: Dictionnaire de MarketData pour différents timeframes
                             (clés = timeframes, valeurs = MarketData)
        
        Returns:
            Dictionnaire des tendances identifiées pour chaque timeframe
        """
        results = {}
        
        for timeframe, market_data in market_data_dict.items():
            timeframe_results = {}
            
            for tf in TimeFrame:
                trend = self.sma_indicator.identify_trend(market_data, tf)
                timeframe_results[tf] = trend
            
            results[timeframe] = timeframe_results
        
        return results
    
    def get_aligned_indicators(self, market_data_dict: Dict[str, MarketData], 
                              target_timeframe: str) -> pd.DataFrame:
        """
        Aligne les indicateurs de différents timeframes sur un timeframe cible.
        
        Args:
            market_data_dict: Dictionnaire de MarketData pour différents timeframes
            target_timeframe: Timeframe cible pour l'alignement
        
        Returns:
            DataFrame avec tous les indicateurs alignés
        """
        # Calculer les indicateurs pour chaque timeframe
        dfs = {}
        
        for timeframe, market_data in market_data_dict.items():
            df = self.sma_indicator.calculate_all(market_data)
            dfs[timeframe] = df
        
        # Aligner sur le timeframe cible
        aligned_df = self.data_merger.align_multi_timeframe_data(dfs, target_timeframe)
        
        return aligned_df
    
    def detect_trend_alignment(self, market_data_dict: Dict[str, MarketData]) -> bool:
        """
        Détecte si les tendances sont alignées sur tous les timeframes.
        
        Args:
            market_data_dict: Dictionnaire de MarketData pour différents timeframes
        
        Returns:
            True si toutes les tendances sont alignées, False sinon
        """
        # Analyser chaque timeframe
        results = self.analyze(market_data_dict)
        
        # Vérifier l'alignement des tendances
        trends = []
        
        for timeframe, timeframe_results in results.items():
            for tf, trend in timeframe_results.items():
                if trend in [TrendType.STRONG_UPTREND, TrendType.UPTREND, TrendType.WEAK_UPTREND]:
                    trends.append(1)  # Tendance haussière
                elif trend in [TrendType.STRONG_DOWNTREND, TrendType.DOWNTREND, TrendType.WEAK_DOWNTREND]:
                    trends.append(-1)  # Tendance baissière
                else:
                    trends.append(0)  # Tendance neutre
        
        # Vérifier si toutes les tendances sont dans la même direction
        return all(t == trends[0] for t in trends) and trends[0] != 0
