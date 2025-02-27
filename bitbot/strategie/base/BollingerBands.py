"""
Module pour calculer et analyser les bandes de Bollinger.

Ce module fournit des fonctions pour calculer les bandes de Bollinger,
un indicateur technique qui permet d'identifier les zones de surachat et de survente,
ainsi que la volatilité du marché.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class MarketCondition(Enum):
    """États du marché identifiables par les bandes de Bollinger."""
    OVERSOLD = "Survendu"
    OVERBOUGHT = "Suracheté"
    NEUTRAL = "Neutre"
    SQUEEZE = "Compression"
    EXPANSION = "Expansion"

class BollingerBandsIndicator:
    """
    Classe pour calculer et analyser les bandes de Bollinger.
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0, clean_data: bool = True):
        """
        Initialise la classe des indicateurs de bandes de Bollinger.
        
        Args:
            period: Période pour le calcul de la moyenne mobile (par défaut 20)
            num_std: Nombre d'écarts-types pour les bandes (par défaut 2.0)
            clean_data: Si True, nettoie automatiquement les données avant calcul
        """
        self.period = period
        self.num_std = num_std
        self.data_cleaner = DataCleaner() if clean_data else None
        
    def set_parameters(self, period: int = None, num_std: float = None) -> None:
        """
        Définit les paramètres pour le calcul des bandes de Bollinger.
        
        Args:
            period: Période pour le calcul de la moyenne mobile
            num_std: Nombre d'écarts-types pour les bandes
        """
        if period is not None:
            self.period = period
        if num_std is not None:
            self.num_std = num_std
        logger.info(f"Paramètres des bandes de Bollinger définis: période={self.period}, écarts-types={self.num_std}")
    
    def calculate_bollinger_bands(self, data: Union[pd.DataFrame, MarketData], 
                                 column: str = 'close') -> pd.DataFrame:
        """
        Calcule les bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            
        Returns:
            DataFrame avec les colonnes middle_band, upper_band, lower_band et bandwidth ajoutées
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            df = self.data_cleaner.clean_market_data(data).ohlcv if isinstance(data, MarketData) else df
        
        # Vérifier que la colonne nécessaire existe
        if column not in df.columns:
            raise ValueError(f"La colonne '{column}' est requise pour calculer les bandes de Bollinger")
        
        # Calculer la moyenne mobile
        df['middle_band'] = df[column].rolling(window=self.period).mean()
        
        # Calculer l'écart-type
        df['std_dev'] = df[column].rolling(window=self.period).std()
        
        # Calculer les bandes supérieure et inférieure
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * self.num_std)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * self.num_std)
        
        # Calculer la largeur des bandes (bandwidth)
        df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['middle_band'] * 100
        
        # Calculer le %B (position relative du prix dans les bandes)
        df['percent_b'] = (df[column] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        return df
    
    def identify_market_condition(self, data: Union[pd.DataFrame, MarketData], 
                                 column: str = 'close',
                                 overbought_threshold: float = 0.8,
                                 oversold_threshold: float = 0.2,
                                 squeeze_threshold: float = 2.0) -> MarketCondition:
        """
        Identifie la condition du marché basée sur les bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            overbought_threshold: Seuil pour considérer le marché comme suracheté
            oversold_threshold: Seuil pour considérer le marché comme survendu
            squeeze_threshold: Seuil pour détecter une compression des bandes
            
        Returns:
            Condition du marché (OVERSOLD, OVERBOUGHT, NEUTRAL, SQUEEZE, EXPANSION)
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Obtenir les dernières valeurs
        last_price = df[column].iloc[-1]
        last_upper = df['upper_band'].iloc[-1]
        last_lower = df['lower_band'].iloc[-1]
        last_bandwidth = df['bandwidth'].iloc[-1]
        last_percent_b = df['percent_b'].iloc[-1]
        
        # Calculer la largeur moyenne des bandes sur les 20 dernières périodes
        avg_bandwidth = df['bandwidth'].iloc[-20:].mean()
        
        # Détecter une compression des bandes (squeeze)
        if last_bandwidth < avg_bandwidth / squeeze_threshold:
            return MarketCondition.SQUEEZE
        
        # Détecter une expansion des bandes
        if last_bandwidth > avg_bandwidth * 1.5:
            return MarketCondition.EXPANSION
        
        # Détecter un marché suracheté
        if last_percent_b > overbought_threshold:
            return MarketCondition.OVERBOUGHT
        
        # Détecter un marché survendu
        if last_percent_b < oversold_threshold:
            return MarketCondition.OVERSOLD
        
        # Marché neutre
        return MarketCondition.NEUTRAL
    
    def detect_bollinger_breakout(self, data: Union[pd.DataFrame, MarketData], 
                                column: str = 'close',
                                lookback: int = 3) -> Tuple[bool, bool]:
        """
        Détecte les breakouts des bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            lookback: Nombre de périodes à vérifier pour le breakout
            
        Returns:
            Tuple (upper_breakout, lower_breakout) indiquant si un breakout a été détecté
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback + 1:
            return False, False
        
        # Vérifier un breakout de la bande supérieure
        upper_breakout = False
        for i in range(1, lookback + 1):
            if df[column].iloc[-i] > df['upper_band'].iloc[-i]:
                upper_breakout = True
                break
        
        # Vérifier un breakout de la bande inférieure
        lower_breakout = False
        for i in range(1, lookback + 1):
            if df[column].iloc[-i] < df['lower_band'].iloc[-i]:
                lower_breakout = True
                break
        
        return upper_breakout, lower_breakout
    
    def detect_bollinger_bounce(self, data: Union[pd.DataFrame, MarketData], 
                              column: str = 'close',
                              lookback: int = 5,
                              bounce_threshold: float = 0.05) -> Tuple[bool, bool]:
        """
        Détecte les rebonds sur les bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            lookback: Nombre de périodes à vérifier pour le rebond
            bounce_threshold: Seuil pour considérer un mouvement comme un rebond
            
        Returns:
            Tuple (upper_bounce, lower_bounce) indiquant si un rebond a été détecté
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback + 1:
            return False, False
        
        # Vérifier un rebond sur la bande supérieure
        upper_bounce = False
        for i in range(2, lookback + 1):
            # Prix a touché la bande supérieure puis est redescendu
            if (df[column].iloc[-i] >= df['upper_band'].iloc[-i] * 0.99 and 
                df[column].iloc[-1] < df[column].iloc[-i] * (1 - bounce_threshold)):
                upper_bounce = True
                break
        
        # Vérifier un rebond sur la bande inférieure
        lower_bounce = False
        for i in range(2, lookback + 1):
            # Prix a touché la bande inférieure puis est remonté
            if (df[column].iloc[-i] <= df['lower_band'].iloc[-i] * 1.01 and 
                df[column].iloc[-1] > df[column].iloc[-i] * (1 + bounce_threshold)):
                lower_bounce = True
                break
        
        return upper_bounce, lower_bounce
    
    def detect_bollinger_squeeze(self, data: Union[pd.DataFrame, MarketData], 
                               lookback_period: int = 50,
                               percentile_threshold: int = 10) -> bool:
        """
        Détecte une compression des bandes de Bollinger (squeeze).
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            lookback_period: Période historique pour comparer la largeur des bandes
            percentile_threshold: Percentile pour considérer une compression
            
        Returns:
            True si une compression est détectée, False sinon
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback_period:
            lookback_period = len(df)
            logger.warning(f"Période de lookback ajustée à {lookback_period} en raison du manque de données")
        
        # Obtenir la largeur actuelle des bandes
        current_bandwidth = df['bandwidth'].iloc[-1]
        
        # Calculer le percentile de la largeur actuelle par rapport à l'historique
        historical_bandwidth = df['bandwidth'].iloc[-lookback_period:-1]
        percentile = np.percentile(historical_bandwidth, percentile_threshold)
        
        # Déterminer s'il y a une compression
        return current_bandwidth < percentile
    
    def detect_bollinger_trend(self, data: Union[pd.DataFrame, MarketData], 
                             column: str = 'close',
                             lookback: int = 20) -> str:
        """
        Détecte la tendance basée sur les bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            lookback: Nombre de périodes à vérifier pour la tendance
            
        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Vérifier s'il y a assez de données
        if len(df) < lookback + 1:
            return "neutral"
        
        # Vérifier la tendance de la moyenne mobile
        start_middle = df['middle_band'].iloc[-lookback]
        end_middle = df['middle_band'].iloc[-1]
        
        # Calculer le pourcentage de temps passé au-dessus de la moyenne mobile
        above_middle_count = sum(df[column].iloc[-lookback:] > df['middle_band'].iloc[-lookback:])
        above_middle_pct = above_middle_count / lookback
        
        # Déterminer la tendance
        if end_middle > start_middle * 1.01 and above_middle_pct > 0.6:
            return "bullish"
        elif end_middle < start_middle * 0.99 and above_middle_pct < 0.4:
            return "bearish"
        else:
            return "neutral"
    
    def calculate_bollinger_signals(self, data: Union[pd.DataFrame, MarketData], 
                                  column: str = 'close') -> pd.DataFrame:
        """
        Calcule les signaux d'achat et de vente basés sur les bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            
        Returns:
            DataFrame avec les colonnes de signaux ajoutées
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Initialiser les colonnes de signaux
        df['bb_signal'] = 0  # 1 pour achat, -1 pour vente, 0 pour neutre
        
        # Signaux basés sur le %B
        df.loc[df['percent_b'] < 0.05, 'bb_signal'] = 1  # Signal d'achat fort (prix très proche de la bande inférieure)
        df.loc[df['percent_b'] > 0.95, 'bb_signal'] = -1  # Signal de vente fort (prix très proche de la bande supérieure)
        
        # Signaux basés sur les croisements
        for i in range(1, len(df)):
            # Croisement de la bande inférieure vers le haut (signal d'achat)
            if df[column].iloc[i-1] <= df['lower_band'].iloc[i-1] and df[column].iloc[i] > df['lower_band'].iloc[i]:
                df.loc[df.index[i], 'bb_signal'] = 1
            
            # Croisement de la bande supérieure vers le bas (signal de vente)
            if df[column].iloc[i-1] >= df['upper_band'].iloc[i-1] and df[column].iloc[i] < df['upper_band'].iloc[i]:
                df.loc[df.index[i], 'bb_signal'] = -1
        
        return df
    
    def calculate_bollinger_strength(self, data: Union[pd.DataFrame, MarketData], 
                                   column: str = 'close') -> float:
        """
        Calcule la force du signal des bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            
        Returns:
            Valeur entre -100 et 100 indiquant la force du signal
        """
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Obtenir le dernier %B
        last_percent_b = df['percent_b'].iloc[-1]
        
        # Calculer la force du signal
        if last_percent_b <= 0.5:
            # Signal d'achat (0 à 100)
            strength = (0.5 - last_percent_b) * 200
        else:
            # Signal de vente (0 à -100)
            strength = (0.5 - last_percent_b) * 200
        
        return strength
    
    def analyze(self, data: Union[pd.DataFrame, MarketData], 
               column: str = 'close') -> Dict:
        """
        Analyse complète des bandes de Bollinger.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            column: Colonne à utiliser pour le calcul (par défaut 'close')
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse
        """
        results = {}
        
        # Calculer les bandes de Bollinger
        df = self.calculate_bollinger_bands(data, column)
        
        # Obtenir les dernières valeurs
        results['middle_band'] = df['middle_band'].iloc[-1]
        results['upper_band'] = df['upper_band'].iloc[-1]
        results['lower_band'] = df['lower_band'].iloc[-1]
        results['bandwidth'] = df['bandwidth'].iloc[-1]
        results['percent_b'] = df['percent_b'].iloc[-1]
        
        # Déterminer la condition du marché
        results['market_condition'] = self.identify_market_condition(data, column)
        
        # Détecter les breakouts
        upper_breakout, lower_breakout = self.detect_bollinger_breakout(data, column)
        results['upper_breakout'] = upper_breakout
        results['lower_breakout'] = lower_breakout
        
        # Détecter les rebonds
        upper_bounce, lower_bounce = self.detect_bollinger_bounce(data, column)
        results['upper_bounce'] = upper_bounce
        results['lower_bounce'] = lower_bounce
        
        # Détecter une compression
        results['squeeze'] = self.detect_bollinger_squeeze(data)
        
        # Déterminer la tendance
        results['trend'] = self.detect_bollinger_trend(data, column)
        
        # Calculer la force du signal
        results['signal_strength'] = self.calculate_bollinger_strength(data, column)
        
        return results
