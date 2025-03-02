"""
Module implémentant la stratégie de divergence MACD avec filtres de volatilité.

Ce module fournit une implémentation complète d'une stratégie basée sur les divergences MACD
et inclut des filtres de volatilité pour réduire les faux signaux dans les marchés en range.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum
from datetime import datetime

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.strategie.base.MACD import MACDIndicator, MACDSignalType
from bitbot.strategie.base.ATR import ATRIndicator, VolatilityLevel
from bitbot.utils.logger import logger

class DivergenceType(Enum):
    """Types de divergences entre le prix et le MACD."""
    REGULAR_BULLISH = "Divergence haussière régulière"
    REGULAR_BEARISH = "Divergence baissière régulière"
    HIDDEN_BULLISH = "Divergence haussière cachée"
    HIDDEN_BEARISH = "Divergence baissière cachée"
    NONE = "Pas de divergence"

class MACDDivergenceStrategy(StrategyBase):
    """
    Stratégie basée sur les divergences MACD avec filtres de volatilité.
    
    Cette stratégie identifie les retournements de tendance potentiels en détectant
    les divergences entre le prix et l'indicateur MACD, tout en appliquant des
    filtres de volatilité pour éviter les faux signaux pendant les phases de range.
    """
    
    def __init__(self, 
                macd_fast_period: int = 12,
                macd_slow_period: int = 26,
                macd_signal_period: int = 9,
                lookback_period: int = 30,
                divergence_threshold: float = 0.05,
                use_volatility_filter: bool = True,
                atr_period: int = 14,
                atr_threshold_pct: float = 0.5,
                confirmation_periods: int = 2):
        """
        Initialise la stratégie de divergence MACD avec filtres de volatilité.
        
        Args:
            macd_fast_period: Période pour l'EMA rapide du MACD (défaut: 12)
            macd_slow_period: Période pour l'EMA lente du MACD (défaut: 26)
            macd_signal_period: Période pour la ligne de signal du MACD (défaut: 9)
            lookback_period: Période de recherche pour les divergences (défaut: 30)
            divergence_threshold: Seuil minimal pour considérer une divergence significative (défaut: 0.05, soit 5%)
            use_volatility_filter: Si True, applique le filtre de volatilité pour éviter les faux signaux
            atr_period: Période pour le calcul de l'ATR (défaut: 14)
            atr_threshold_pct: Seuil ATR en pourcentage pour filtrer les signaux (défaut: 0.5%)
            confirmation_periods: Nombre de périodes de confirmation avant de valider un signal (défaut: 2)
        """
        super().__init__()
        
        self.name = "MACDDivergenceStrategy"
        self.description = "Stratégie basée sur les divergences MACD avec filtres de volatilité"
        
        # Paramètres MACD
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        
        # Paramètres de divergence
        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold
        self.confirmation_periods = confirmation_periods
        
        # Paramètres de filtre de volatilité
        self.use_volatility_filter = use_volatility_filter
        self.atr_period = atr_period
        self.atr_threshold_pct = atr_threshold_pct
        
        # Initialiser les indicateurs
        self.macd_indicator = MACDIndicator(
            fast_period=macd_fast_period,
            slow_period=macd_slow_period,
            signal_period=macd_signal_period
        )
        
        self.atr_indicator = ATRIndicator(period=atr_period)
        
        logger.info(f"Stratégie de divergence MACD initialisée: "
                  f"MACD({macd_fast_period},{macd_slow_period},{macd_signal_period}), "
                  f"Filtre de volatilité: {'Activé' if use_volatility_filter else 'Désactivé'}")
    
    def set_parameters(self, **kwargs):
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir
        """
        for param, value in kwargs.items():
            if param == 'macd_fast_period' and isinstance(value, int) and value > 0:
                self.macd_fast_period = value
                self.macd_indicator.set_parameters(fast_period=value)
            elif param == 'macd_slow_period' and isinstance(value, int) and value > 0:
                self.macd_slow_period = value
                self.macd_indicator.set_parameters(slow_period=value)
            elif param == 'macd_signal_period' and isinstance(value, int) and value > 0:
                self.macd_signal_period = value
                self.macd_indicator.set_parameters(signal_period=value)
            elif param == 'lookback_period' and isinstance(value, int) and value > 0:
                self.lookback_period = value
            elif param == 'divergence_threshold' and isinstance(value, (int, float)) and value > 0:
                self.divergence_threshold = value
            elif param == 'use_volatility_filter' and isinstance(value, bool):
                self.use_volatility_filter = value
            elif param == 'atr_period' and isinstance(value, int) and value > 0:
                self.atr_period = value
                self.atr_indicator.set_period(value)
            elif param == 'atr_threshold_pct' and isinstance(value, (int, float)) and value >= 0:
                self.atr_threshold_pct = value
            elif param == 'confirmation_periods' and isinstance(value, int) and value >= 0:
                self.confirmation_periods = value
        
        logger.info(f"Paramètres mis à jour: {kwargs}")
    
    def _find_peaks_and_troughs(self, data: pd.Series, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """
        Trouve les sommets (peaks) et les creux (troughs) dans une série temporelle.
        
        Args:
            data: Série de données à analyser
            min_distance: Distance minimale entre deux sommets ou deux creux
            
        Returns:
            Tuple contenant (indices des sommets, indices des creux)
        """
        # Identifier les candidats pour les sommets et les creux
        peak_candidates = []
        trough_candidates = []
        
        for i in range(1, len(data) - 1):
            # Détection des sommets (peaks)
            if data.iloc[i] > data.iloc[i-1] and data.iloc[i] > data.iloc[i+1]:
                peak_candidates.append(i)
            # Détection des creux (troughs)
            elif data.iloc[i] < data.iloc[i-1] and data.iloc[i] < data.iloc[i+1]:
                trough_candidates.append(i)
        
        # Filtrer les sommets et les creux pour respecter la distance minimale
        peaks = []
        if peak_candidates:
            peaks.append(peak_candidates[0])
            for peak in peak_candidates[1:]:
                if peak - peaks[-1] >= min_distance:
                    peaks.append(peak)
        
        troughs = []
        if trough_candidates:
            troughs.append(trough_candidates[0])
            for trough in trough_candidates[1:]:
                if trough - troughs[-1] >= min_distance:
                    troughs.append(trough)
        
        return peaks, troughs
    
    def _calculate_slope(self, y1: float, y2: float) -> float:
        """
        Calcule la pente entre deux points.
        
        Args:
            y1: Première valeur
            y2: Deuxième valeur
            
        Returns:
            Pente (positive si ascendante, négative si descendante)
        """
        return y2 - y1
    
    def detect_divergence(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Détecte les divergences entre le prix et le MACD.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Dictionnaire avec les informations sur les divergences détectées
        """
        # Calculer le MACD
        df = self.macd_indicator.calculate_macd(data)
        
        # Vérifier s'il y a assez de données
        if len(df) < self.lookback_period:
            return {'type': DivergenceType.NONE, 'strength': 0.0, 'message': "Données insuffisantes"}
        
        # Extraire la période de recherche
        df_period = df.iloc[-self.lookback_period:].copy()
        
        # Trouver les sommets et les creux du prix et du MACD
        price_peaks, price_troughs = self._find_peaks_and_troughs(df_period['close'])
        macd_peaks, macd_troughs = self._find_peaks_and_troughs(df_period['macd'])
        
        if not (price_peaks and price_troughs and macd_peaks and macd_troughs):
            return {'type': DivergenceType.NONE, 'strength': 0.0, 'message': "Pas assez de sommets/creux détectés"}
        
        # Divergence haussière régulière:
        # Le prix fait des creux plus bas, mais le MACD fait des creux plus hauts
        bullish_regular = False
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            price_slope = self._calculate_slope(df_period['close'].iloc[price_troughs[-2]], 
                                              df_period['close'].iloc[price_troughs[-1]])
            macd_slope = self._calculate_slope(df_period['macd'].iloc[macd_troughs[-2]], 
                                             df_period['macd'].iloc[macd_troughs[-1]])
            
            if price_slope < 0 and macd_slope > 0:
                bullish_regular = True
                divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                
                if divergence_strength >= self.divergence_threshold:
                    return {
                        'type': DivergenceType.REGULAR_BULLISH,
                        'strength': divergence_strength,
                        'price_troughs': [price_troughs[-2], price_troughs[-1]],
                        'macd_troughs': [macd_troughs[-2], macd_troughs[-1]],
                        'message': "Divergence haussière régulière détectée"
                    }
        
        # Divergence baissière régulière:
        # Le prix fait des sommets plus hauts, mais le MACD fait des sommets plus bas
        bearish_regular = False
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            price_slope = self._calculate_slope(df_period['close'].iloc[price_peaks[-2]], 
                                              df_period['close'].iloc[price_peaks[-1]])
            macd_slope = self._calculate_slope(df_period['macd'].iloc[macd_peaks[-2]], 
                                             df_period['macd'].iloc[macd_peaks[-1]])
            
            if price_slope > 0 and macd_slope < 0:
                bearish_regular = True
                divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                
                if divergence_strength >= self.divergence_threshold:
                    return {
                        'type': DivergenceType.REGULAR_BEARISH,
                        'strength': divergence_strength,
                        'price_peaks': [price_peaks[-2], price_peaks[-1]],
                        'macd_peaks': [macd_peaks[-2], macd_peaks[-1]],
                        'message': "Divergence baissière régulière détectée"
                    }
        
        # Divergence haussière cachée:
        # Le prix fait des creux plus hauts, mais le MACD fait des creux plus bas
        bullish_hidden = False
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            price_slope = self._calculate_slope(df_period['close'].iloc[price_troughs[-2]], 
                                              df_period['close'].iloc[price_troughs[-1]])
            macd_slope = self._calculate_slope(df_period['macd'].iloc[macd_troughs[-2]], 
                                             df_period['macd'].iloc[macd_troughs[-1]])
            
            if price_slope > 0 and macd_slope < 0:
                bullish_hidden = True
                divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                
                if divergence_strength >= self.divergence_threshold:
                    return {
                        'type': DivergenceType.HIDDEN_BULLISH,
                        'strength': divergence_strength,
                        'price_troughs': [price_troughs[-2], price_troughs[-1]],
                        'macd_troughs': [macd_troughs[-2], macd_troughs[-1]],
                        'message': "Divergence haussière cachée détectée"
                    }
        
        # Divergence baissière cachée:
        # Le prix fait des sommets plus bas, mais le MACD fait des sommets plus hauts
        bearish_hidden = False
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            price_slope = self._calculate_slope(df_period['close'].iloc[price_peaks[-2]], 
                                              df_period['close'].iloc[price_peaks[-1]])
            macd_slope = self._calculate_slope(df_period['macd'].iloc[macd_peaks[-2]], 
                                             df_period['macd'].iloc[macd_peaks[-1]])
            
            if price_slope < 0 and macd_slope > 0:
                bearish_hidden = True
                divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                
                if divergence_strength >= self.divergence_threshold:
                    return {
                        'type': DivergenceType.HIDDEN_BEARISH,
                        'strength': divergence_strength,
                        'price_peaks': [price_peaks[-2], price_peaks[-1]],
                        'macd_peaks': [macd_peaks[-2], macd_peaks[-1]],
                        'message': "Divergence baissière cachée détectée"
                    }
        
        # Aucune divergence significative détectée
        return {'type': DivergenceType.NONE, 'strength': 0.0, 'message': "Pas de divergence significative"}
    
    def check_volatility(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Vérifie le niveau de volatilité actuel en utilisant l'ATR.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Dictionnaire avec les informations sur la volatilité
        """
        if not self.use_volatility_filter:
            return {'volatility_level': VolatilityLevel.MODERATE, 'atr_pct': 0.0, 'is_valid': True}
        
        # Calculer l'ATR
        df = self.atr_indicator.calculate_atr(data)
        
        if len(df) == 0:
            return {'volatility_level': VolatilityLevel.VERY_LOW, 'atr_pct': 0.0, 'is_valid': False}
        
        # Calculer l'ATR en pourcentage du prix
        latest_price = df['close'].iloc[-1]
        latest_atr = df['atr'].iloc[-1]
        atr_pct = (latest_atr / latest_price) * 100
        
        # Déterminer le niveau de volatilité
        if atr_pct < self.atr_threshold_pct:
            volatility_level = VolatilityLevel.LOW
            is_valid = False
            message = f"Volatilité trop faible (ATR: {atr_pct:.2f}% < seuil: {self.atr_threshold_pct:.2f}%)"
        else:
            volatility_level = VolatilityLevel.MODERATE
            is_valid = True
            message = f"Volatilité suffisante (ATR: {atr_pct:.2f}% >= seuil: {self.atr_threshold_pct:.2f}%)"
        
        return {
            'volatility_level': volatility_level,
            'atr_pct': atr_pct,
            'is_valid': is_valid,
            'message': message
        }
    
    def calculate_signal(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Calcule le signal de trading basé sur les divergences MACD et le filtre de volatilité.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Dictionnaire avec le signal de trading et les informations associées
        """
        # Vérifier la volatilité
        volatility_check = self.check_volatility(data)
        
        # Détecter les divergences
        divergence = self.detect_divergence(data)
        
        # Initialiser le résultat avec les infos de divergence et de volatilité
        result = {
            'signal': MACDSignalType.NEUTRAL,
            'divergence': divergence,
            'volatility': volatility_check,
            'message': "Pas de signal",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Si la volatilité est trop faible et que le filtre est activé, pas de signal
        if self.use_volatility_filter and not volatility_check['is_valid']:
            result['message'] = f"Signal ignoré: {volatility_check['message']}"
            return result
        
        # Traiter les divergences pour générer des signaux
        if divergence['type'] in [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]:
            result['signal'] = MACDSignalType.BUY
            result['message'] = f"Signal d'achat: {divergence['message']}"
        elif divergence['type'] in [DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH]:
            result['signal'] = MACDSignalType.SELL
            result['message'] = f"Signal de vente: {divergence['message']}"
        
        return result
    
    def calculate_signals_historical(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule les signaux historiques sur un jeu de données complet.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix historiques
            
        Returns:
            DataFrame avec les colonnes de prix, MACD et signaux
        """
        # Calculer le MACD et l'ATR
        df_macd = self.macd_indicator.calculate_macd(data)
        df_atr = self.atr_indicator.calculate_atr(data)
        
        # Fusionner les DataFrames
        df = pd.merge(df_macd, df_atr[['atr']], left_index=True, right_index=True)
        
        # Initialiser les colonnes de signaux
        df['signal'] = MACDSignalType.NEUTRAL
        df['divergence_type'] = DivergenceType.NONE
        df['divergence_strength'] = 0.0
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        df['is_volatile_enough'] = df['atr_pct'] >= self.atr_threshold_pct
        
        # Minimum de périodes nécessaires pour calculer les signaux
        min_periods = max(self.macd_slow_period, self.atr_period) + self.lookback_period
        
        if len(df) < min_periods:
            logger.warning(f"Données insuffisantes pour calculer les signaux historiques. Nécessite {min_periods} périodes, reçu {len(df)}")
            return df
        
        # Calculer les signaux pour chaque période, à partir du moment où nous avons assez de données
        for i in range(min_periods, len(df)):
            # Extraire les données jusqu'à ce point
            data_slice = df.iloc[:i+1].copy()
            
            # Détection des divergences
            data_window = data_slice.iloc[-self.lookback_period:].copy()
            price_peaks, price_troughs = self._find_peaks_and_troughs(data_window['close'])
            macd_peaks, macd_troughs = self._find_peaks_and_troughs(data_window['macd'])
            
            # Vérifier les divergences seulement si nous avons des sommets et des creux
            if len(price_peaks) >= 2 and len(price_troughs) >= 2 and len(macd_peaks) >= 2 and len(macd_troughs) >= 2:
                # Divergence haussière régulière
                if (price_troughs[-2] and price_troughs[-1] and macd_troughs[-2] and macd_troughs[-1]):
                    price_slope = self._calculate_slope(data_window['close'].iloc[price_troughs[-2]], 
                                                       data_window['close'].iloc[price_troughs[-1]])
                    macd_slope = self._calculate_slope(data_window['macd'].iloc[macd_troughs[-2]], 
                                                      data_window['macd'].iloc[macd_troughs[-1]])
                    
                    if price_slope < 0 and macd_slope > 0:
                        divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                        
                        if divergence_strength >= self.divergence_threshold:
                            df.loc[df.index[i], 'divergence_type'] = DivergenceType.REGULAR_BULLISH.value
                            df.loc[df.index[i], 'divergence_strength'] = divergence_strength
                            
                            # Générer un signal d'achat si la volatilité est suffisante
                            if not self.use_volatility_filter or df.loc[df.index[i], 'is_volatile_enough']:
                                df.loc[df.index[i], 'signal'] = MACDSignalType.BUY
                
                # Divergence baissière régulière
                if (price_peaks[-2] and price_peaks[-1] and macd_peaks[-2] and macd_peaks[-1]):
                    price_slope = self._calculate_slope(data_window['close'].iloc[price_peaks[-2]], 
                                                       data_window['close'].iloc[price_peaks[-1]])
                    macd_slope = self._calculate_slope(data_window['macd'].iloc[macd_peaks[-2]], 
                                                      data_window['macd'].iloc[macd_peaks[-1]])
                    
                    if price_slope > 0 and macd_slope < 0:
                        divergence_strength = abs(macd_slope / price_slope) if price_slope != 0 else 0
                        
                        if divergence_strength >= self.divergence_threshold:
                            df.loc[df.index[i], 'divergence_type'] = DivergenceType.REGULAR_BEARISH.value
                            df.loc[df.index[i], 'divergence_strength'] = divergence_strength
                            
                            # Générer un signal de vente si la volatilité est suffisante
                            if not self.use_volatility_filter or df.loc[df.index[i], 'is_volatile_enough']:
                                df.loc[df.index[i], 'signal'] = MACDSignalType.SELL
        
        return df
    
    def run(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Exécute la stratégie sur les données et retourne le signal.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            Dictionnaire avec le signal et autres infos
        """
        return self.calculate_signal(data)
    
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Génère des signaux pour le jeu de données fourni.
        
        Cette méthode est requise par l'interface StrategyBase.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix
            
        Returns:
            DataFrame avec les colonnes originales plus les signaux générés
        """
        return self.calculate_signals_historical(data)
