"""
Module pour intégrer le RSI dans les stratégies de trading.

Ce module fournit des classes et fonctions pour utiliser le Relative Strength Index (RSI)
dans les stratégies de trading, en générant des signaux d'achat et de vente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum
from datetime import datetime

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.RSI import RSIIndicator, RSISignalType, TrendType
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.models.trade_signal import TradeSignal, SignalType
from bitbot.utils.logger import logger


class RSIStrategy(StrategyBase):
    """
    Stratégie de trading basée sur le RSI avec ajustement dynamique des seuils
    et importance réduite pendant les périodes plates.
    """
    
    def __init__(self, period: int = 14, 
                overbought_threshold: int = 70, 
                oversold_threshold: int = 30,
                strong_overbought_threshold: int = 80,
                strong_oversold_threshold: int = 20,
                use_dynamic_thresholds: bool = True,
                trend_weight: float = 1.0,
                range_weight: float = 0.5,
                lookback_period: int = 50):
        """
        Initialise la stratégie basée sur le RSI.
        
        Args:
            period: Période pour le calcul du RSI (par défaut 14)
            overbought_threshold: Niveau de surachat standard (par défaut 70)
            oversold_threshold: Niveau de survente standard (par défaut 30)
            strong_overbought_threshold: Niveau de surachat fort (par défaut 80)
            strong_oversold_threshold: Niveau de survente fort (par défaut 20)
            use_dynamic_thresholds: Si True, ajuste dynamiquement les seuils selon la tendance (par défaut True)
            trend_weight: Poids du RSI dans le score composite pendant les périodes de tendance (par défaut 1.0)
            range_weight: Poids du RSI dans le score composite pendant les périodes plates (par défaut 0.5)
            lookback_period: Période d'analyse pour la détection de tendance (par défaut 50)
        """
        super().__init__()
        
        self.name = "RSIStrategy"
        self.description = "Stratégie basée sur le RSI avec ajustement dynamique des seuils"
        
        self.period = period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.strong_overbought_threshold = strong_overbought_threshold
        self.strong_oversold_threshold = strong_oversold_threshold
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.trend_weight = trend_weight
        self.range_weight = range_weight
        self.lookback_period = lookback_period
        
        # Créer l'indicateur RSI
        self.rsi_indicator = RSIIndicator(
            period=period,
            overbought_threshold=overbought_threshold,
            oversold_threshold=oversold_threshold,
            strong_overbought_threshold=strong_overbought_threshold,
            strong_oversold_threshold=strong_oversold_threshold
        )
        
        logger.info(f"Stratégie RSI initialisée: RSI({period}), "
                  f"Seuils: {oversold_threshold}/{overbought_threshold}, "
                  f"Seuils forts: {strong_oversold_threshold}/{strong_overbought_threshold}, "
                  f"Ajustement dynamique: {'Activé' if use_dynamic_thresholds else 'Désactivé'}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir
        """
        super().set_parameters(**kwargs)
        
        # Mettre à jour les paramètres de l'indicateur RSI
        rsi_params = {}
        if 'period' in kwargs:
            rsi_params['period'] = kwargs['period']
        if 'overbought_threshold' in kwargs:
            rsi_params['overbought_threshold'] = kwargs['overbought_threshold']
        if 'oversold_threshold' in kwargs:
            rsi_params['oversold_threshold'] = kwargs['oversold_threshold']
        if 'strong_overbought_threshold' in kwargs:
            rsi_params['strong_overbought_threshold'] = kwargs['strong_overbought_threshold']
        if 'strong_oversold_threshold' in kwargs:
            rsi_params['strong_oversold_threshold'] = kwargs['strong_oversold_threshold']
        
        if rsi_params:
            self.rsi_indicator.set_parameters(**rsi_params)
    
    def calculate_composite_score(self, 
                                 signal_info: Dict, 
                                 other_indicators: Optional[Dict] = None) -> float:
        """
        Calcule un score composite basé sur le RSI et d'autres indicateurs.
        Le poids du RSI est réduit pendant les marchés sans tendance.
        
        Args:
            signal_info: Dictionnaire avec les informations de signal RSI
            other_indicators: Dictionnaire avec les résultats d'autres indicateurs
            
        Returns:
            Score composite entre -1 (fort signal de vente) et 1 (fort signal d'achat)
        """
        # Poids par défaut du RSI
        rsi_weight = self.trend_weight
        
        # Réduire le poids si nous sommes dans un marché plat
        if signal_info['trend'] == TrendType.RANGE:
            rsi_weight = self.range_weight
            logger.debug(f"Marché sans tendance forte détecté, réduction du poids du RSI à {rsi_weight}")
        
        # Score RSI de base
        rsi_value = signal_info['current_rsi']
        rsi_score = 0.0
        
        # Normaliser le RSI en score entre -1 et 1
        if rsi_value <= 30:
            # Zone de survente: score positif (signal d'achat)
            rsi_score = (30 - rsi_value) / 30
        elif rsi_value >= 70:
            # Zone de surachat: score négatif (signal de vente)
            rsi_score = -1 * (rsi_value - 70) / 30
        else:
            # Zone neutre: score proportionnel
            rsi_score = (50 - rsi_value) / 40
        
        # Score final: RSI uniquement si pas d'autres indicateurs
        if not other_indicators:
            return rsi_score * rsi_weight
        
        # Si d'autres indicateurs sont fournis, les intégrer dans le score composite
        # (Cette partie est extensible selon les autres indicateurs à intégrer)
        total_weight = rsi_weight
        composite_score = rsi_score * rsi_weight
        
        # Exemple: intégrer un score MACD si présent
        if other_indicators and 'macd_score' in other_indicators:
            macd_weight = 1.0 - rsi_weight  # Poids complémentaire
            composite_score += other_indicators['macd_score'] * macd_weight
            total_weight += macd_weight
        
        # Normaliser le score final
        if total_weight > 0:
            composite_score /= total_weight
        
        return composite_score
    
    def generate_signal(self, data: Union[pd.DataFrame, MarketData], 
                      other_indicators: Optional[Dict] = None) -> Dict:
        """
        Génère un signal de trading basé sur le RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            other_indicators: Dictionnaire avec les résultats d'autres indicateurs
            
        Returns:
            Dictionnaire avec le signal de trading et les informations associées
        """
        # Analyser le RSI
        signal_info = self.rsi_indicator.get_signal(data, use_dynamic_thresholds=self.use_dynamic_thresholds)
        
        # Calculer le score composite
        composite_score = self.calculate_composite_score(signal_info, other_indicators)
        
        # Déterminer le signal final basé sur le score composite
        if composite_score > 0.8:
            trade_signal = SignalType.STRONG_BUY
        elif composite_score > 0.3:
            trade_signal = SignalType.BUY
        elif composite_score < -0.8:
            trade_signal = SignalType.STRONG_SELL
        elif composite_score < -0.3:
            trade_signal = SignalType.SELL
        else:
            trade_signal = SignalType.NEUTRAL
        
        # Construire le résultat
        result = {
            'signal': trade_signal,
            'rsi_signal': signal_info['signal'],
            'composite_score': composite_score,
            'current_rsi': signal_info['current_rsi'],
            'trend': signal_info['trend'],
            'message': f"{signal_info['message']}, Score composite: {composite_score:.2f}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur le RSI.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Liste des signaux de trading générés
        """
        # Générer le signal
        signal_info = self.generate_signal(data)
        
        # Créer un objet TradeSignal
        trade_signal = TradeSignal(
            timestamp=datetime.now(),
            symbol=data.symbol if isinstance(data, MarketData) else "UNKNOWN",
            signal_type=signal_info['signal'],
            source=self.name,
            message=signal_info['message'],
            confidence=abs(signal_info['composite_score']),
            metadata={
                'rsi': signal_info['current_rsi'],
                'trend': signal_info['trend'].value,
                'composite_score': signal_info['composite_score']
            }
        )
        
        return [trade_signal]
    
    def calculate_signals_historical(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule les signaux historiques sur un jeu de données complet.
        
        Args:
            data: DataFrame ou MarketData contenant les données de prix historiques
            
        Returns:
            DataFrame avec les colonnes de prix, RSI et signaux
        """
        # Calculer le RSI sur l'ensemble des données
        df_rsi = self.rsi_indicator.calculate_rsi(data)
        
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Fusionner avec les données de prix si nécessaire
        if 'close' not in df_rsi.columns:
            df_rsi = df_rsi.join(df[['close']], how='left')
        
        # Initialiser les colonnes de signaux
        df_rsi['signal'] = SignalType.NEUTRAL
        df_rsi['trend'] = TrendType.UNKNOWN.value
        df_rsi['composite_score'] = 0.0
        
        # Minimum de périodes nécessaires pour calculer les signaux
        min_periods = self.period + 10
        
        if len(df_rsi) < min_periods:
            logger.warning(f"Données insuffisantes pour calculer les signaux historiques. Nécessite {min_periods} périodes, reçu {len(df_rsi)}")
            return df_rsi
        
        # Calculer les signaux pour chaque période, à partir du moment où nous avons assez de données
        for i in range(min_periods, len(df_rsi)):
            # Extraire les données jusqu'à ce point
            data_slice = df_rsi.iloc[:i+1].copy()
            
            # Détecter la tendance
            trend = self.rsi_indicator.detect_trend(data_slice, lookback_period=min(self.lookback_period, i))
            df_rsi.loc[df_rsi.index[i], 'trend'] = trend.value
            
            # Obtenir les seuils RSI ajustés en fonction de la tendance
            if self.use_dynamic_thresholds:
                thresholds = self.rsi_indicator.get_dynamic_thresholds(data_slice)
                overbought_threshold = thresholds['overbought']
                oversold_threshold = thresholds['oversold']
            else:
                overbought_threshold = self.overbought_threshold
                oversold_threshold = self.oversold_threshold
            
            # Calculer le score composite
            current_rsi = df_rsi.loc[df_rsi.index[i], 'rsi']
            
            # Réduire le poids du RSI dans le score composite si nous sommes dans un marché plat
            rsi_weight = self.trend_weight if trend != TrendType.RANGE else self.range_weight
            
            # Normaliser le RSI en score
            if current_rsi <= 30:
                rsi_score = (30 - current_rsi) / 30
            elif current_rsi >= 70:
                rsi_score = -1 * (current_rsi - 70) / 30
            else:
                rsi_score = (50 - current_rsi) / 40
            
            # Stocker le score composite
            df_rsi.loc[df_rsi.index[i], 'composite_score'] = rsi_score * rsi_weight
            
            # Générer le signal en fonction du score
            composite_score = df_rsi.loc[df_rsi.index[i], 'composite_score']
            if composite_score > 0.8:
                df_rsi.loc[df_rsi.index[i], 'signal'] = SignalType.STRONG_BUY
            elif composite_score > 0.3:
                df_rsi.loc[df_rsi.index[i], 'signal'] = SignalType.BUY
            elif composite_score < -0.8:
                df_rsi.loc[df_rsi.index[i], 'signal'] = SignalType.STRONG_SELL
            elif composite_score < -0.3:
                df_rsi.loc[df_rsi.index[i], 'signal'] = SignalType.SELL
            else:
                df_rsi.loc[df_rsi.index[i], 'signal'] = SignalType.NEUTRAL
        
        return df_rsi
