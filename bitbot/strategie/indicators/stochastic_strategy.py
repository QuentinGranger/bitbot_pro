"""
Module pour intégrer l'oscillateur stochastique dans les stratégies de trading.

Ce module fournit des classes et fonctions pour utiliser l'oscillateur stochastique
dans les stratégies de trading, en générant des signaux d'achat et de vente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.StochasticOscillator import StochasticOscillatorIndicator, StochasticSignal
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.models.trade_signal import TradeSignal, SignalType
from bitbot.utils.logger import logger

class StochasticStrategy(StrategyBase):
    """
    Stratégie de trading basée sur l'oscillateur stochastique.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3, 
                overbought: int = 80, oversold: int = 20, 
                use_stoch_rsi: bool = False, use_divergence: bool = True):
        """
        Initialise la stratégie basée sur l'oscillateur stochastique.
        
        Args:
            k_period: Période pour le calcul de %K (par défaut 14)
            d_period: Période pour le calcul de %D (par défaut 3)
            slowing: Période de ralentissement (par défaut 3)
            overbought: Niveau de surachat (par défaut 80)
            oversold: Niveau de survente (par défaut 20)
            use_stoch_rsi: Si True, utilise le Stochastic RSI au lieu du Stochastic standard
            use_divergence: Si True, prend en compte les divergences dans la génération de signaux
        """
        super().__init__()
        
        self.name = "StochasticStrategy"
        self.description = "Stratégie basée sur l'oscillateur stochastique"
        
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.overbought = overbought
        self.oversold = oversold
        self.use_stoch_rsi = use_stoch_rsi
        self.use_divergence = use_divergence
        
        # Initialiser l'indicateur
        self.stoch_indicator = StochasticOscillatorIndicator(
            k_period=k_period,
            d_period=d_period,
            slowing=slowing,
            overbought=overbought,
            oversold=oversold
        )
        
        logger.info(f"Stratégie {self.name} initialisée avec les paramètres: "
                   f"k_period={k_period}, d_period={d_period}, slowing={slowing}, "
                   f"overbought={overbought}, oversold={oversold}, "
                   f"use_stoch_rsi={use_stoch_rsi}, use_divergence={use_divergence}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Mettre à jour l'indicateur
        self.stoch_indicator.set_parameters(
            k_period=self.k_period,
            d_period=self.d_period,
            slowing=self.slowing,
            overbought=self.overbought,
            oversold=self.oversold
        )
        
        logger.info(f"Paramètres de la stratégie {self.name} mis à jour: {kwargs}")
    
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur l'oscillateur stochastique.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            
        Returns:
            Liste de signaux de trading
        """
        signals = []
        
        # Analyser les données avec l'oscillateur stochastique
        analysis = self.stoch_indicator.analyze(data)
        
        # Extraire les données
        signal = analysis['signal']
        is_overbought = analysis['is_overbought']
        is_oversold = analysis['is_oversold']
        bullish_divergence = analysis['bullish_divergence']
        bearish_divergence = analysis['bearish_divergence']
        
        # Obtenir la dernière date/heure
        if isinstance(data, MarketData):
            timestamp = data.ohlcv.index[-1]
            symbol = data.symbol
            timeframe = data.timeframe
        else:
            timestamp = data.index[-1]
            symbol = "UNKNOWN"
            timeframe = "UNKNOWN"
        
        # Générer des signaux basés sur les conditions
        
        # 1. Signal d'achat fort: survente + divergence haussière ou signal d'achat fort
        if (is_oversold and (bullish_divergence if self.use_divergence else True)) or signal == StochasticSignal.STRONG_BUY:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.STRONG_BUY,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.8,
                source=self.name,
                metadata={
                    "stochastic_k": analysis['%K'],
                    "stochastic_d": analysis['%D'],
                    "is_oversold": is_oversold,
                    "bullish_divergence": bullish_divergence,
                    "signal": signal.value
                }
            ))
        
        # 2. Signal d'achat: signal d'achat standard sans conditions supplémentaires
        elif signal == StochasticSignal.BUY:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.6,
                source=self.name,
                metadata={
                    "stochastic_k": analysis['%K'],
                    "stochastic_d": analysis['%D'],
                    "signal": signal.value
                }
            ))
        
        # 3. Signal de vente fort: surachat + divergence baissière ou signal de vente fort
        elif (is_overbought and (bearish_divergence if self.use_divergence else True)) or signal == StochasticSignal.STRONG_SELL:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.STRONG_SELL,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.8,
                source=self.name,
                metadata={
                    "stochastic_k": analysis['%K'],
                    "stochastic_d": analysis['%D'],
                    "is_overbought": is_overbought,
                    "bearish_divergence": bearish_divergence,
                    "signal": signal.value
                }
            ))
        
        # 4. Signal de vente: signal de vente standard sans conditions supplémentaires
        elif signal == StochasticSignal.SELL:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.6,
                source=self.name,
                metadata={
                    "stochastic_k": analysis['%K'],
                    "stochastic_d": analysis['%D'],
                    "signal": signal.value
                }
            ))
        
        return signals
    
    def backtest(self, data: Union[pd.DataFrame, MarketData], initial_capital: float = 10000.0) -> Dict:
        """
        Effectue un backtest simple de la stratégie sur les données historiques.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLC
            initial_capital: Capital initial pour le backtest
            
        Returns:
            Dictionnaire contenant les résultats du backtest
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Calculer l'oscillateur stochastique
        if self.use_stoch_rsi:
            df_with_indicators = self.stoch_indicator.calculate_stochastic_rsi(data)
            k_col = 'StochRSI_K'
            d_col = 'StochRSI_D'
        else:
            df_with_indicators = self.stoch_indicator.calculate_stochastic(data)
            k_col = '%K'
            d_col = '%D'
        
        # Initialiser les colonnes pour le backtest
        df_with_indicators['signal'] = 0  # 0: pas de signal, 1: achat, -1: vente
        df_with_indicators['position'] = 0  # 0: pas de position, 1: position longue
        df_with_indicators['capital'] = initial_capital
        df_with_indicators['holdings'] = 0.0
        df_with_indicators['total_value'] = initial_capital
        
        # Générer les signaux pour chaque période
        for i in range(self.k_period + self.d_period, len(df_with_indicators)):
            # Extraire les données jusqu'à l'index i
            subset_data = df_with_indicators.iloc[:i+1]
            
            # Vérifier les conditions pour les signaux
            k_value = subset_data[k_col].iloc[-1]
            d_value = subset_data[d_col].iloc[-1]
            prev_k = subset_data[k_col].iloc[-2]
            prev_d = subset_data[d_col].iloc[-2]
            
            # Signal d'achat: %K croise %D vers le haut en zone de survente
            if (prev_k < prev_d and k_value > d_value and 
                k_value < self.oversold and d_value < self.oversold):
                df_with_indicators.at[subset_data.index[-1], 'signal'] = 1
            
            # Signal de vente: %K croise %D vers le bas en zone de surachat
            elif (prev_k > prev_d and k_value < d_value and 
                  k_value > self.overbought and d_value > self.overbought):
                df_with_indicators.at[subset_data.index[-1], 'signal'] = -1
        
        # Simuler le trading
        position = 0
        capital = initial_capital
        holdings = 0.0
        
        for i in range(len(df_with_indicators)):
            price = df_with_indicators['close'].iloc[i]
            signal = df_with_indicators['signal'].iloc[i]
            
            # Mettre à jour la position
            if signal == 1 and position == 0:  # Signal d'achat et pas de position
                holdings = capital / price
                capital = 0
                position = 1
            elif signal == -1 and position == 1:  # Signal de vente et position longue
                capital = holdings * price
                holdings = 0
                position = 0
            
            # Mettre à jour les valeurs
            df_with_indicators.at[df_with_indicators.index[i], 'position'] = position
            df_with_indicators.at[df_with_indicators.index[i], 'capital'] = capital
            df_with_indicators.at[df_with_indicators.index[i], 'holdings'] = holdings
            df_with_indicators.at[df_with_indicators.index[i], 'total_value'] = capital + (holdings * price)
        
        # Calculer les métriques de performance
        initial_value = initial_capital
        final_value = df_with_indicators['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculer le nombre de trades
        buy_signals = df_with_indicators[df_with_indicators['signal'] == 1]
        sell_signals = df_with_indicators[df_with_indicators['signal'] == -1]
        num_trades = min(len(buy_signals), len(sell_signals))
        
        # Calculer le ratio de Sharpe (simplifié)
        daily_returns = df_with_indicators['total_value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Calculer le drawdown maximum
        df_with_indicators['drawdown'] = 1 - df_with_indicators['total_value'] / df_with_indicators['total_value'].cummax()
        max_drawdown = df_with_indicators['drawdown'].max() * 100
        
        # Résultats du backtest
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_value,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'backtest_data': df_with_indicators
        }
        
        return results
