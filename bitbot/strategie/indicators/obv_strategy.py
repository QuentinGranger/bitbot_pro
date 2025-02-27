"""
Module pour intégrer l'indicateur On-Balance Volume (OBV) dans les stratégies de trading.

Ce module fournit des classes et fonctions pour utiliser l'OBV
dans les stratégies de trading, en générant des signaux d'achat et de vente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.OBV import OBVIndicator, OBVSignal
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.models.trade_signal import TradeSignal, SignalType
from bitbot.utils.logger import logger

class OBVStrategy(StrategyBase):
    """
    Stratégie de trading basée sur l'indicateur On-Balance Volume (OBV).
    """
    
    def __init__(self, ema_period: int = 20, signal_period: int = 9, 
                use_divergence: bool = True, use_vpt: bool = False):
        """
        Initialise la stratégie basée sur l'OBV.
        
        Args:
            ema_period: Période pour le calcul de l'EMA de l'OBV (par défaut 20)
            signal_period: Période pour le calcul de la ligne de signal (par défaut 9)
            use_divergence: Si True, prend en compte les divergences dans la génération de signaux
            use_vpt: Si True, utilise le Volume Price Trend au lieu de l'OBV standard
        """
        super().__init__()
        
        self.name = "OBVStrategy"
        self.description = "Stratégie basée sur l'indicateur On-Balance Volume"
        
        self.ema_period = ema_period
        self.signal_period = signal_period
        self.use_divergence = use_divergence
        self.use_vpt = use_vpt
        
        # Initialiser l'indicateur
        self.obv_indicator = OBVIndicator(
            ema_period=ema_period,
            signal_period=signal_period
        )
        
        logger.info(f"Stratégie {self.name} initialisée avec les paramètres: "
                   f"ema_period={ema_period}, signal_period={signal_period}, "
                   f"use_divergence={use_divergence}, use_vpt={use_vpt}")
    
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
        self.obv_indicator.set_parameters(
            ema_period=self.ema_period,
            signal_period=self.signal_period
        )
        
        logger.info(f"Paramètres de la stratégie {self.name} mis à jour: {kwargs}")
    
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur l'indicateur OBV.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            Liste de signaux de trading
        """
        signals = []
        
        # Analyser les données avec l'OBV
        analysis = self.obv_indicator.analyze(data)
        
        # Extraire les données
        signal = analysis['signal']
        is_increasing = analysis['is_increasing']
        is_decreasing = analysis['is_decreasing']
        bullish_divergence = analysis['bullish_divergence']
        bearish_divergence = analysis['bearish_divergence']
        obv_momentum = analysis['OBV_Momentum']
        
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
        
        # 1. Signal d'achat fort: pression acheteuse forte + divergence haussière ou momentum positif fort
        if (signal == OBVSignal.STRONG_BUY or 
            (signal == OBVSignal.BUY and 
             ((bullish_divergence if self.use_divergence else False) or obv_momentum > 5))):
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.STRONG_BUY,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.8,
                source=self.name,
                metadata={
                    "obv": analysis['OBV'],
                    "obv_ema": analysis['OBV_EMA'],
                    "obv_signal": analysis['OBV_Signal'],
                    "is_increasing": is_increasing,
                    "bullish_divergence": bullish_divergence,
                    "obv_momentum": obv_momentum,
                    "signal": signal.value
                }
            ))
        
        # 2. Signal d'achat: pression acheteuse standard
        elif signal == OBVSignal.BUY:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.6,
                source=self.name,
                metadata={
                    "obv": analysis['OBV'],
                    "obv_ema": analysis['OBV_EMA'],
                    "obv_signal": analysis['OBV_Signal'],
                    "is_increasing": is_increasing,
                    "signal": signal.value
                }
            ))
        
        # 3. Signal de vente fort: pression vendeuse forte + divergence baissière ou momentum négatif fort
        elif (signal == OBVSignal.STRONG_SELL or 
              (signal == OBVSignal.SELL and 
               ((bearish_divergence if self.use_divergence else False) or obv_momentum < -5))):
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.STRONG_SELL,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.8,
                source=self.name,
                metadata={
                    "obv": analysis['OBV'],
                    "obv_ema": analysis['OBV_EMA'],
                    "obv_signal": analysis['OBV_Signal'],
                    "is_decreasing": is_decreasing,
                    "bearish_divergence": bearish_divergence,
                    "obv_momentum": obv_momentum,
                    "signal": signal.value
                }
            ))
        
        # 4. Signal de vente: pression vendeuse standard
        elif signal == OBVSignal.SELL:
            signals.append(TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=data.ohlcv['close'].iloc[-1] if isinstance(data, MarketData) else data['close'].iloc[-1],
                confidence=0.6,
                source=self.name,
                metadata={
                    "obv": analysis['OBV'],
                    "obv_ema": analysis['OBV_EMA'],
                    "obv_signal": analysis['OBV_Signal'],
                    "is_decreasing": is_decreasing,
                    "signal": signal.value
                }
            ))
        
        return signals
    
    def backtest(self, data: Union[pd.DataFrame, MarketData], initial_capital: float = 10000.0) -> Dict:
        """
        Effectue un backtest simple de la stratégie sur les données historiques.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            initial_capital: Capital initial pour le backtest
            
        Returns:
            Dictionnaire contenant les résultats du backtest
        """
        # Extraire le DataFrame si MarketData est fourni
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Calculer l'OBV ou le VPT selon le paramètre
        if self.use_vpt:
            df_with_indicators = self.obv_indicator.calculate_volume_price_trend(data)
            indicator_col = 'VPT'
            signal_col = 'VPT_EMA'
        else:
            df_with_indicators = self.obv_indicator.calculate_obv(data)
            indicator_col = 'OBV'
            signal_col = 'OBV_Signal'
        
        # Initialiser les colonnes pour le backtest
        df_with_indicators['signal'] = 0  # 0: pas de signal, 1: achat, -1: vente
        df_with_indicators['position'] = 0  # 0: pas de position, 1: position longue
        df_with_indicators['capital'] = initial_capital
        df_with_indicators['holdings'] = 0.0
        df_with_indicators['total_value'] = initial_capital
        
        # Générer les signaux pour chaque période
        for i in range(self.ema_period + self.signal_period, len(df_with_indicators)):
            # Extraire les données jusqu'à l'index i
            subset_data = df_with_indicators.iloc[:i+1]
            
            # Vérifier les conditions pour les signaux
            obv_value = subset_data[indicator_col].iloc[-1]
            signal_value = subset_data[signal_col].iloc[-1]
            prev_obv = subset_data[indicator_col].iloc[-2]
            prev_signal = subset_data[signal_col].iloc[-2]
            
            # Signal d'achat: OBV croise sa ligne de signal vers le haut
            if prev_obv < prev_signal and obv_value > signal_value:
                df_with_indicators.at[subset_data.index[-1], 'signal'] = 1
            
            # Signal de vente: OBV croise sa ligne de signal vers le bas
            elif prev_obv > prev_signal and obv_value < signal_value:
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
