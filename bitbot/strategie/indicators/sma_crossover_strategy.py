"""
Module pour la stratégie basée sur le croisement des moyennes mobiles simples (SMA).

Ce module implémente une stratégie de croisement de SMA qui:
1. Utilise les croisements de SMA pour repérer les inversions de tendance à court terme
2. Emploie des croisements rapides (SMA9 vs SMA21) pour détecter les mouvements rapides
3. Applique un filtre ATR pour éviter les faux signaux lors de faibles volatilités
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from datetime import datetime

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.strategie.base.SMA import SMAIndicator
from bitbot.strategie.base.ATR import ATRIndicator, VolatilityLevel
from bitbot.utils.logger import logger

class SMACrossoverStrategy(StrategyBase):
    """
    Stratégie basée sur le croisement des moyennes mobiles simples avec filtre ATR.
    
    Cette stratégie détecte les croisements entre deux moyennes mobiles simples
    pour identifier les inversions de tendance à court terme. Un filtre ATR est appliqué
    pour éviter les faux signaux pendant les périodes de faible volatilité.
    """
    
    def __init__(self, 
                fast_period: int = 9, 
                slow_period: int = 21, 
                atr_period: int = 14, 
                atr_threshold_pct: float = 0.5,
                use_atr_filter: bool = True,
                use_price_rejection: bool = True):
        """
        Initialise la stratégie de croisement SMA avec filtre ATR.
        
        Args:
            fast_period: Période pour la moyenne mobile rapide (défaut: 9)
            slow_period: Période pour la moyenne mobile lente (défaut: 21)
            atr_period: Période pour le calcul de l'ATR (défaut: 14)
            atr_threshold_pct: Seuil ATR en pourcentage pour filtrer les signaux (défaut: 0.5%)
            use_atr_filter: Si True, applique le filtre ATR pour éviter les faux signaux
            use_price_rejection: Si True, vérifie la rejection de prix pour confirmer les signaux
        """
        super().__init__()
        
        self.name = "SMACrossoverStrategy"
        self.description = "Stratégie basée sur le croisement des moyennes mobiles simples"
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_threshold_pct = atr_threshold_pct
        self.use_atr_filter = use_atr_filter
        self.use_price_rejection = use_price_rejection
        
        # Initialiser les indicateurs
        self.sma_indicator = SMAIndicator()
        self.atr_indicator = ATRIndicator(period=atr_period)
        
        logger.info(f"Stratégie de croisement SMA initialisée: "
                   f"SMA{fast_period} vs SMA{slow_period}, "
                   f"Filtre ATR: {'Activé' if use_atr_filter else 'Désactivé'}")
    
    def set_parameters(self, **kwargs):
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir
        """
        for param, value in kwargs.items():
            if param == 'fast_period' and isinstance(value, int) and value > 0:
                self.fast_period = value
            elif param == 'slow_period' and isinstance(value, int) and value > 0:
                self.slow_period = value
            elif param == 'atr_period' and isinstance(value, int) and value > 0:
                self.atr_period = value
                self.atr_indicator.set_period(value)
            elif param == 'atr_threshold_pct' and isinstance(value, (int, float)) and value >= 0:
                self.atr_threshold_pct = value
            elif param == 'use_atr_filter' and isinstance(value, bool):
                self.use_atr_filter = value
            elif param == 'use_price_rejection' and isinstance(value, bool):
                self.use_price_rejection = value
        
        logger.info(f"Paramètres mis à jour: {kwargs}")
    
    def analyze(self, data: Union[pd.DataFrame, MarketData]) -> Dict:
        """
        Analyse les données de marché et génère des signaux de trading.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            Dictionnaire avec le résultat de l'analyse
        """
        # Convertir MarketData en DataFrame si nécessaire
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Vérifier qu'il y a suffisamment de données
        if len(df) < max(self.fast_period, self.slow_period) + 5:
            logger.warning(f"Pas assez de données pour l'analyse: {len(df)} points")
            return {'signal': 0, 'signal_strength': 0, 'message': "Données insuffisantes"}
        
        # Calculer les SMA
        df_with_sma = self.sma_indicator.calculate_sma(
            data, 
            periods=[self.fast_period, self.slow_period]
        )
        
        # Calculer l'ATR si le filtre est activé
        if self.use_atr_filter:
            df_with_atr = self.atr_indicator.calculate_atr(data)
            
            # Fusionner les résultats
            df = pd.concat([df_with_sma, df_with_atr[['atr', 'atr_pct']]], axis=1)
        else:
            df = df_with_sma
        
        # Récupérer les dernières valeurs des SMA
        fast_col = f'sma_{self.fast_period}'
        slow_col = f'sma_{self.slow_period}'
        
        fast_sma_current = df[fast_col].iloc[-1]
        slow_sma_current = df[slow_col].iloc[-1]
        fast_sma_prev = df[fast_col].iloc[-2]
        slow_sma_prev = df[slow_col].iloc[-2]
        
        # Initialiser le résultat
        result = {
            'signal': 0,
            'signal_strength': 0,
            'message': "Pas de signal",
            'fast_sma': fast_sma_current,
            'slow_sma': slow_sma_current,
            'fast_sma_prev': fast_sma_prev,
            'slow_sma_prev': slow_sma_prev
        }
        
        # Détecter les croisements
        golden_cross, death_cross = self.sma_indicator.detect_crossover(
            data, 
            fast_period=self.fast_period, 
            slow_period=self.slow_period
        )
        
        # Signal basé sur le croisement
        signal = 0
        if golden_cross:
            signal = 1
            result['message'] = f"Croisement haussier: SMA{self.fast_period} > SMA{self.slow_period}"
        elif death_cross:
            signal = -1
            result['message'] = f"Croisement baissier: SMA{self.fast_period} < SMA{self.slow_period}"
        
        # Force du signal - basée sur la distance entre les deux moyennes mobiles
        signal_strength = abs(fast_sma_current - slow_sma_current) / slow_sma_current * 100
        result['raw_signal_strength'] = signal_strength
        
        # Appliquer le filtre ATR si activé
        if self.use_atr_filter and signal != 0:
            current_atr_pct = df['atr_pct'].iloc[-1]
            result['atr_pct'] = current_atr_pct
            
            volatility_level = self.atr_indicator.get_volatility_level(data)
            result['volatility_level'] = volatility_level.value
            
            # Vérifier si la volatilité est suffisante
            if current_atr_pct < self.atr_threshold_pct:
                logger.info(f"Signal filtré: volatilité trop faible (ATR: {current_atr_pct:.2f}% < seuil: {self.atr_threshold_pct:.2f}%)")
                signal = 0
                result['message'] = f"Signal ignoré: volatilité insuffisante (ATR: {current_atr_pct:.2f}%)"
            else:
                # Ajuster la force du signal en fonction de la volatilité
                # Plus la volatilité est élevée, plus le signal est fort
                vol_multiplier = min(current_atr_pct / self.atr_threshold_pct, 2.0)
                signal_strength *= vol_multiplier
        
        # Vérifier la rejection de prix si activé
        if self.use_price_rejection and signal != 0:
            last_close = df['close'].iloc[-1]
            last_open = df['open'].iloc[-1]
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            
            # Calculer la taille du corps et des mèches
            body_size = abs(last_close - last_open)
            upper_wick = last_high - max(last_open, last_close)
            lower_wick = min(last_open, last_close) - last_low
            
            result['candle_body'] = body_size
            result['upper_wick'] = upper_wick
            result['lower_wick'] = lower_wick
            
            # Vérifier si la bougie confirme le signal
            if signal == 1 and (last_close < last_open or lower_wick > body_size * 2):
                logger.info("Signal d'achat rejeté: structure de bougie non confirmative")
                signal = 0
                result['message'] = "Signal d'achat ignoré: rejection de prix"
            elif signal == -1 and (last_close > last_open or upper_wick > body_size * 2):
                logger.info("Signal de vente rejeté: structure de bougie non confirmative")
                signal = 0
                result['message'] = "Signal de vente ignoré: rejection de prix"
        
        # Finaliser le résultat
        result['signal'] = signal
        result['signal_strength'] = signal_strength if signal != 0 else 0
        
        return result
    
    def generate_signals(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Génère des signaux de trading pour toute la série temporelle.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec des colonnes de signal ajoutées
        """
        # Convertir MarketData en DataFrame si nécessaire
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Calculer les SMA
        df_with_sma = self.sma_indicator.calculate_sma(
            data, 
            periods=[self.fast_period, self.slow_period]
        )
        
        # Calculer l'ATR si le filtre est activé
        if self.use_atr_filter:
            df_with_atr = self.atr_indicator.calculate_atr(data)
            
            # Fusionner les résultats
            df = pd.concat([df_with_sma, df_with_atr[['atr', 'atr_pct']]], axis=1)
        else:
            df = df_with_sma
        
        # Initialiser les colonnes de signal
        df['signal'] = 0
        df['valid_signal'] = 0
        df['signal_strength'] = 0.0  # Initialiser comme float pour éviter les warnings
        
        # Noms des colonnes SMA
        fast_col = f'sma_{self.fast_period}'
        slow_col = f'sma_{self.slow_period}'
        
        # Détecter les croisements
        for i in range(1, len(df)):
            # Croisement à la hausse (Golden Cross)
            if (df[fast_col].iloc[i-1] <= df[slow_col].iloc[i-1] and 
                df[fast_col].iloc[i] > df[slow_col].iloc[i]):
                df.loc[df.index[i], 'signal'] = 1
            
            # Croisement à la baisse (Death Cross)
            elif (df[fast_col].iloc[i-1] >= df[slow_col].iloc[i-1] and 
                  df[fast_col].iloc[i] < df[slow_col].iloc[i]):
                df.loc[df.index[i], 'signal'] = -1
            
            # Calculer la force du signal
            if df.loc[df.index[i], 'signal'] != 0:
                signal_strength = abs(df[fast_col].iloc[i] - df[slow_col].iloc[i]) / df[slow_col].iloc[i] * 100
                df.loc[df.index[i], 'signal_strength'] = signal_strength
        
        # Appliquer le filtre ATR si activé
        if self.use_atr_filter:
            df['valid_signal'] = np.where(
                (df['signal'] != 0) & (df['atr_pct'] >= self.atr_threshold_pct),
                df['signal'],
                0
            )
        else:
            df['valid_signal'] = df['signal']
        
        return df
    
    def calculate_risk_reward(self, data: Union[pd.DataFrame, MarketData], 
                            signal: int) -> Tuple[float, float, float]:
        """
        Calcule les niveaux de stop loss et take profit basés sur l'ATR.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            signal: Direction du signal (1 pour achat, -1 pour vente)
            
        Returns:
            Tuple (stop_loss, take_profit, risk_reward_ratio)
        """
        if signal == 0:
            return 0, 0, 0
        
        # Calculer l'ATR
        df_with_atr = self.atr_indicator.calculate_atr(data)
        
        # Récupérer le dernier prix et ATR
        last_close = df_with_atr['close'].iloc[-1]
        last_atr = df_with_atr['atr'].iloc[-1]
        
        # Calculer le stop loss (2 x ATR)
        if signal == 1:  # Signal d'achat
            stop_loss = last_close - (2 * last_atr)
            take_profit = last_close + (3 * last_atr)  # 1.5x risk/reward
        else:  # Signal de vente
            stop_loss = last_close + (2 * last_atr)
            take_profit = last_close - (3 * last_atr)  # 1.5x risk/reward
        
        # Calculer le ratio risk/reward
        risk = abs(last_close - stop_loss)
        reward = abs(last_close - take_profit)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return stop_loss, take_profit, risk_reward_ratio
    
    def __str__(self) -> str:
        """Retourne une représentation textuelle de la stratégie."""
        return (f"Stratégie de croisement SMA ({self.name})\n"
                f"- SMA rapide: {self.fast_period}\n"
                f"- SMA lente: {self.slow_period}\n"
                f"- Filtre ATR: {'Activé' if self.use_atr_filter else 'Désactivé'}\n"
                f"- Seuil ATR: {self.atr_threshold_pct:.2f}%\n"
                f"- Vérification rejection: {'Activée' if self.use_price_rejection else 'Désactivée'}")
