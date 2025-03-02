"""
Module pour la gestion dynamique des stop-loss basés sur l'ATR.

Ce module fournit une stratégie pour adapter les stop-loss en fonction de la volatilité
du marché mesurée par l'Average True Range (ATR).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum
from datetime import datetime

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.ATR import ATRIndicator, VolatilityLevel
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.models.trade_signal import TradeSignal, SignalType
from bitbot.utils.logger import logger

class StopLossType(Enum):
    """Types de stop-loss disponibles."""
    FIXED = "Stop-loss fixe"
    TRAILING = "Stop-loss suiveur"
    VOLATILITY_ADJUSTED = "Stop-loss ajusté à la volatilité"
    CHANDELIER = "Stop-loss chandelier"
    MULTI_LEVEL = "Stop-loss multi-niveaux"

class TrailingSLMode(Enum):
    """Modes de fonctionnement du stop-loss suiveur."""
    PERCENT = "Pourcentage du prix"
    ATR_MULTIPLE = "Multiple de l'ATR"
    HYBRID = "Hybride (ATR + pourcentage)"

class ATRStopLossStrategy(StrategyBase):
    """
    Stratégie pour la gestion des stop-loss basée sur l'ATR.
    
    Cette stratégie adapte dynamiquement les stop-loss en fonction de la volatilité
    du marché, mesurée par l'ATR, pour éviter les sorties prématurées et
    protéger efficacement le capital.
    """
    
    def __init__(self, 
                 atr_period: int = 14, 
                 stop_type: StopLossType = StopLossType.VOLATILITY_ADJUSTED,
                 atr_multiplier: float = 2.0,
                 trailing_mode: TrailingSLMode = TrailingSLMode.ATR_MULTIPLE,
                 trailing_factor: float = 1.0,
                 volatility_scaling: bool = True,
                 multi_level_factors: List[float] = None,
                 clean_data: bool = True):
        """
        Initialise la stratégie de stop-loss basée sur l'ATR.
        
        Args:
            atr_period: Période pour le calcul de l'ATR
            stop_type: Type de stop-loss à utiliser
            atr_multiplier: Multiplicateur de base pour l'ATR
            trailing_mode: Mode de fonctionnement du stop-loss suiveur
            trailing_factor: Facteur pour le stop-loss suiveur
            volatility_scaling: Ajuster dynamiquement le multiplicateur selon la volatilité
            multi_level_factors: Liste des facteurs pour les niveaux multiples de stop-loss
            clean_data: Nettoyer automatiquement les données
        """
        super().__init__()
        
        self.name = "ATRStopLossStrategy"
        self.description = "Stratégie de stop-loss adaptative basée sur l'ATR"
        
        self.atr_period = atr_period
        self.stop_type = stop_type
        self.atr_multiplier = atr_multiplier
        self.trailing_mode = trailing_mode
        self.trailing_factor = trailing_factor
        self.volatility_scaling = volatility_scaling
        
        # Valeurs par défaut pour les niveaux multiples (25%, 50%, 75%, 100%)
        self.multi_level_factors = multi_level_factors or [0.25, 0.5, 0.75, 1.0]
        
        # Initialiser l'indicateur ATR
        self.atr_indicator = ATRIndicator(period=atr_period, clean_data=clean_data)
        
        # Variables internes pour le suivi des stops
        self.initial_stop = None
        self.current_stop = None
        self.highest_price = None
        self.lowest_price = None
        self.entry_price = None
        self.is_long_position = True  # Par défaut, considérer une position longue
        
        logger.info(f"Stratégie de stop-loss ATR initialisée: Type={stop_type.value}, "
                   f"Multiplicateur ATR={atr_multiplier}, Période ATR={atr_period}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Met à jour les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à mettre à jour
        """
        super().set_parameters(**kwargs)
        
        if 'atr_period' in kwargs:
            self.atr_period = kwargs['atr_period']
            self.atr_indicator.set_period(self.atr_period)
        
        if 'stop_type' in kwargs:
            self.stop_type = kwargs['stop_type']
        
        if 'atr_multiplier' in kwargs:
            self.atr_multiplier = kwargs['atr_multiplier']
        
        if 'trailing_mode' in kwargs:
            self.trailing_mode = kwargs['trailing_mode']
        
        if 'trailing_factor' in kwargs:
            self.trailing_factor = kwargs['trailing_factor']
        
        if 'volatility_scaling' in kwargs:
            self.volatility_scaling = kwargs['volatility_scaling']
        
        if 'multi_level_factors' in kwargs:
            self.multi_level_factors = kwargs['multi_level_factors']
    
    def initialize_position(self, 
                           entry_price: float, 
                           is_long: bool = True, 
                           initial_stop: float = None,
                           data: Union[pd.DataFrame, MarketData] = None) -> Dict:
        """
        Initialise une nouvelle position avec un stop-loss.
        
        Args:
            entry_price: Prix d'entrée dans la position
            is_long: True pour une position longue, False pour une position courte
            initial_stop: Stop-loss initial, si None, calculé à partir des paramètres
            data: Données de marché pour calculer l'ATR si nécessaire
            
        Returns:
            Dictionnaire contenant les informations de la position et du stop-loss
        """
        self.entry_price = entry_price
        self.is_long_position = is_long
        self.highest_price = entry_price if is_long else None
        self.lowest_price = entry_price if not is_long else None
        
        # Calculer le stop-loss initial si non spécifié
        if initial_stop is None and data is not None:
            if self.stop_type == StopLossType.FIXED:
                # Stop fixe basé sur un pourcentage du prix d'entrée
                risk_pct = 2.0  # 2% par défaut
                self.initial_stop = entry_price * (1 - risk_pct/100) if is_long else entry_price * (1 + risk_pct/100)
            else:
                # Calculer l'ATR et le stop basé sur la volatilité
                atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                self.initial_stop = entry_price - (atr_value * self.atr_multiplier) if is_long else entry_price + (atr_value * self.atr_multiplier)
        else:
            self.initial_stop = initial_stop
        
        self.current_stop = self.initial_stop
        
        return {
            'entry_price': self.entry_price,
            'is_long': self.is_long_position,
            'initial_stop': self.initial_stop,
            'current_stop': self.current_stop
        }

    def update_stop_loss(self, 
                       current_price: float, 
                       data: Union[pd.DataFrame, MarketData] = None) -> Dict:
        """
        Met à jour le stop-loss en fonction du prix actuel et du type de stop-loss.
        
        Args:
            current_price: Prix actuel du marché
            data: Données de marché pour mettre à jour l'ATR si nécessaire
            
        Returns:
            Dictionnaire contenant les informations de stop-loss mises à jour
        """
        if self.entry_price is None:
            raise ValueError("Position non initialisée. Appelez initialize_position d'abord.")
        
        # Mettre à jour les prix hauts/bas atteints
        if self.is_long_position:
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
        else:
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
        
        # Calculer le nouveau stop-loss en fonction du type
        if self.stop_type == StopLossType.FIXED:
            # Le stop-loss fixe ne change pas
            new_stop = self.initial_stop
            
        elif self.stop_type == StopLossType.TRAILING:
            # Stop-loss suiveur basé sur le prix le plus favorable atteint
            if self.is_long_position:
                if self.trailing_mode == TrailingSLMode.PERCENT:
                    # Basé sur un pourcentage du prix le plus haut
                    trailing_amount = self.highest_price * (self.trailing_factor / 100)
                    new_stop = self.highest_price - trailing_amount
                elif self.trailing_mode == TrailingSLMode.ATR_MULTIPLE:
                    # Basé sur un multiple de l'ATR
                    if data is not None:
                        atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                        new_stop = self.highest_price - (atr_value * self.trailing_factor)
                    else:
                        # Si pas de données pour l'ATR, utiliser le stop existant
                        new_stop = self.current_stop
                else:  # TrailingSLMode.HYBRID
                    # Combinaison des deux approches
                    if data is not None:
                        atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                        percent_trail = self.highest_price * (self.trailing_factor / 200)
                        atr_trail = atr_value * (self.trailing_factor / 2)
                        new_stop = self.highest_price - (percent_trail + atr_trail)
                    else:
                        new_stop = self.current_stop
            else:
                # Position courte
                if self.trailing_mode == TrailingSLMode.PERCENT:
                    trailing_amount = self.lowest_price * (self.trailing_factor / 100)
                    new_stop = self.lowest_price + trailing_amount
                elif self.trailing_mode == TrailingSLMode.ATR_MULTIPLE:
                    if data is not None:
                        atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                        new_stop = self.lowest_price + (atr_value * self.trailing_factor)
                    else:
                        new_stop = self.current_stop
                else:  # TrailingSLMode.HYBRID
                    if data is not None:
                        atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                        percent_trail = self.lowest_price * (self.trailing_factor / 200)
                        atr_trail = atr_value * (self.trailing_factor / 2)
                        new_stop = self.lowest_price + (percent_trail + atr_trail)
                    else:
                        new_stop = self.current_stop
                        
        elif self.stop_type == StopLossType.VOLATILITY_ADJUSTED:
            # Stop-loss qui s'ajuste en fonction de la volatilité actuelle
            if data is not None:
                # Calculer l'ATR actuel
                atr_df = self.atr_indicator.calculate_atr(data)
                atr_value = atr_df['atr'].iloc[-1]
                
                # Ajuster le multiplicateur en fonction du niveau de volatilité si demandé
                multiplier = self.atr_multiplier
                if self.volatility_scaling:
                    vol_level = self.atr_indicator.get_volatility_level(data)
                    
                    # Ajuster le multiplicateur en fonction du niveau de volatilité
                    if vol_level == VolatilityLevel.VERY_LOW:
                        multiplier = self.atr_multiplier * 0.7  # Réduire pour volatilité faible
                    elif vol_level == VolatilityLevel.LOW:
                        multiplier = self.atr_multiplier * 0.85
                    elif vol_level == VolatilityLevel.HIGH:
                        multiplier = self.atr_multiplier * 1.15
                    elif vol_level == VolatilityLevel.VERY_HIGH:
                        multiplier = self.atr_multiplier * 1.3
                    elif vol_level == VolatilityLevel.EXTREME:
                        multiplier = self.atr_multiplier * 1.5  # Augmenter pour volatilité extrême
                
                # Calculer le stop en fonction de la volatilité ajustée
                if self.is_long_position:
                    new_stop = current_price - (atr_value * multiplier)
                    # Ne jamais baisser le stop-loss en position longue
                    new_stop = max(new_stop, self.current_stop or 0)
                else:
                    new_stop = current_price + (atr_value * multiplier)
                    # Ne jamais augmenter le stop-loss en position courte
                    new_stop = min(new_stop, self.current_stop or float('inf'))
            else:
                new_stop = self.current_stop
                
        elif self.stop_type == StopLossType.CHANDELIER:
            # Stop-loss chandelier (basé sur le plus haut/bas atteint)
            if data is not None:
                atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                
                if self.is_long_position:
                    new_stop = self.highest_price - (atr_value * self.atr_multiplier)
                    # Ne jamais baisser le stop-loss en position longue
                    new_stop = max(new_stop, self.current_stop or 0)
                else:
                    new_stop = self.lowest_price + (atr_value * self.atr_multiplier)
                    # Ne jamais augmenter le stop-loss en position courte
                    new_stop = min(new_stop, self.current_stop or float('inf'))
            else:
                new_stop = self.current_stop
                
        elif self.stop_type == StopLossType.MULTI_LEVEL:
            # Stop-loss à niveaux multiples (plusieurs stop partiels)
            # Ici, nous retournons simplement une liste de niveaux de stop
            if data is not None:
                atr_value = self.atr_indicator.calculate_atr(data)['atr'].iloc[-1]
                stops = []
                
                for factor in self.multi_level_factors:
                    if self.is_long_position:
                        level_stop = current_price - (atr_value * self.atr_multiplier * factor)
                        stops.append(level_stop)
                    else:
                        level_stop = current_price + (atr_value * self.atr_multiplier * factor)
                        stops.append(level_stop)
                
                # Pour la compatibilité, le stop actuel est le plus conservateur
                if self.is_long_position:
                    new_stop = max(stops)
                else:
                    new_stop = min(stops)
                
                # Stocker les niveaux multiples pour référence
                self.multi_level_stops = stops
            else:
                new_stop = self.current_stop
        else:
            # Type de stop-loss non reconnu
            new_stop = self.current_stop
        
        # Mettre à jour le stop-loss actuel
        old_stop = self.current_stop
        self.current_stop = new_stop
        
        # Journaliser les mises à jour significatives
        if old_stop is not None and abs(new_stop - old_stop) / old_stop > 0.01:  # Changement > 1%
            direction = "rehaussé" if (self.is_long_position and new_stop > old_stop) or (not self.is_long_position and new_stop < old_stop) else "inchangé"
            logger.debug(f"Stop-loss {direction} de {old_stop:.2f} à {new_stop:.2f} ({self.stop_type.value})")
        
        result = {
            'current_price': current_price,
            'current_stop': self.current_stop,
            'stop_type': self.stop_type,
            'is_long': self.is_long_position
        }
        
        # Ajouter les niveaux multiples si applicable
        if self.stop_type == StopLossType.MULTI_LEVEL and hasattr(self, 'multi_level_stops'):
            result['multi_level_stops'] = self.multi_level_stops
            
        return result
    
    def is_stop_triggered(self, current_price: float) -> bool:
        """
        Vérifie si le stop-loss est déclenché au prix actuel.
        
        Args:
            current_price: Prix actuel du marché
            
        Returns:
            True si le stop-loss est déclenché, False sinon
        """
        if self.current_stop is None:
            return False
            
        if self.is_long_position:
            return current_price <= self.current_stop
        else:
            return current_price >= self.current_stop
    
    def calculate_risk_ratio(self, current_price: float) -> float:
        """
        Calcule le ratio risque/récompense actuel.
        
        Args:
            current_price: Prix actuel du marché
            
        Returns:
            Ratio de risque (distance au stop / gain potentiel)
        """
        if self.entry_price is None or self.current_stop is None:
            raise ValueError("Position non initialisée.")
            
        if self.is_long_position:
            risk = current_price - self.current_stop
            reward = current_price - self.entry_price
        else:
            risk = self.current_stop - current_price
            reward = self.entry_price - current_price
            
        # Éviter la division par zéro
        if reward == 0:
            return float('inf')
            
        return risk / reward

    def calculate_volatility_bands(self, 
                                 data: Union[pd.DataFrame, MarketData],
                                 multipliers: List[float] = None) -> pd.DataFrame:
        """
        Calcule les bandes de volatilité basées sur l'ATR.
        
        Ces bandes permettent d'éviter des sorties prématurées lors de mouvements normaux
        du marché, tout en permettant de capturer les tendances à plus long terme.
        
        Args:
            data: Données de marché
            multipliers: Liste des multiplicateurs ATR pour différentes bandes
                         (par défaut: [1.0, 1.5, 2.0, 2.5, 3.0])
            
        Returns:
            DataFrame avec les bandes de volatilité calculées
        """
        if multipliers is None:
            multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
            
        # Utiliser directement la méthode de l'indicateur ATR pour calculer les bandes
        df = self.atr_indicator.calculate_atr_bands(data, multiplier=max(multipliers))
        
        # Préparer les nouvelles colonnes pour les bandes
        for mult in multipliers:
            if mult != max(multipliers):  # Ne pas recalculer la bande déjà calculée
                df[f'upper_band_{mult}'] = df['close'] + (df['atr'] * mult)
                df[f'lower_band_{mult}'] = df['close'] - (df['atr'] * mult)
                
        return df
        
    def calculate_adaptive_bands(self, 
                                data: Union[pd.DataFrame, MarketData],
                                base_multiplier: float = 2.0,
                                volatility_scaling: bool = True) -> pd.DataFrame:
        """
        Calcule des bandes adaptatives qui s'ajustent en fonction de la volatilité.
        
        Ces bandes s'élargissent pendant les périodes de forte volatilité et se
        rétrécissent pendant les périodes de faible volatilité.
        
        Args:
            data: Données de marché
            base_multiplier: Multiplicateur de base pour l'ATR
            volatility_scaling: Ajuster dynamiquement les bandes selon la volatilité
            
        Returns:
            DataFrame avec les bandes adaptatives calculées
        """
        # Calculer l'ATR
        df = self.atr_indicator.calculate_atr(data).copy()
        
        # Calculer le changement de volatilité sur une période
        vol_change = self.atr_indicator.calculate_volatility_change(data)
        df['volatility_change'] = vol_change['volatility_change_pct']
        
        # Déterminer le niveau de volatilité
        if volatility_scaling:
            df['volatility_level'] = df.apply(
                lambda row: self.atr_indicator.get_volatility_level_row(row), axis=1
            )
            
            # Ajuster le multiplicateur en fonction du niveau de volatilité
            def adjust_multiplier(vol_level, base_mult):
                if vol_level == VolatilityLevel.VERY_LOW:
                    return base_mult * 0.7
                elif vol_level == VolatilityLevel.LOW:
                    return base_mult * 0.85
                elif vol_level == VolatilityLevel.NORMAL:
                    return base_mult
                elif vol_level == VolatilityLevel.HIGH:
                    return base_mult * 1.15
                elif vol_level == VolatilityLevel.VERY_HIGH:
                    return base_mult * 1.3
                elif vol_level == VolatilityLevel.EXTREME:
                    return base_mult * 1.5
                return base_mult
                
            df['adjusted_multiplier'] = df['volatility_level'].apply(
                lambda x: adjust_multiplier(x, base_multiplier)
            )
        else:
            df['adjusted_multiplier'] = base_multiplier
        
        # Calculer les bandes adaptatives
        df['upper_adaptive_band'] = df['close'] + (df['atr'] * df['adjusted_multiplier'])
        df['lower_adaptive_band'] = df['close'] - (df['atr'] * df['adjusted_multiplier'])
        
        return df
            
    def stress_test_stop_loss(self, 
                            data: Union[pd.DataFrame, MarketData],
                            entry_price: float, 
                            initial_stop: float,
                            is_long: bool = True,
                            stop_type: StopLossType = None,
                            volatility_increase: float = 2.0,
                            n_scenarios: int = 5) -> Dict:
        """
        Effectue un stress test du stop-loss dans différents scénarios de volatilité.
        
        Cette méthode simule ce qui arriverait au stop-loss si la volatilité augmentait
        soudainement d'un certain facteur. Elle est utile pour s'assurer que le stop-loss
        reste efficace même en cas de conditions de marché extrêmes.
        
        Args:
            data: Données de marché historiques
            entry_price: Prix d'entrée de la position
            initial_stop: Stop-loss initial
            is_long: True pour une position longue, False pour une position courte
            stop_type: Type de stop-loss à tester (utilise celui de l'instance si None)
            volatility_increase: Facteur multiplicatif de la volatilité pour le stress test
            n_scenarios: Nombre de scénarios de volatilité à tester
            
        Returns:
            Dictionnaire contenant les résultats du stress test pour chaque scénario
        """
        stop_type = stop_type or self.stop_type
        results = {}
        
        # Créer une copie des données pour ne pas modifier l'original
        df = data.copy() if isinstance(data, pd.DataFrame) else data.data.copy()
        
        # Calculer l'ATR original
        atr_df = self.atr_indicator.calculate_atr(df)
        original_atr = atr_df['atr'].iloc[-1]
        
        # Pour chaque scénario, augmenter la volatilité et recalculer le stop-loss
        for i in range(n_scenarios):
            # Augmenter progressivement la volatilité pour chaque scénario
            volatility_factor = 1.0 + (i + 1) * (volatility_increase - 1.0) / n_scenarios
            
            # Simuler l'ATR augmenté
            simulated_atr = original_atr * volatility_factor
            
            # Créer une instance temporaire avec le type de stop requis
            temp_strategy = ATRStopLossStrategy(
                stop_type=stop_type,
                atr_multiplier=self.atr_multiplier,
                trailing_mode=self.trailing_mode,
                trailing_factor=self.trailing_factor,
                volatility_scaling=self.volatility_scaling
            )
            
            # Initialiser la position
            temp_strategy.initialize_position(
                entry_price=entry_price,
                is_long=is_long,
                initial_stop=initial_stop
            )
            
            # Pour calculer le stop-loss dans ce scénario, nous devons créer un DataFrame
            # avec une valeur ATR modifiée
            last_row = df.iloc[-1:].copy()
            
            # Remplacer l'ATR par la valeur simulée dans la copie temporaire
            temp_df = atr_df.iloc[-1:].copy()
            temp_df['atr'] = simulated_atr
            
            # Calculer le stop-loss dans ce scénario
            current_price = last_row['close'].iloc[0]
            stop_info = temp_strategy.update_stop_loss(current_price, data=temp_df)
            
            # Stocker les résultats
            scenario_name = f"volatility_x{volatility_factor:.1f}"
            results[scenario_name] = {
                'volatility_factor': volatility_factor,
                'simulated_atr': simulated_atr,
                'original_stop': initial_stop,
                'stressed_stop': stop_info['current_stop'],
                'stop_change_pct': ((stop_info['current_stop'] - initial_stop) / initial_stop) * 100,
                'price_to_stop_pct': ((current_price - stop_info['current_stop']) / current_price) * 100 if is_long else
                                    ((stop_info['current_stop'] - current_price) / current_price) * 100
            }
            
        return results

    def apply_strategy(self, 
                     data: Union[pd.DataFrame, MarketData],
                     entry_signal_col: str = None,
                     exit_signal_col: str = None) -> pd.DataFrame:
        """
        Applique la stratégie de stop-loss sur des données historiques.
        
        Cette méthode simule l'application de la stratégie sur l'ensemble des données,
        en utilisant les signaux d'entrée et de sortie fournis, ou en générant des
        signaux basiques si non fournis.
        
        Args:
            data: Données de marché historiques
            entry_signal_col: Nom de la colonne contenant les signaux d'entrée en position
            exit_signal_col: Nom de la colonne contenant les signaux de sortie de position
            
        Returns:
            DataFrame avec les résultats de la stratégie appliquée
        """
        # Préparer le DataFrame
        df = data.copy() if isinstance(data, pd.DataFrame) else data.data.copy()
        
        # Calculer l'ATR
        df = self.atr_indicator.calculate_atr(df)
        
        # Si les colonnes de signal ne sont pas fournies, créer des signaux basiques
        # basés sur la volatilité pour l'exemple
        if entry_signal_col is None:
            breakout_df = self.atr_indicator.is_volatility_breakout(df)
            df['entry_signal'] = breakout_df['is_breakout']
            entry_signal_col = 'entry_signal'
            
        if exit_signal_col is None:
            # Créer une colonne de sortie basée sur l'entrée (pour initialiser)
            df['exit_signal'] = False
            exit_signal_col = 'exit_signal'
        
        # Initialiser les colonnes pour le suivi des positions
        df['in_position'] = False
        df['position_type'] = None  # 'long' ou 'short'
        df['entry_price'] = None
        df['stop_loss'] = None
        df['exit_price'] = None
        df['exit_type'] = None  # 'signal', 'stop', 'end'
        df['profit_pct'] = None
        
        # Variables pour suivre l'état actuel
        in_position = False
        entry_price = None
        entry_index = None
        position_type = None
        current_stop = None
        
        # Parcourir les données pour simuler la stratégie
        for i in range(1, len(df)):
            # Récupérer les données actuelles
            current_row = df.iloc[i]
            previous_row = df.iloc[i-1]
            current_price = current_row['close']
            
            # Mettre à jour l'état de position pour la ligne actuelle
            df.at[i, 'in_position'] = in_position
            
            if in_position:
                df.at[i, 'position_type'] = position_type
                df.at[i, 'entry_price'] = entry_price
                
                # Mettre à jour le stop-loss si nous sommes en position
                if self.stop_type != StopLossType.FIXED:
                    # Créer un DataFrame temporaire avec les données jusqu'à la ligne actuelle
                    temp_data = df.iloc[:i+1].copy()
                    stop_info = self.update_stop_loss(current_price, data=temp_data)
                    current_stop = stop_info['current_stop']
                    
                df.at[i, 'stop_loss'] = current_stop
                
                # Vérifier si le stop-loss est déclenché
                stop_triggered = False
                
                if position_type == 'long' and current_row['low'] <= current_stop:
                    # Pour une position longue, le stop est déclenché si le prix bas touche le stop
                    stop_triggered = True
                    exit_price = current_stop  # Utiliser le niveau de stop comme prix de sortie
                elif position_type == 'short' and current_row['high'] >= current_stop:
                    # Pour une position courte, le stop est déclenché si le prix haut touche le stop
                    stop_triggered = True
                    exit_price = current_stop
                
                # Vérifier s'il y a un signal de sortie ou si le stop est déclenché
                if current_row[exit_signal_col] or stop_triggered:
                    # Sortir de la position
                    in_position = False
                    
                    # Calculer le profit en pourcentage
                    if stop_triggered:
                        exit_type = 'stop'
                        exit_price = current_stop
                    else:
                        exit_type = 'signal'
                        exit_price = current_price
                    
                    profit_pct = ((exit_price / entry_price) - 1) * 100 if position_type == 'long' else \
                                 ((entry_price / exit_price) - 1) * 100
                    
                    # Mettre à jour les données pour la sortie
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_type'] = exit_type
                    df.at[i, 'profit_pct'] = profit_pct
                    df.at[i, 'in_position'] = False
                    
                    # Réinitialiser les variables de position
                    entry_price = None
                    entry_index = None
                    position_type = None
                    current_stop = None
            
            # Vérifier s'il y a un signal d'entrée et que nous ne sommes pas déjà en position
            elif current_row[entry_signal_col] and not in_position:
                # Déterminer le type de position (long/short) - pour simplifier, on suppose
                # que le signal est directionnel (True pour long, False pour short)
                is_long = True  # Par défaut, long
                
                # Entrer en position
                in_position = True
                entry_price = current_price
                entry_index = i
                position_type = 'long' if is_long else 'short'
                
                # Initialiser le stop-loss
                temp_data = df.iloc[:i+1].copy()
                position_info = self.initialize_position(
                    entry_price=entry_price,
                    is_long=is_long,
                    data=temp_data
                )
                current_stop = position_info['initial_stop']
                
                # Mettre à jour les données pour l'entrée
                df.at[i, 'in_position'] = True
                df.at[i, 'position_type'] = position_type
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss'] = current_stop
        
        # Pour la dernière ligne, si nous sommes toujours en position, simuler une sortie
        if in_position and i == len(df) - 1:
            exit_price = df.iloc[-1]['close']
            exit_type = 'end'
            
            profit_pct = ((exit_price / entry_price) - 1) * 100 if position_type == 'long' else \
                         ((entry_price / exit_price) - 1) * 100
            
            df.at[i, 'exit_price'] = exit_price
            df.at[i, 'exit_type'] = exit_type
            df.at[i, 'profit_pct'] = profit_pct
        
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyse les résultats de la stratégie appliquée.
        
        Args:
            results_df: DataFrame contenant les résultats de apply_strategy
            
        Returns:
            Dictionnaire contenant diverses métriques d'analyse
        """
        # Filtrer pour obtenir uniquement les lignes où il y a eu une sortie
        trades = results_df[results_df['exit_price'].notnull()].copy()
        
        # S'il n'y a pas de trades, retourner des statistiques vides
        if len(trades) == 0:
            return {
                'n_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'stop_hit_rate': 0
            }
        
        # Calculer les statistiques de base
        n_trades = len(trades)
        profitable_trades = trades[trades['profit_pct'] > 0]
        losing_trades = trades[trades['profit_pct'] <= 0]
        
        n_profitable = len(profitable_trades)
        n_losing = len(losing_trades)
        
        win_rate = (n_profitable / n_trades) * 100 if n_trades > 0 else 0
        
        avg_profit = trades['profit_pct'].mean() if n_trades > 0 else 0
        max_profit = trades['profit_pct'].max() if n_trades > 0 else 0
        max_loss = trades['profit_pct'].min() if n_trades > 0 else 0
        
        # Calculer le profit factor (somme des gains / somme des pertes)
        total_profit = profitable_trades['profit_pct'].sum() if n_profitable > 0 else 0
        total_loss = abs(losing_trades['profit_pct'].sum()) if n_losing > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculer le taux de déclenchement des stops
        stop_hits = trades[trades['exit_type'] == 'stop']
        stop_hit_rate = (len(stop_hits) / n_trades) * 100 if n_trades > 0 else 0
        
        # Calculer la distribution des types de sortie
        exit_types = trades['exit_type'].value_counts(normalize=True) * 100
        
        # Calculer des statistiques spécifiques au stop-loss
        avg_risk_reward = trades.apply(
            lambda row: abs(row['entry_price'] - row['stop_loss']) / abs(row['exit_price'] - row['entry_price'])
            if row['exit_type'] != 'stop' else None, axis=1
        ).mean()
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'stop_hit_rate': stop_hit_rate,
            'exit_types': exit_types.to_dict(),
            'avg_risk_reward': avg_risk_reward
        }
    
    def run_backtest(self, 
                   data: Union[pd.DataFrame, MarketData],
                   entry_signal_col: str = None,
                   exit_signal_col: str = None,
                   plot_results: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Exécute un backtest complet de la stratégie de stop-loss.
        
        Args:
            data: Données de marché historiques
            entry_signal_col: Nom de la colonne contenant les signaux d'entrée
            exit_signal_col: Nom de la colonne contenant les signaux de sortie
            plot_results: Si True, affiche un graphique des résultats
            
        Returns:
            Tuple (DataFrame avec résultats, Dict avec métriques)
        """
        # Appliquer la stratégie
        results = self.apply_strategy(
            data=data,
            entry_signal_col=entry_signal_col,
            exit_signal_col=exit_signal_col
        )
        
        # Analyser les résultats
        metrics = self.analyze_results(results)
        
        # Afficher les résultats si demandé
        if plot_results:
            try:
                import matplotlib.pyplot as plt
                
                # Créer une figure avec plusieurs sous-graphiques
                fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
                
                # Tracer le prix et les stop-loss
                ax1 = axes[0]
                ax1.plot(results.index, results['close'], label='Prix')
                
                # Tracer les points d'entrée et de sortie
                entries = results[results['entry_price'].notnull()]
                exits = results[results['exit_price'].notnull()]
                
                ax1.scatter(entries.index, entries['entry_price'], marker='^', color='green', s=100, label='Entrée')
                ax1.scatter(exits.index, exits['exit_price'], marker='v', color='red', s=100, label='Sortie')
                
                # Tracer les stop-loss pour les périodes en position
                for i in range(len(results)):
                    if results.iloc[i]['in_position']:
                        ax1.plot(results.index[i], results.iloc[i]['stop_loss'], 'ro', alpha=0.3)
                
                ax1.set_title('Stratégie de Stop-Loss ATR')
                ax1.set_ylabel('Prix')
                ax1.legend()
                
                # Tracer l'ATR
                ax2 = axes[1]
                ax2.plot(results.index, results['atr'], label='ATR')
                ax2.set_ylabel('ATR')
                ax2.legend()
                
                # Tracer les profits
                ax3 = axes[2]
                trades = results[results['exit_price'].notnull()]
                ax3.bar(trades.index, trades['profit_pct'], label='Profit %')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_ylabel('Profit %')
                ax3.legend()
                
                # Ajouter des annotations pour les métriques clés
                textstr = f"Nombre de trades: {metrics['n_trades']}\n"
                textstr += f"Win rate: {metrics['win_rate']:.1f}%\n"
                textstr += f"Profit moyen: {metrics['avg_profit']:.2f}%\n"
                textstr += f"Profit factor: {metrics['profit_factor']:.2f}\n"
                textstr += f"Taux de stop: {metrics['stop_hit_rate']:.1f}%"
                
                fig.text(0.02, 0.02, textstr, fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                logger.warning("Matplotlib non disponible pour l'affichage des résultats")
        
        return results, metrics
