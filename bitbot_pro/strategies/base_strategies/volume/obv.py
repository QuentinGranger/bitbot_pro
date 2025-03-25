#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratégie basée sur l'indicateur On-Balance Volume (OBV).

L'OBV est un indicateur cumulatif qui mesure la pression acheteuse ou vendeuse
en ajoutant le volume lorsque le prix augmente et en le soustrayant lorsque
le prix diminue. Il permet de détecter les divergences entre le prix et le volume.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional

from bitbot_pro.utils.performance import timeit

logger = logging.getLogger(__name__)


class OBVStrategy:
    """
    Stratégie basée sur l'indicateur On-Balance Volume (OBV).
    
    L'OBV est un indicateur cumulatif qui mesure la pression acheteuse ou vendeuse
    en fonction du volume. L'indicateur ajoute le volume lorsque le prix augmente
    et le soustrait lorsque le prix diminue.
    
    Attributs:
        window (int): Taille de la fenêtre pour le calcul de la moyenne mobile de l'OBV
        signal_window (int): Taille de la fenêtre pour le calcul de la ligne de signal
        use_divergence (bool): Indique si la détection des divergences doit être utilisée
        divergence_window (int): Taille de la fenêtre pour la recherche de divergences
        price_column (str): Colonne de prix à utiliser pour les calculs
    """
    
    def __init__(
        self,
        window: int = 20,
        signal_window: int = 9,
        use_divergence: bool = True,
        divergence_window: int = 14,
        price_column: str = 'close'
    ):
        """
        Initialise la stratégie OBV.
        
        Args:
            window (int): Taille de la fenêtre pour le calcul de la moyenne mobile de l'OBV
            signal_window (int): Taille de la fenêtre pour le calcul de la ligne de signal
            use_divergence (bool): Indique si la détection des divergences doit être utilisée
            divergence_window (int): Taille de la fenêtre pour la recherche de divergences
            price_column (str): Colonne de prix à utiliser pour les calculs
        """
        self.window = window
        self.signal_window = signal_window
        self.use_divergence = use_divergence
        self.divergence_window = divergence_window
        self.price_column = price_column
        
        logger.info(
            f"Stratégie OBV initialisée avec fenêtre: {window}, "
            f"fenêtre de signal: {signal_window}, "
            f"détection de divergence: {use_divergence}, "
            f"fenêtre de divergence: {divergence_window}, "
            f"colonne de prix: {price_column}"
        )
    
    @timeit
    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'indicateur On-Balance Volume (OBV).
        
        Args:
            data: DataFrame contenant au minimum les colonnes 'close' et 'volume'
            
        Returns:
            DataFrame contenant l'OBV et ses dérivés
        """
        # Vérification des colonnes requises
        required_columns = [self.price_column, 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"La colonne '{col}' est requise pour le calcul de l'OBV")
        
        # Copie des données pour éviter les modifications en place
        result = data.copy()
        
        # Calcul de l'OBV de base
        # On-Balance Volume = OBV₀ + Σ(Volume * direction)
        # direction = 1 si prix augmente, -1 si prix diminue, 0 si prix inchangé
        price = result[self.price_column]
        price_change = price.diff()
        
        # Création d'un masque pour les directions
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        
        # Calcul de l'OBV
        volume_direction = result['volume'] * direction
        result['obv'] = volume_direction.cumsum()
        
        # Calcul de la moyenne mobile de l'OBV
        result['obv_ma'] = result['obv'].rolling(window=self.window).mean()
        
        # Calcul de la ligne de signal (moyenne mobile de l'OBV)
        result['obv_signal'] = result['obv'].rolling(window=self.signal_window).mean()
        
        # Calcul des pentes (taux de variation)
        result['obv_slope'] = result['obv'].diff(self.window) / self.window
        result['price_slope'] = price.diff(self.window) / self.window
        
        return result
    
    @timeit
    def identify_divergences(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifie les divergences entre le prix et l'OBV.
        
        Args:
            data: DataFrame contenant au minimum les colonnes 'close' et 'obv'
            
        Returns:
            DataFrame contenant les indicateurs de divergence
        """
        if not self.use_divergence:
            return data
            
        # Copie des données
        result = data.copy()
        
        # Fenêtre pour chercher les extrema locaux
        window = self.divergence_window
        
        # Fonction pour trouver les extrema locaux
        def find_local_extrema(series, window):
            # Ajout de colonnes pour les extrema locaux
            roll_max = series.rolling(window=window, center=True).max()
            roll_min = series.rolling(window=window, center=True).min()
            
            # Un point est un maximum local si sa valeur est égale au maximum dans la fenêtre
            local_max = (series == roll_max) & (series.shift(window//2) < series) & (series.shift(-window//2) < series)
            
            # Un point est un minimum local si sa valeur est égale au minimum dans la fenêtre
            local_min = (series == roll_min) & (series.shift(window//2) > series) & (series.shift(-window//2) > series)
            
            return local_max, local_min
        
        # Trouver les extrema locaux du prix et de l'OBV
        price_high, price_low = find_local_extrema(result[self.price_column], window)
        obv_high, obv_low = find_local_extrema(result['obv'], window)
        
        # Initialiser les colonnes de divergence
        result['bullish_divergence'] = False
        result['bearish_divergence'] = False
        result['hidden_bullish_divergence'] = False
        result['hidden_bearish_divergence'] = False
        
        # Détection des divergences régulières
        # Divergence haussière: prix fait un nouveau plus bas, mais OBV fait un plus bas plus haut
        for i in range(window, len(result) - window):
            if price_low.iloc[i]:
                # Chercher le dernier minimum local du prix
                for j in range(i-1, max(0, i-3*window), -1):
                    if price_low.iloc[j]:
                        # Si le prix a fait un plus bas mais l'OBV non, c'est une divergence haussière
                        if (result[self.price_column].iloc[i] < result[self.price_column].iloc[j] and 
                            result['obv'].iloc[i] > result['obv'].iloc[j] and 
                            obv_low.iloc[i]):
                            result.loc[result.index[i], 'bullish_divergence'] = True
                        break
            
            # Divergence baissière: prix fait un nouveau plus haut, mais OBV fait un plus haut plus bas
            if price_high.iloc[i]:
                # Chercher le dernier maximum local du prix
                for j in range(i-1, max(0, i-3*window), -1):
                    if price_high.iloc[j]:
                        # Si le prix a fait un plus haut mais l'OBV non, c'est une divergence baissière
                        if (result[self.price_column].iloc[i] > result[self.price_column].iloc[j] and 
                            result['obv'].iloc[i] < result['obv'].iloc[j] and 
                            obv_high.iloc[i]):
                            result.loc[result.index[i], 'bearish_divergence'] = True
                        break
            
            # Détection des divergences cachées
            # Divergence haussière cachée: prix fait un plus haut plus haut, mais OBV fait un nouveau plus haut
            if obv_high.iloc[i]:
                for j in range(i-1, max(0, i-3*window), -1):
                    if obv_high.iloc[j]:
                        if (result['obv'].iloc[i] > result['obv'].iloc[j] and 
                            result[self.price_column].iloc[i] < result[self.price_column].iloc[j] and 
                            price_high.iloc[i]):
                            result.loc[result.index[i], 'hidden_bullish_divergence'] = True
                        break
            
            # Divergence baissière cachée: prix fait un plus bas plus bas, mais OBV fait un nouveau plus bas
            if obv_low.iloc[i]:
                for j in range(i-1, max(0, i-3*window), -1):
                    if obv_low.iloc[j]:
                        if (result['obv'].iloc[i] < result['obv'].iloc[j] and 
                            result[self.price_column].iloc[i] > result[self.price_column].iloc[j] and 
                            price_low.iloc[i]):
                            result.loc[result.index[i], 'hidden_bearish_divergence'] = True
                        break
        
        return result
    
    @timeit
    def identify_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifie les croisements entre l'OBV et sa ligne de signal.
        
        Args:
            data: DataFrame contenant au minimum les colonnes 'obv' et 'obv_signal'
            
        Returns:
            DataFrame contenant les indicateurs de croisement
        """
        # Copie des données
        result = data.copy()
        
        # Calcul des positions relatives
        result['obv_above_signal'] = result['obv'] > result['obv_signal']
        result['obv_below_signal'] = result['obv'] < result['obv_signal']
        
        # Détection des croisements
        result['obv_cross_up'] = (result['obv_above_signal']) & (~result['obv_above_signal'].shift(1).fillna(False))
        result['obv_cross_down'] = (result['obv_below_signal']) & (~result['obv_below_signal'].shift(1).fillna(False))
        
        return result
    
    @timeit
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les indicateurs OBV.
        
        Args:
            data: DataFrame contenant les données de marché
            
        Returns:
            DataFrame contenant tous les indicateurs OBV calculés
        """
        # Calcul de l'OBV
        result = self.calculate_obv(data)
        
        # Identification des divergences
        if self.use_divergence:
            result = self.identify_divergences(result)
        
        # Identification des croisements
        result = self.identify_crossovers(result)
        
        return result
    
    @timeit
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur l'OBV.
        
        Args:
            data: DataFrame contenant les indicateurs OBV calculés
            
        Returns:
            DataFrame contenant les signaux de trading
        """
        # Initialiser le DataFrame de signaux
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        
        # Signaux basés sur les croisements OBV/Signal
        signals['signal_buy'] = data['obv_cross_up']
        signals['signal_sell'] = data['obv_cross_down']
        
        # Signaux basés sur les divergences
        if self.use_divergence:
            # Signal d'achat fort: divergence haussière (régulière ou cachée)
            signals['signal_strong_buy'] = data['bullish_divergence'] | data['hidden_bullish_divergence']
            
            # Signal de vente fort: divergence baissière (régulière ou cachée)
            signals['signal_strong_sell'] = data['bearish_divergence'] | data['hidden_bearish_divergence']
        else:
            signals['signal_strong_buy'] = False
            signals['signal_strong_sell'] = False
        
        # Autres signaux (neutres, hold, etc.)
        signals['signal_neutral'] = ~(signals['signal_buy'] | signals['signal_sell'] | 
                                     signals['signal_strong_buy'] | signals['signal_strong_sell'])
        
        # Ajouter les indicateurs principaux pour référence
        signals['obv'] = data['obv']
        signals['obv_ma'] = data['obv_ma']
        signals['obv_signal'] = data['obv_signal']
        signals['price'] = data[self.price_column]
        
        return signals
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applique la stratégie OBV aux données fournies.
        
        Args:
            data: DataFrame contenant les données de marché
            
        Returns:
            Tuple contenant:
                - DataFrame avec tous les indicateurs calculés
                - DataFrame contenant les signaux de trading
        """
        # Vérification des données
        if data.empty:
            logger.warning("Les données d'entrée sont vides")
            return pd.DataFrame(), pd.DataFrame()
            
        # Calcul de tous les indicateurs
        indicators = self.calculate_all_indicators(data)
        
        # Génération des signaux
        signals = self.generate_signals(indicators)
        
        return indicators, signals


class OBVDivergenceStrategy(OBVStrategy):
    """
    Stratégie OBV spécialisée dans la détection des divergences.
    
    Cette variante de la stratégie OBV se concentre sur l'identification des
    divergences entre le prix et l'OBV, qui sont souvent des indicateurs
    avancés de retournement de tendance.
    """
    
    def __init__(
        self,
        window: int = 20,
        signal_window: int = 9,
        divergence_window: int = 20,
        price_column: str = 'close',
        divergence_threshold: float = 0.1
    ):
        """
        Initialise la stratégie OBV de divergence.
        
        Args:
            window (int): Taille de la fenêtre pour le calcul de la moyenne mobile de l'OBV
            signal_window (int): Taille de la fenêtre pour le calcul de la ligne de signal
            divergence_window (int): Taille de la fenêtre pour la recherche de divergences
            price_column (str): Colonne de prix à utiliser pour les calculs
            divergence_threshold (float): Seuil minimum pour considérer une divergence
        """
        super().__init__(
            window=window,
            signal_window=signal_window,
            use_divergence=True,
            divergence_window=divergence_window,
            price_column=price_column
        )
        self.divergence_threshold = divergence_threshold
        
        logger.info(
            f"Stratégie OBV Divergence initialisée avec fenêtre: {window}, "
            f"fenêtre de signal: {signal_window}, "
            f"fenêtre de divergence: {divergence_window}, "
            f"seuil de divergence: {divergence_threshold}"
        )
    
    @timeit
    def identify_divergences(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifie les divergences entre le prix et l'OBV avec un seuil minimum.
        
        Cette version améliorée de la méthode de détection des divergences
        utilise un seuil pour filtrer les divergences non significatives.
        
        Args:
            data: DataFrame contenant au minimum les colonnes 'close' et 'obv'
            
        Returns:
            DataFrame contenant les indicateurs de divergence
        """
        # Utiliser la méthode parente pour détecter les divergences
        result = super().identify_divergences(data)
        
        # Calculer l'amplitude des divergences
        for div_type in ['bullish_divergence', 'bearish_divergence', 
                         'hidden_bullish_divergence', 'hidden_bearish_divergence']:
            if div_type in result.columns:
                # Filtrer les divergences dont l'amplitude est inférieure au seuil
                div_indices = result.index[result[div_type]]
                
                for idx in div_indices:
                    # Calculer les pentes normalisées du prix et de l'OBV sur la fenêtre précédente
                    price_change_pct = result[self.price_column].loc[:idx].pct_change(
                        periods=self.divergence_window).iloc[-1]
                    obv_change_pct = result['obv'].loc[:idx].diff(
                        periods=self.divergence_window).iloc[-1] / (result['obv'].iloc[-1] + 1)
                    
                    # Calculer la différence des pentes normalisées (amplitude de la divergence)
                    divergence_amplitude = abs(price_change_pct - obv_change_pct)
                    
                    # Filtrer les divergences dont l'amplitude est inférieure au seuil
                    if divergence_amplitude < self.divergence_threshold:
                        result.loc[idx, div_type] = False
        
        return result
    
    @timeit
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés principalement sur les divergences OBV.
        
        Args:
            data: DataFrame contenant les indicateurs OBV calculés
            
        Returns:
            DataFrame contenant les signaux de trading
        """
        # Initialiser le DataFrame de signaux
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        
        # Signaux basés sur les divergences (principal)
        signals['signal_buy'] = data['bullish_divergence']
        signals['signal_sell'] = data['bearish_divergence']
        
        # Signaux basés sur les divergences cachées (confirmation de tendance)
        signals['signal_strong_buy'] = data['hidden_bullish_divergence']
        signals['signal_strong_sell'] = data['hidden_bearish_divergence']
        
        # Signaux secondaires basés sur les croisements
        signals['signal_buy_crossover'] = data['obv_cross_up']
        signals['signal_sell_crossover'] = data['obv_cross_down']
        
        # Autres signaux (neutres, hold, etc.)
        signals['signal_neutral'] = ~(signals['signal_buy'] | signals['signal_sell'] | 
                                     signals['signal_strong_buy'] | signals['signal_strong_sell'])
        
        # Ajouter les indicateurs principaux pour référence
        signals['obv'] = data['obv']
        signals['obv_ma'] = data['obv_ma']
        signals['obv_signal'] = data['obv_signal']
        signals['price'] = data[self.price_column]
        
        return signals


class RateOfChangeOBVStrategy(OBVStrategy):
    """
    Stratégie OBV basée sur le taux de variation (ROC) de l'OBV.
    
    Cette variante de la stratégie OBV se concentre sur la vitesse de changement
    de l'OBV plutôt que sur sa valeur absolue, ce qui peut être utile pour
    détecter des accélérations ou décélérations dans le momentum du volume.
    """
    
    def __init__(
        self,
        window: int = 20,
        signal_window: int = 9,
        roc_period: int = 14,
        roc_threshold: float = 0.05,
        price_column: str = 'close'
    ):
        """
        Initialise la stratégie ROC-OBV.
        
        Args:
            window (int): Taille de la fenêtre pour le calcul de la moyenne mobile de l'OBV
            signal_window (int): Taille de la fenêtre pour le calcul de la ligne de signal
            roc_period (int): Période pour le calcul du taux de variation
            roc_threshold (float): Seuil pour les signaux basés sur le ROC
            price_column (str): Colonne de prix à utiliser pour les calculs
        """
        super().__init__(
            window=window,
            signal_window=signal_window,
            use_divergence=False,
            price_column=price_column
        )
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        
        logger.info(
            f"Stratégie ROC-OBV initialisée avec fenêtre: {window}, "
            f"fenêtre de signal: {signal_window}, "
            f"période ROC: {roc_period}, "
            f"seuil ROC: {roc_threshold}"
        )
    
    @timeit
    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'OBV et son taux de variation.
        
        Args:
            data: DataFrame contenant au minimum les colonnes 'close' et 'volume'
            
        Returns:
            DataFrame contenant l'OBV, le ROC-OBV et leurs dérivés
        """
        # Utiliser la méthode parente pour calculer l'OBV
        result = super().calculate_obv(data)
        
        # Calculer le taux de variation (ROC) de l'OBV
        result['obv_roc'] = result['obv'].pct_change(periods=self.roc_period) * 100
        
        # Moyenne mobile du ROC pour lisser le signal
        result['obv_roc_ma'] = result['obv_roc'].rolling(window=self.signal_window).mean()
        
        # Calculer les zones de surachat/survente basées sur le ROC
        result['obv_overbought'] = result['obv_roc'] > self.roc_threshold * 100
        result['obv_oversold'] = result['obv_roc'] < -self.roc_threshold * 100
        
        return result
    
    @timeit
    def identify_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifie les croisements du ROC-OBV avec sa moyenne mobile et avec le zéro.
        
        Args:
            data: DataFrame contenant les indicateurs OBV et ROC-OBV
            
        Returns:
            DataFrame contenant les indicateurs de croisement
        """
        # Utiliser la méthode parente pour les croisements de base
        result = super().identify_crossovers(data)
        
        # Croisements du ROC avec zéro (changement de direction)
        result['roc_above_zero'] = result['obv_roc'] > 0
        result['roc_below_zero'] = result['obv_roc'] < 0
        
        result['roc_cross_above_zero'] = (result['roc_above_zero']) & (~result['roc_above_zero'].shift(1).fillna(False))
        result['roc_cross_below_zero'] = (result['roc_below_zero']) & (~result['roc_below_zero'].shift(1).fillna(False))
        
        # Croisements du ROC avec sa moyenne mobile
        result['roc_above_ma'] = result['obv_roc'] > result['obv_roc_ma']
        result['roc_below_ma'] = result['obv_roc'] < result['obv_roc_ma']
        
        result['roc_cross_above_ma'] = (result['roc_above_ma']) & (~result['roc_above_ma'].shift(1).fillna(False))
        result['roc_cross_below_ma'] = (result['roc_below_ma']) & (~result['roc_below_ma'].shift(1).fillna(False))
        
        return result
    
    @timeit
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les indicateurs ROC-OBV.
        
        Args:
            data: DataFrame contenant les données de marché
            
        Returns:
            DataFrame contenant tous les indicateurs ROC-OBV calculés
        """
        # Calculer l'OBV et le ROC
        result = self.calculate_obv(data)
        
        # Identifier les croisements
        result = self.identify_crossovers(result)
        
        return result
    
    @timeit
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur le ROC-OBV.
        
        Args:
            data: DataFrame contenant les indicateurs ROC-OBV calculés
            
        Returns:
            DataFrame contenant les signaux de trading
        """
        # Initialiser le DataFrame de signaux
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        
        # Signaux basés sur les croisements du zéro
        signals['signal_buy'] = data['roc_cross_above_zero']
        signals['signal_sell'] = data['roc_cross_below_zero']
        
        # Signaux basés sur les conditions de surachat/survente
        # Acheter lorsque le ROC sort de la zone de survente
        signals['signal_strong_buy'] = (data['obv_oversold'].shift(1) & ~data['obv_oversold']) 
        
        # Vendre lorsque le ROC sort de la zone de surachat
        signals['signal_strong_sell'] = (data['obv_overbought'].shift(1) & ~data['obv_overbought'])
        
        # Signaux secondaires basés sur les croisements avec la MA
        signals['signal_buy_crossover'] = data['roc_cross_above_ma']
        signals['signal_sell_crossover'] = data['roc_cross_below_ma']
        
        # Autres signaux (neutres, hold, etc.)
        signals['signal_neutral'] = ~(signals['signal_buy'] | signals['signal_sell'] | 
                                     signals['signal_strong_buy'] | signals['signal_strong_sell'])
        
        # Ajouter les indicateurs principaux pour référence
        signals['obv'] = data['obv']
        signals['obv_ma'] = data['obv_ma']
        signals['obv_roc'] = data['obv_roc']
        signals['obv_roc_ma'] = data['obv_roc_ma']
        signals['price'] = data[self.price_column]
        
        return signals
