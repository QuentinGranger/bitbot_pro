"""
Module pour la fusion et la synchronisation des données de différentes sources.

Ce module permet d'uniformiser les timestamps entre différentes sources de données,
de fusionner des séries temporelles avec différentes fréquences et de gérer les
décalages temporels entre les sources de données.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta
import pytz
import logging

from bitbot.models.market_data import MarketData

logger = logging.getLogger(__name__)

class DataMerger:
    """
    Classe pour fusionner et synchroniser des données de différentes sources.
    
    Cette classe fournit des méthodes pour:
    - Uniformiser les timestamps entre différentes sources
    - Rééchantillonner des données avec différentes fréquences
    - Aligner des séries temporelles pour analyses multi-sources
    """
    
    def __init__(self):
        """Initialise le fusionneur de données."""
        self.merged_data = {}
        self.timezone = pytz.UTC
    
    def set_timezone(self, timezone_str: str) -> None:
        """
        Définit le fuseau horaire pour la synchronisation des données.
        
        Args:
            timezone_str: Chaîne représentant le fuseau horaire (ex: 'UTC', 'Europe/Paris')
        """
        try:
            self.timezone = pytz.timezone(timezone_str)
            logger.info(f"Fuseau horaire défini sur: {timezone_str}")
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Fuseau horaire inconnu: {timezone_str}. Utilisation de UTC.")
            self.timezone = pytz.UTC
    
    def normalize_timestamps(self, 
                           df: pd.DataFrame, 
                           freq: str = None, 
                           method: str = 'nearest') -> pd.DataFrame:
        """
        Normalise les timestamps d'un DataFrame pour qu'ils correspondent aux intervalles exacts.
        
        Par exemple, si freq='5min', les timestamps seront alignés sur des intervalles de 5 minutes
        (00:00, 00:05, 00:10, etc.) pour éviter les petits décalages entre différentes sources.
        
        Args:
            df: DataFrame avec un index temporel
            freq: Fréquence de normalisation (ex: '1min', '5min', '1h', '1d')
            method: Méthode d'assignation ('nearest', 'forward', 'backward')
                   - 'nearest': assigne au timestamp normalisé le plus proche
                   - 'forward': assigne au prochain timestamp normalisé
                   - 'backward': assigne au timestamp normalisé précédent
        
        Returns:
            DataFrame avec des timestamps normalisés
        """
        if df.empty:
            return df
        
        if freq is None:
            # Déterminer automatiquement la fréquence
            if len(df) > 1:
                # Calculer la différence médiane entre les timestamps
                time_diffs = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
                median_diff = time_diffs.median()
                
                # Convertir en minutes et arrondir à l'intervalle le plus proche
                minutes = median_diff.total_seconds() / 60
                if minutes < 1:
                    freq = '1min'
                elif minutes < 15:
                    freq = '5min'
                elif minutes < 60:
                    freq = '15min'
                elif minutes < 240:
                    freq = '1h'
                else:
                    freq = '1d'
                
                logger.info(f"Fréquence automatiquement détectée: {freq}")
            else:
                freq = '1h'
        
        # Convertir les timestamps en timezone spécifiée
        df_tz = df.copy()
        if df_tz.index.tz is None:
            df_tz.index = df_tz.index.tz_localize('UTC')
        
        df_tz.index = df_tz.index.tz_convert(self.timezone)
        
        # Rééchantillonner pour normaliser les timestamps
        if method == 'nearest':
            # Créer une grille temporelle régulière
            start = df_tz.index.min().floor(freq)
            end = df_tz.index.max().ceil(freq)
            regular_grid = pd.date_range(start=start, end=end, freq=freq, tz=self.timezone)
            
            # Trouver le timestamp normalisé le plus proche pour chaque timestamp original
            nearest_indices = []
            for idx in df_tz.index:
                grid_distances = np.abs(regular_grid - idx.to_pydatetime())
                nearest_idx = np.argmin(grid_distances)
                nearest_indices.append(regular_grid[nearest_idx])
            
            # Créer un nouveau DataFrame avec les timestamps normalisés
            df_normalized = pd.DataFrame(index=nearest_indices, data=df_tz.values, columns=df_tz.columns)
            
            # Gérer les doublons potentiels en prenant la moyenne
            df_normalized = df_normalized.groupby(df_normalized.index).mean()
            
        elif method == 'forward':
            # Assigner chaque timestamp au prochain timestamp normalisé
            df_normalized = df_tz.copy()
            df_normalized.index = df_normalized.index.ceil(freq)
            df_normalized = df_normalized.groupby(df_normalized.index).mean()
            
        elif method == 'backward':
            # Assigner chaque timestamp au timestamp normalisé précédent
            df_normalized = df_tz.copy()
            df_normalized.index = df_normalized.index.floor(freq)
            df_normalized = df_normalized.groupby(df_normalized.index).mean()
            
        else:
            raise ValueError(f"Méthode invalide: {method}. Utilisez 'nearest', 'forward' ou 'backward'.")
        
        return df_normalized
    
    def merge_market_data(self, 
                        market_data_list: List[MarketData], 
                        normalize: bool = True, 
                        target_freq: str = None) -> pd.DataFrame:
        """
        Fusionne plusieurs objets MarketData en synchronisant leurs timestamps.
        
        Args:
            market_data_list: Liste d'objets MarketData à fusionner
            normalize: Si True, normalise les timestamps
            target_freq: Fréquence cible pour la normalisation
        
        Returns:
            DataFrame avec toutes les données fusionnées et alignées
        """
        if not market_data_list:
            return pd.DataFrame()
        
        # Collecter tous les DataFrames
        dfs = []
        symbols = []
        
        for market_data in market_data_list:
            if market_data.ohlcv.empty:
                logger.warning(f"Données vides pour {market_data.symbol} ({market_data.timeframe})")
                continue
                
            # Copier et renommer les colonnes pour éviter les conflits
            df = market_data.ohlcv.copy()
            
            # Déterminer la fréquence si non spécifiée
            if target_freq is None:
                target_freq = self._timeframe_to_freq(market_data.timeframe)
                
            # Normaliser les timestamps si demandé
            if normalize:
                df = self.normalize_timestamps(df, freq=target_freq)
            
            # Ajouter des préfixes aux colonnes avec le symbole
            symbol = market_data.symbol.replace('/', '_')
            df = df.add_prefix(f"{symbol}_")
            
            dfs.append(df)
            symbols.append(market_data.symbol)
        
        if not dfs:
            logger.warning("Aucune donnée valide à fusionner")
            return pd.DataFrame()
        
        # Fusionner tous les DataFrames
        merged_df = dfs[0]
        
        for i in range(1, len(dfs)):
            # Outer join pour préserver toutes les timestamps
            merged_df = pd.merge(
                merged_df, 
                dfs[i], 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # Trier par timestamp
        merged_df.sort_index(inplace=True)
        
        # Garder une trace de cette fusion
        key = "_".join(symbols)
        self.merged_data[key] = merged_df
        
        logger.info(f"Fusion réussie de {len(dfs)} sources de données: {', '.join(symbols)}")
        
        return merged_df
    
    def merge_with_alternative_data(self, 
                                  market_data: MarketData, 
                                  alt_data: Dict[str, pd.DataFrame],
                                  normalize: bool = True) -> pd.DataFrame:
        """
        Fusionne des données de marché avec des données alternatives.
        
        Args:
            market_data: Objet MarketData principal
            alt_data: Dictionnaire de DataFrames contenant des données alternatives
                     (clé = nom de la source, valeur = DataFrame)
            normalize: Si True, normalise les timestamps
            
        Returns:
            DataFrame fusionné avec les données de marché et alternatives
        """
        if market_data.ohlcv.empty:
            logger.warning(f"Données de marché vides pour {market_data.symbol}")
            return pd.DataFrame()
        
        # Commencer avec les données de marché
        merged_df = market_data.ohlcv.copy()
        symbol = market_data.symbol.replace('/', '_')
        
        # Renommer les colonnes pour éviter les conflits
        merged_df = merged_df.add_prefix(f"{symbol}_")
        
        # Déterminer la fréquence
        target_freq = self._timeframe_to_freq(market_data.timeframe)
        
        # Normaliser les timestamps si demandé
        if normalize:
            merged_df = self.normalize_timestamps(merged_df, freq=target_freq)
        
        # Fusionner avec chaque source de données alternative
        for source_name, alt_df in alt_data.items():
            if alt_df.empty:
                logger.warning(f"Données alternatives vides pour {source_name}")
                continue
                
            # Copier et renommer les colonnes
            source_df = alt_df.copy()
            
            # Normaliser les timestamps des données alternatives
            if normalize:
                # Utiliser une méthode différente selon la fréquence des données alternatives
                if len(source_df) > 1:
                    time_diff = (source_df.index[1] - source_df.index[0]).total_seconds()
                    if time_diff > 86400:  # > 1 jour
                        # Pour les données à basse fréquence, utiliser 'forward' pour avoir des données futures
                        source_df = self.normalize_timestamps(source_df, freq=target_freq, method='forward')
                    else:
                        source_df = self.normalize_timestamps(source_df, freq=target_freq)
                else:
                    source_df = self.normalize_timestamps(source_df, freq=target_freq)
            
            # Ajouter des préfixes aux colonnes
            source_df = source_df.add_prefix(f"{source_name}_")
            
            # Fusionner
            merged_df = pd.merge(
                merged_df, 
                source_df, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # Trier par timestamp
        merged_df.sort_index(inplace=True)
        
        return merged_df
    
    def align_multi_timeframe_data(self, 
                                 market_data_dict: Dict[str, MarketData],
                                 target_timeframe: str) -> pd.DataFrame:
        """
        Aligne les données de différents timeframes en les convertissant vers le timeframe cible.
        
        Args:
            market_data_dict: Dictionnaire d'objets MarketData avec différents timeframes
                            (clé = timeframe, valeur = MarketData)
            target_timeframe: Timeframe cible pour l'alignement
            
        Returns:
            DataFrame avec toutes les données alignées sur le timeframe cible
        """
        if not market_data_dict:
            return pd.DataFrame()
        
        # Déterminer la fréquence cible
        target_freq = self._timeframe_to_freq(target_timeframe)
        
        # Préparer les DataFrames pour chaque timeframe
        aligned_dfs = {}
        
        for timeframe, market_data in market_data_dict.items():
            if market_data.ohlcv.empty:
                continue
                
            df = market_data.ohlcv.copy()
            current_freq = self._timeframe_to_freq(timeframe)
            
            # Assurer que l'index est un DatetimeIndex avec timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # Convertir à la frequency cible
            if current_freq != target_freq:
                # Si on passe d'une fréquence plus élevée à une plus basse (ex: 1m -> 1h)
                if self._freq_to_minutes(current_freq) < self._freq_to_minutes(target_freq):
                    # Agréger les données
                    resampled = df.resample(target_freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                # Si on passe d'une fréquence plus basse à une plus élevée (ex: 1h -> 1m)
                else:
                    # Forward fill pour les données OHLC, mais pas pour le volume
                    ohlc_cols = ['open', 'high', 'low', 'close']
                    vol_cols = ['volume']
                    
                    # Créer une grille temporelle régulière
                    start = df.index.min()
                    end = df.index.max()
                    new_index = pd.date_range(start=start, end=end, freq=target_freq, tz='UTC')
                    
                    # Réindexer avec le nouvel index
                    resampled = pd.DataFrame(index=new_index)
                    
                    # Réindexer les colonnes OHLC avec forward fill
                    for col in ohlc_cols:
                        if col in df.columns:
                            resampled[col] = df[col].reindex(new_index, method='ffill')
                    
                    # Réindexer le volume avec distribution proportionnelle
                    if 'volume' in df.columns:
                        # Déterminer le facteur de réduction
                        ratio = self._freq_to_minutes(current_freq) / self._freq_to_minutes(target_freq)
                        
                        # Distribuer le volume uniformément
                        vol_series = df['volume'].reindex(new_index, method='ffill')
                        resampled['volume'] = vol_series / ratio
            else:
                resampled = df.copy()
            
            # Préfixer les colonnes pour identifier la source
            resampled = resampled.add_prefix(f"{timeframe}_")
            aligned_dfs[timeframe] = resampled
        
        if not aligned_dfs:
            logger.warning("Aucune donnée valide à aligner")
            return pd.DataFrame()
        
        # Fusionner tous les DataFrames alignés
        result = None
        
        for timeframe, df in aligned_dfs.items():
            if result is None:
                result = df
            else:
                result = pd.merge(
                    result, 
                    df, 
                    left_index=True, 
                    right_index=True, 
                    how='outer'
                )
        
        # Trier par timestamp
        result.sort_index(inplace=True)
        
        return result
    
    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convertit un timeframe en fréquence pandas."""
        if timeframe.endswith('m'):
            return timeframe.replace('m', 'min')
        elif timeframe.endswith('h'):
            return timeframe
        elif timeframe.endswith('d'):
            return timeframe.replace('d', 'D')
        elif timeframe.endswith('w'):
            return timeframe.replace('w', 'W')
        else:
            return timeframe
    
    def _freq_to_minutes(self, freq: str) -> int:
        """Convertit une fréquence en minutes."""
        if 'min' in freq:
            return int(freq.replace('min', ''))
        elif 'h' in freq:
            return int(freq.replace('h', '')) * 60
        elif 'D' in freq:
            return int(freq.replace('D', '')) * 60 * 24
        elif 'W' in freq:
            return int(freq.replace('W', '')) * 60 * 24 * 7
        else:
            return 60  # par défaut 1h
