"""
Module pour le nettoyage et la préparation des données de marché.

Ce module fournit des outils pour détecter et corriger les valeurs aberrantes,
normaliser les données, et assurer la qualité des données utilisées par les
algorithmes de trading et les modèles d'IA.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter

from bitbot.models.market_data import MarketData, Kline
from bitbot.utils.logger import logger


class CleaningMethod(str, Enum):
    """Méthodes disponibles pour la correction des valeurs aberrantes."""
    
    REMOVE = "remove"
    INTERPOLATE = "interpolate"
    CLIP = "clip"
    MEDIAN_WINDOW = "median_window"
    KALMAN = "kalman"
    SAVGOL = "savgol"


def apply_kalman_filter(series: pd.Series, process_variance: float = 1e-5, 
                       measurement_variance: float = 0.1) -> pd.Series:
    """
    Applique un filtre de Kalman pour lisser une série temporelle
    
    Args:
        series: Série de données à filtrer
        process_variance: Variance du processus (Q), plus c'est petit, plus le lissage est fort
        measurement_variance: Variance de mesure (R), plus c'est grand, moins on fait confiance aux données
        
    Returns:
        Série filtrée
    """
    # Convertir en array numpy pour traitement
    measurements = np.asarray(series)
    n_measurements = len(measurements)
    
    # Initialiser le filtre de Kalman
    kf = KalmanFilter(dim_x=2, dim_z=1)  # État: [position, vélocité]
    
    # Matrice de transition (comment l'état évolue)
    kf.F = np.array([[1., 1.],
                      [0., 1.]])
    
    # Matrice de mesure (comment les mesures sont liées à l'état)
    kf.H = np.array([[1., 0.]])
    
    # Covariance du processus
    kf.Q = np.array([[process_variance, 0],
                      [0, process_variance]])
    
    # Covariance de mesure
    kf.R = np.array([[measurement_variance]])
    
    # État initial
    kf.x = np.array([[measurements[0]], [0.]])
    
    # Covariance initiale
    kf.P = np.array([[1., 0.],
                      [0., 1.]])
    
    # Stocker les états filtrés
    filtered_state_means = np.zeros(n_measurements)
    
    # Filtrer en avant
    for i, measurement in enumerate(measurements):
        kf.predict()
        kf.update(np.array([measurement]))
        filtered_state_means[i] = kf.x[0, 0]
    
    return pd.Series(filtered_state_means, index=series.index)


def apply_savgol_filter(series: pd.Series, window_length: int = 11, polyorder: int = 2) -> pd.Series:
    """
    Applique un filtre Savitzky-Golay pour lisser une série temporelle
    
    Args:
        series: Série de données à filtrer
        window_length: Longueur de la fenêtre (doit être impair)
        polyorder: Ordre du polynôme d'ajustement (doit être inférieur à window_length)
        
    Returns:
        Série filtrée
    """
    # Ajuster window_length si nécessaire
    if window_length >= len(series):
        window_length = max(min(len(series) - 1, 11), 3)
        # S'assurer que window_length est impair
        if window_length % 2 == 0:
            window_length -= 1
    
    # Ajuster polyorder si nécessaire
    if polyorder >= window_length:
        polyorder = max(min(window_length - 1, 2), 1)
    
    # Appliquer le filtre
    filtered_data = savgol_filter(series.values, window_length, polyorder)
    
    return pd.Series(filtered_data, index=series.index)


class DataCleaner:
    """
    Classe pour le nettoyage et la préparation des données de marché.
    
    Cette classe fournit des méthodes pour:
    - Détecter et corriger les valeurs aberrantes en utilisant des seuils dynamiques
    - Gérer les données manquantes
    - Normaliser les données pour les modèles d'IA
    - Valider la qualité des données
    """
    
    def __init__(self):
        """Initialise le nettoyeur de données."""
        self.cleaning_stats = {}  # Statistiques de nettoyage par symbole
    
    def clean_market_data(self, 
                          market_data: MarketData, 
                          std_threshold: float = 3.0,
                          window_size: int = 20,
                          method: CleaningMethod = CleaningMethod.INTERPOLATE,
                          filter_type: str = None,
                          filter_params: Dict = None,
                          handle_missing: bool = True,
                          max_gap_minutes: int = 5,
                          missing_method: str = "linear",
                          columns: List[str] = None,
                          corruption_threshold: float = 0.1) -> MarketData:
        """
        Nettoie les données de marché en détectant et corrigeant les valeurs aberrantes.
        
        Args:
            market_data: Données de marché à nettoyer
            std_threshold: Nombre d'écarts-types pour définir le seuil (par défaut: 3.0)
            window_size: Taille de la fenêtre pour calculer la volatilité locale
            method: Méthode de nettoyage (suppression, interpolation, écrêtage)
            filter_type: Type de filtre à appliquer après nettoyage ("kalman", "savgol", None)
            filter_params: Paramètres du filtre si applicable
            handle_missing: Si True, détecte et corrige les valeurs manquantes
            max_gap_minutes: Écart maximum en minutes pour interpoler les valeurs manquantes
            missing_method: Méthode d'interpolation pour les valeurs manquantes
            columns: Colonnes à nettoyer (par défaut: ['open', 'high', 'low', 'close', 'volume'])
            corruption_threshold: Seuil de corruption (pourcentage de valeurs manquantes) au-delà duquel 
                                une période est considérée comme corrompue (défaut: 0.1 = 10%)
            
        Returns:
            Données de marché nettoyées
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Créer une copie pour éviter de modifier l'original
        market_data_copy = MarketData(market_data.symbol, market_data.timeframe)
        market_data_copy.ohlcv = market_data.ohlcv.copy()
        market_data_copy.indicators = {k: v.copy() if hasattr(v, 'copy') else v 
                                      for k, v in market_data.indicators.items()}
        market_data_copy.metadata = market_data.metadata.copy()
        
        # Initialiser les stats de nettoyage pour ce symbole
        symbol = market_data.symbol
        if symbol not in self.cleaning_stats:
            self.cleaning_stats[symbol] = {
                'outliers_detected': 0,
                'outliers_corrected': 0,
                'missing_detected': 0,
                'missing_filled': 0,
                'corrupted_periods_removed': 0
            }
        
        # 1. Gestion des valeurs manquantes (si activée)
        if handle_missing and not market_data_copy.ohlcv.empty:
            # Détecter les valeurs manquantes
            with_missing = detect_missing_values(market_data_copy.ohlcv, market_data.timeframe)
            
            # Compter les valeurs manquantes
            missing_count = with_missing['missing'].sum()
            self.cleaning_stats[symbol]['missing_detected'] += missing_count
            
            if missing_count > 0:
                logger.info(f"Traitement des valeurs manquantes pour {symbol}: {missing_count} détectées")
                
                # Supprimer les périodes trop corrompues
                cleaned_df = remove_corrupted_periods(with_missing, market_data.timeframe, corruption_threshold)
                
                # Vérifier combien de périodes ont été supprimées
                points_removed = len(with_missing) - len(cleaned_df)
                if points_removed > 0:
                    logger.warning(f"Suppression de {points_removed} points de données corrompus pour {symbol}")
                    self.cleaning_stats[symbol]['corrupted_periods_removed'] += points_removed
                
                # Remplir les valeurs manquantes pour les petits gaps
                filled_df = fill_missing_values(
                    cleaned_df, 
                    max_gap_minutes=max_gap_minutes, 
                    method=missing_method
                )
                
                # Calculer le nombre de valeurs remplies
                filled_count = missing_count - (missing_count - len(filled_df))
                self.cleaning_stats[symbol]['missing_filled'] += filled_count
                
                if filled_count > 0:
                    logger.info(f"Valeurs manquantes interpolées pour {symbol}: {filled_count} ({filled_count/missing_count:.2%})")
                
                # Mettre à jour les données
                market_data_copy.ohlcv = filled_df
        
        # 2. Correction des valeurs aberrantes
        total_outliers_detected = 0
        total_outliers_corrected = 0
        
        for column in columns:
            if column in market_data_copy.ohlcv.columns:
                cleaned_series, outliers_detected, outliers_corrected = self._clean_column(
                    market_data_copy.ohlcv[column],
                    std_threshold=std_threshold,
                    window_size=window_size,
                    method=method
                )
                market_data_copy.ohlcv[column] = cleaned_series
                total_outliers_detected += outliers_detected
                total_outliers_corrected += outliers_corrected
                
        # Mettre à jour les statistiques
        self.cleaning_stats[symbol]['outliers_detected'] += total_outliers_detected
        self.cleaning_stats[symbol]['outliers_corrected'] += total_outliers_corrected
        
        # Ajouter les méta-informations de nettoyage
        outlier_percent = (total_outliers_detected / len(market_data_copy.ohlcv)) * 100 if len(market_data_copy.ohlcv) > 0 else 0
        logger.info(f"Nettoyage des données pour {symbol}: {total_outliers_detected} valeurs aberrantes détectées ({outlier_percent:.2f}%), {total_outliers_corrected} corrigées")
        
        # 3. Appliquer un filtre supplémentaire si spécifié
        if filter_type and not market_data_copy.ohlcv.empty:
            filter_params = filter_params or {}
            
            if filter_type.lower() == "kalman":
                for column in columns:
                    if column in market_data_copy.ohlcv.columns:
                        market_data_copy.ohlcv[column] = apply_kalman_filter(
                            market_data_copy.ohlcv[column],
                            **filter_params
                        )
                market_data_copy.metadata['filter_type'] = 'kalman'
                market_data_copy.metadata['filter_params'] = filter_params
                
            elif filter_type.lower() == "savgol":
                for column in columns:
                    if column in market_data_copy.ohlcv.columns:
                        market_data_copy.ohlcv[column] = apply_savgol_filter(
                            market_data_copy.ohlcv[column],
                            **filter_params
                        )
                market_data_copy.metadata['filter_type'] = 'savgol'
                market_data_copy.metadata['filter_params'] = filter_params
        
        return market_data_copy
    
    def _clean_column(self, 
                      series: pd.Series, 
                      std_threshold: float = 3.0,
                      window_size: int = 20,
                      method: CleaningMethod = CleaningMethod.INTERPOLATE) -> Tuple[pd.Series, int, int]:
        """
        Nettoie une série de données en détectant et corrigeant les valeurs aberrantes.
        
        Args:
            series: Série de données à nettoyer
            std_threshold: Nombre d'écarts-types pour définir le seuil
            window_size: Taille de la fenêtre pour calculer la volatilité locale
            method: Méthode de nettoyage
            
        Returns:
            Tuple contenant (série nettoyée, nombre d'outliers détectés, nombre d'outliers corrigés)
        """
        cleaned_series = series.copy()
        outliers_detected = 0
        outliers_corrected = 0
        
        # Utilisation d'une fenêtre glissante pour calculer la moyenne et l'écart-type locaux
        rolling_mean = series.rolling(window=window_size, center=True).mean()
        rolling_std = series.rolling(window=window_size, center=True).std()
        
        # Pour les premières et dernières valeurs sans fenêtre complète, utiliser les valeurs globales
        mean_global = series.mean()
        std_global = series.std()
        
        # Remplacer les NaN par les valeurs globales
        rolling_mean.fillna(mean_global, inplace=True)
        rolling_std.fillna(std_global, inplace=True)
        
        # Calcul des limites supérieure et inférieure
        upper_bound = rolling_mean + (std_threshold * rolling_std)
        lower_bound = rolling_mean - (std_threshold * rolling_std)
        
        # Indices des valeurs aberrantes
        outliers = (series > upper_bound) | (series < lower_bound)
        outliers_idx = outliers[outliers].index
        
        # Nombre d'outliers détectés
        outliers_detected = len(outliers_idx)
        
        if outliers_detected > 0:
            # Application de la méthode de nettoyage choisie
            if method == CleaningMethod.REMOVE:
                # Pas adapté aux séries temporelles OHLCV
                # On utilise l'interpolation à la place
                cleaned_series[outliers_idx] = np.nan
                cleaned_series = cleaned_series.ffill()  # Forward fill
                cleaned_series = cleaned_series.bfill()  # Backward fill
                outliers_corrected = outliers_detected
                
            elif method == CleaningMethod.INTERPOLATE:
                cleaned_series[outliers_idx] = np.nan
                cleaned_series = cleaned_series.ffill()  # Forward fill
                cleaned_series = cleaned_series.bfill()  # Backward fill
                outliers_corrected = outliers_detected
                
            elif method == CleaningMethod.CLIP:
                cleaned_series = cleaned_series.clip(lower=lower_bound, upper=upper_bound)
                outliers_corrected = outliers_detected
                
            elif method == CleaningMethod.MEDIAN_WINDOW:
                for idx in outliers_idx:
                    pos = series.index.get_loc(idx)
                    start = max(0, pos - window_size // 2)
                    end = min(len(series), pos + window_size // 2)
                    window_values = series.iloc[start:end]
                    cleaned_series.loc[idx] = window_values.median()
                outliers_corrected = outliers_detected
                
            elif method == CleaningMethod.KALMAN:
                cleaned_series = apply_kalman_filter(cleaned_series)
                outliers_corrected = outliers_detected
                
            elif method == CleaningMethod.SAVGOL:
                cleaned_series = apply_savgol_filter(cleaned_series)
                outliers_corrected = outliers_detected
        
        return cleaned_series, outliers_detected, outliers_corrected
    
    def clean_klines(self, 
                     klines: List[Kline], 
                     std_threshold: float = 3.0,
                     window_size: int = 20,
                     method: CleaningMethod = CleaningMethod.INTERPOLATE,
                     filter_type: str = None,
                     filter_params: Dict = None) -> List[Kline]:
        """
        Nettoie une liste de Klines en détectant et corrigeant les valeurs aberrantes.
        
        Args:
            klines: Liste de Klines à nettoyer
            std_threshold: Nombre d'écarts-types pour définir le seuil
            window_size: Taille de la fenêtre pour calculer la volatilité locale
            method: Méthode de nettoyage
            filter_type: Type de filtre à appliquer après nettoyage ("kalman", "savgol", None)
            filter_params: Paramètres du filtre si applicable
            
        Returns:
            Liste de Klines nettoyées
        """
        if not klines:
            return []
        
        # Conversion en DataFrame pour faciliter le nettoyage
        data = []
        for k in klines:
            data.append({
                'timestamp': k.timestamp,
                'open': float(k.open),
                'high': float(k.high),
                'low': float(k.low),
                'close': float(k.close),
                'volume': float(k.volume),
                'close_time': k.close_time,
                'quote_volume': float(k.quote_volume),
                'trades': k.trades,
                'taker_buy_volume': float(k.taker_buy_volume),
                'taker_buy_quote_volume': float(k.taker_buy_quote_volume),
                'interval': k.interval
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Colonnes à nettoyer
        columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Compteurs pour les statistiques
        outliers_detected = 0
        outliers_corrected = 0
        
        # Traitement colonne par colonne
        for column in columns:
            df[column], col_outliers, col_corrected = self._clean_column(
                df[column], 
                std_threshold=std_threshold,
                window_size=window_size,
                method=method
            )
            
            # Appliquer un filtre si demandé
            if filter_type:
                if filter_type == "kalman":
                    params = filter_params or {}
                    process_var = params.get("process_variance", 1e-5)
                    measure_var = params.get("measurement_variance", 0.1)
                    df[column] = apply_kalman_filter(
                        df[column], process_var, measure_var
                    )
                elif filter_type == "savgol":
                    params = filter_params or {}
                    window_len = params.get("window_length", 11)
                    polyorder = params.get("polyorder", 2)
                    df[column] = apply_savgol_filter(
                        df[column], window_len, polyorder
                    )
            
            outliers_detected += col_outliers
            outliers_corrected += col_corrected
        
        # Enregistrement des statistiques
        symbol = klines[0].interval  # Utiliser l'intervalle comme identifiant
        self.cleaning_stats[symbol] = {
            'timestamp': datetime.now(),
            'outliers_detected': outliers_detected,
            'outliers_corrected': outliers_corrected,
            'total_data_points': len(df) * len(columns),
            'outlier_percentage': (outliers_detected / (len(df) * len(columns))) * 100 if len(df) > 0 else 0
        }
        
        # Reconversion en Klines
        cleaned_klines = []
        df.reset_index(inplace=True)
        
        for _, row in df.iterrows():
            kline = Kline(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                close_time=row['close_time'],
                quote_volume=row['quote_volume'],
                trades=row['trades'],
                taker_buy_volume=row['taker_buy_volume'],
                taker_buy_quote_volume=row['taker_buy_quote_volume'],
                interval=row['interval']
            )
            cleaned_klines.append(kline)
        
        logger.info(f"Nettoyage des Klines pour {symbol}: "
                   f"{outliers_detected} valeurs aberrantes détectées "
                   f"({self.cleaning_stats[symbol]['outlier_percentage']:.2f}%), "
                   f"{outliers_corrected} corrigées")
        
        return cleaned_klines
    
    def get_cleaning_stats(self, symbol: str = None) -> Dict:
        """
        Récupère les statistiques de nettoyage.
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Statistiques de nettoyage
        """
        if symbol:
            return self.cleaning_stats.get(symbol, {})
        return self.cleaning_stats
    
    def verify_ohlc_integrity(self, market_data: MarketData) -> Dict:
        """
        Vérifie l'intégrité des données OHLC.
        
        Args:
            market_data: Données de marché à vérifier
            
        Returns:
            Dictionnaire avec les résultats de la vérification
        """
        df = market_data.ohlcv
        if df.empty:
            return {'status': 'error', 'message': 'Données vides'}
        
        issues = []
        
        # Vérification que high >= open, close, low
        high_violations = df[~(df['high'] >= df['open']) | ~(df['high'] >= df['close']) | ~(df['high'] >= df['low'])].index
        if not high_violations.empty:
            issues.append({
                'type': 'high_violation',
                'count': len(high_violations),
                'indexes': high_violations.tolist()
            })
        
        # Vérification que low <= open, close, high
        low_violations = df[~(df['low'] <= df['open']) | ~(df['low'] <= df['close']) | ~(df['low'] <= df['high'])].index
        if not low_violations.empty:
            issues.append({
                'type': 'low_violation',
                'count': len(low_violations),
                'indexes': low_violations.tolist()
            })
        
        # Vérification des valeurs négatives pour le volume
        negative_volume = df[df['volume'] < 0].index
        if not negative_volume.empty:
            issues.append({
                'type': 'negative_volume',
                'count': len(negative_volume),
                'indexes': negative_volume.tolist()
            })
        
        # Vérification des écarts temporels
        if len(df) > 1:
            expected_interval = pd.Timedelta(market_data.timeframe.replace('m', ' minutes').replace('h', ' hours').replace('d', ' days'))
            time_diffs = df.index.to_series().diff().dropna()
            irregular_intervals = time_diffs[time_diffs != expected_interval].index
            
            if not irregular_intervals.empty:
                issues.append({
                    'type': 'irregular_interval',
                    'count': len(irregular_intervals),
                    'indexes': irregular_intervals.tolist()
                })
        
        # Résultat de la vérification
        result = {
            'status': 'ok' if not issues else 'issues_found',
            'symbol': market_data.symbol,
            'timeframe': market_data.timeframe,
            'data_points': len(df),
            'start_date': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': df.index.max().strftime('%Y-%m-%d %H:%M:%S'),
            'issues': issues
        }
        
        return result


def detect_missing_values(ohlcv_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Détecte les valeurs manquantes dans un DataFrame OHLCV.
    
    Args:
        ohlcv_df: DataFrame OHLCV avec un index temporel
        timeframe: Intervalle de temps (ex: "1m", "5m", "1h", "1d")
        
    Returns:
        DataFrame avec une colonne booléenne 'missing' indiquant où sont les valeurs manquantes
    """
    # S'assurer que l'index est de type datetime
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        try:
            ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        except:
            logger.warning("Impossible de convertir l'index en DatetimeIndex")
            return ohlcv_df.copy()
    
    # Convertir le timeframe en timedelta
    if timeframe.endswith('m'):
        delta = timedelta(minutes=int(timeframe[:-1]))
    elif timeframe.endswith('h'):
        delta = timedelta(hours=int(timeframe[:-1]))
    elif timeframe.endswith('d'):
        delta = timedelta(days=int(timeframe[:-1]))
    else:
        logger.warning(f"Format de timeframe non reconnu: {timeframe}")
        return ohlcv_df.copy()
    
    # Vérifier si l'index est trié
    if not ohlcv_df.index.is_monotonic_increasing:
        ohlcv_df = ohlcv_df.sort_index()
    
    # Créer un index théorique complet
    ideal_index = pd.date_range(
        start=ohlcv_df.index.min(),
        end=ohlcv_df.index.max(),
        freq=delta
    )
    
    # Identifier les timestamps manquants
    missing_timestamps = ideal_index.difference(ohlcv_df.index)
    logger.info(f"Valeurs manquantes détectées: {len(missing_timestamps)}")
    
    # Créer un masque pour identifier les valeurs manquantes
    result = ohlcv_df.copy()
    result['missing'] = False
    
    if len(missing_timestamps) > 0:
        # Ajouter les timestamps manquants avec NaN
        missing_df = pd.DataFrame(index=missing_timestamps)
        missing_df['missing'] = True
        
        # Combiner les deux DataFrames
        result = pd.concat([result, missing_df])
        result = result.sort_index()
    
    return result

def detect_corrupted_periods(ohlcv_df: pd.DataFrame, timeframe: str, corruption_threshold: float = 0.1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Détecte les périodes corrompues dans les données où il y a trop de valeurs manquantes.
    
    Args:
        ohlcv_df: DataFrame OHLCV avec une colonne 'missing'
        timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
        corruption_threshold: Seuil de corruption (pourcentage de valeurs manquantes) au-delà duquel 
                             une période est considérée comme corrompue (défaut: 0.1 = 10%)
    
    Returns:
        Liste de tuples (début, fin) des périodes corrompues
    """
    # Vérifier si la colonne 'missing' existe
    if 'missing' not in ohlcv_df.columns:
        logger.warning("La colonne 'missing' n'existe pas. Utilisez detect_missing_values d'abord.")
        return []
    
    # La taille de la fenêtre dépend du timeframe
    if timeframe.endswith('m'):
        window_size = 60  # 1h pour les minutes
    elif timeframe.endswith('h'):
        window_size = 24  # 1 jour pour les heures
    elif timeframe.endswith('d'):
        window_size = 7   # 1 semaine pour les jours
    else:
        logger.warning(f"Format de timeframe non reconnu: {timeframe}")
        window_size = 20  # valeur par défaut
    
    corrupted_periods = []
    
    # Créer une série temporelle de valeurs manquantes (True/False)
    missing_series = ohlcv_df['missing']
    
    # Si la série est vide ou ne contient pas de valeurs manquantes, retourner une liste vide
    if missing_series.empty or not missing_series.any():
        return []
    
    # Calculer le taux de valeurs manquantes dans des fenêtres glissantes
    missing_rate = missing_series.rolling(window=window_size, min_periods=1).mean()
    
    # Identifier les points où le taux dépasse le seuil
    corrupted_mask = missing_rate > corruption_threshold
    
    if not corrupted_mask.any():
        return []
    
    # Trouver les débuts et fins des périodes corrompues
    corrupted_regions = []
    in_corrupted_region = False
    start_idx = None
    
    for i, is_corrupted in enumerate(corrupted_mask):
        if is_corrupted and not in_corrupted_region:
            # Début d'une nouvelle région corrompue
            in_corrupted_region = True
            start_idx = i
        elif not is_corrupted and in_corrupted_region:
            # Fin d'une région corrompue
            in_corrupted_region = False
            corrupted_regions.append((start_idx, i-1))
    
    # Ne pas oublier la dernière région si elle est toujours active
    if in_corrupted_region:
        corrupted_regions.append((start_idx, len(corrupted_mask)-1))
    
    # Convertir les indices en timestamps
    timestamps = ohlcv_df.index
    corrupted_periods = [(timestamps[start], timestamps[end]) for start, end in corrupted_regions]
    
    return corrupted_periods

def remove_corrupted_periods(ohlcv_df: pd.DataFrame, 
                             timeframe: str, 
                             corruption_threshold: float = 0.1) -> pd.DataFrame:
    """
    Supprime les périodes trop corrompues (trop de valeurs manquantes) des données.
    
    Args:
        ohlcv_df: DataFrame OHLCV avec une colonne 'missing'
        timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
        corruption_threshold: Seuil de corruption (pourcentage de valeurs manquantes) au-delà duquel 
                             une période est considérée comme corrompue (défaut: 0.1 = 10%)
    
    Returns:
        DataFrame sans les périodes corrompues
    """
    # Vérifier si la colonne 'missing' existe
    if 'missing' not in ohlcv_df.columns:
        logger.warning("La colonne 'missing' n'existe pas. Utilisez detect_missing_values d'abord.")
        return ohlcv_df
    
    # Détecter les périodes corrompues
    corrupted_periods = detect_corrupted_periods(ohlcv_df, timeframe, corruption_threshold)
    
    if not corrupted_periods:
        logger.info("Aucune période corrompue détectée.")
        return ohlcv_df
    
    # Créer une copie pour ne pas modifier l'original
    result = ohlcv_df.copy()
    
    # Supprimer les périodes corrompues
    for start, end in corrupted_periods:
        logger.warning(f"Suppression d'une période corrompue : {start} à {end}")
        result = result.drop(result.loc[start:end].index)
    
    logger.info(f"Suppression de {len(corrupted_periods)} périodes corrompues. Points de données restants : {len(result)} sur {len(ohlcv_df)} originaux.")
    
    return result

def fill_missing_values(ohlcv_df: pd.DataFrame, 
                        max_gap_minutes: int = 5, 
                        method: str = "linear") -> pd.DataFrame:
    """
    Remplit les valeurs manquantes dans un DataFrame OHLCV.
    
    Args:
        ohlcv_df: DataFrame OHLCV avec une colonne 'missing'
        max_gap_minutes: Écart maximum en minutes pour l'interpolation (défaut: 5 minutes)
        method: Méthode d'interpolation ('linear', 'ffill', 'bfill', 'cubic', 'polynomial')
        
    Returns:
        DataFrame avec les valeurs manquantes remplies selon la méthode spécifiée
    """
    # Vérifier si la colonne 'missing' existe
    if 'missing' not in ohlcv_df.columns:
        logger.warning("La colonne 'missing' n'existe pas. Utilisez detect_missing_values d'abord.")
        return ohlcv_df
    
    # Créer une copie pour ne pas modifier l'original
    result = ohlcv_df.copy()
    
    # Extraire les timestamps des valeurs manquantes
    missing_rows = result[result['missing']].index
    
    if len(missing_rows) == 0:
        # Pas de valeurs manquantes
        return result.drop(columns=['missing'])
    
    # Déterminer les gaps et leur taille
    if isinstance(result.index, pd.DatetimeIndex):
        # Calculer la durée de chaque gap
        for timestamp in missing_rows:
            # Trouver l'entrée précédente et suivante non manquante
            prev_timestamps = result.index[result.index < timestamp]
            next_timestamps = result.index[result.index > timestamp]
            
            if len(prev_timestamps) == 0 or len(next_timestamps) == 0:
                # Ne pas interpoler si nous sommes au début ou à la fin
                continue
            
            prev_timestamp = prev_timestamps[-1]
            next_timestamp = next_timestamps[0]
            
            # Calculer la durée du gap en minutes
            gap_duration = (next_timestamp - prev_timestamp).total_seconds() / 60
            
            # Interpoler uniquement si le gap est petit
            if gap_duration <= max_gap_minutes:
                # Créer une vue du gap (précédent, manquant, suivant)
                gap_indices = [prev_timestamp, timestamp, next_timestamp]
                gap_view = result.loc[gap_indices]
                
                # Interpoler les valeurs numériques uniquement
                numeric_columns = gap_view.select_dtypes(include=['number']).columns
                numeric_columns = [col for col in numeric_columns if col != 'missing']
                
                if len(numeric_columns) > 0:
                    # Appliquer l'interpolation
                    result.loc[timestamp, numeric_columns] = gap_view.loc[
                        [prev_timestamp, next_timestamp], numeric_columns
                    ].interpolate(method=method).loc[prev_timestamp:next_timestamp].iloc[1]
                    
                    # Marquer comme interpolée
                    result.loc[timestamp, 'missing'] = False
                    
                    logger.debug(f"Valeur manquante interpolée à {timestamp}")
                else:
                    logger.warning("Aucune colonne numérique à interpoler")
                    
    # Supprimer toutes les valeurs manquantes restantes
    result = result[~result['missing']]
    
    # Supprimer la colonne 'missing'
    if 'missing' in result.columns:
        result = result.drop(columns=['missing'])
    
    return result


def auto_select_filter(market_data: MarketData, use_case: str = "general") -> dict:
    """
    Sélectionne automatiquement la meilleure méthode de filtre en fonction des données et du cas d'utilisation.
    
    Args:
        market_data: Données de marché à analyser
        use_case: Cas d'utilisation ("general", "trend_following", "mean_reversion", "breakout", "volatility")
        
    Returns:
        Dictionnaire de configuration pour le nettoyage et le filtrage
    """
    # Configuration par défaut
    config = {
        "std_threshold": 3.0,
        "window_size": 20,
        "method": "interpolate",
        "filter_type": None,
        "filter_params": {}
    }
    
    # Extraire le timeframe et calculer la volatilité
    timeframe = market_data.timeframe
    
    # Calculer les métriques de volatilité
    returns = market_data.ohlcv['close'].pct_change().dropna()
    volatility = returns.std()
    
    # Ajuster le seuil d'outliers en fonction de la volatilité
    if volatility > 0.03:  # Très volatile
        config["std_threshold"] = 4.0  # Plus tolérant pour les mouvements extrêmes
    elif volatility < 0.005:  # Peu volatile
        config["std_threshold"] = 2.5  # Plus strict pour les outliers
    
    # Sélectionner la méthode de filtrage en fonction du timeframe et du cas d'utilisation
    if timeframe in ["1m", "3m", "5m"]:
        if use_case in ["trend_following", "general"]:
            # Pour les stratégies de suivi de tendance, lissage fort
            config["filter_type"] = "kalman"
            config["filter_params"] = {
                "process_variance": 1e-5,
                "measurement_variance": 0.1
            }
        elif use_case == "mean_reversion":
            # Pour le mean reversion, moins de lissage
            config["filter_type"] = "savgol"
            config["filter_params"] = {
                "window_length": 11,
                "polyorder": 3
            }
        elif use_case == "breakout":
            # Pour les stratégies de breakout, conserver plus de détails
            config["filter_type"] = "kalman"
            config["filter_params"] = {
                "process_variance": 1e-4,
                "measurement_variance": 0.05
            }
        elif use_case == "volatility":
            # Pour les stratégies basées sur la volatilité, lissage moyen
            config["filter_type"] = "savgol"
            config["filter_params"] = {
                "window_length": 7,
                "polyorder": 2
            }
    
    elif timeframe in ["15m", "30m"]:
        if use_case in ["trend_following", "general"]:
            # Filtre de Savitzky-Golay pour timeframes intermédiaires
            config["filter_type"] = "savgol"
            config["filter_params"] = {
                "window_length": 15,
                "polyorder": 3
            }
        elif use_case in ["breakout", "volatility"]:
            # Moins de lissage pour breakout et volatilité
            config["filter_type"] = "savgol"
            config["filter_params"] = {
                "window_length": 9,
                "polyorder": 3
            }
    
    elif timeframe in ["1h", "2h", "4h", "6h"]:
        # Filtrage plus léger pour timeframes plus longs
        if use_case in ["trend_following", "general"]:
            config["filter_type"] = "kalman"
            config["filter_params"] = {
                "process_variance": 1e-4,
                "measurement_variance": 0.05
            }
        # Pour les autres cas d'utilisation, un simple nettoyage d'outliers suffit
    
    elif timeframe in ["1d", "3d", "1w"]:
        # Pour les timeframes journaliers ou plus, juste nettoyage d'outliers
        config["filter_type"] = None
    
    # Ajuster la taille de la fenêtre en fonction du timeframe
    if timeframe in ["1m", "3m", "5m"]:
        config["window_size"] = 30
    elif timeframe in ["15m", "30m"]:
        config["window_size"] = 20
    elif timeframe in ["1h", "2h", "4h"]:
        config["window_size"] = 15
    else:
        config["window_size"] = 10
    
    return config


def clean_market_data(market_data: MarketData, 
                     std_threshold: float = 3.0,
                     window_size: int = 20,
                     method: str = "interpolate",
                     filter_type: str = None,
                     filter_params: Dict = None,
                     use_case: str = None,
                     handle_missing: bool = True,
                     max_gap_minutes: int = 5,
                     missing_method: str = "linear",
                     columns: List[str] = None,
                     corruption_threshold: float = 0.1) -> MarketData:
    """
    Nettoie les données de marché en détectant et corrigeant les valeurs aberrantes.
    
    Args:
        market_data: Données de marché à nettoyer
        std_threshold: Nombre d'écarts-types pour définir le seuil
        window_size: Taille de la fenêtre pour calculer la volatilité locale
        method: Méthode de nettoyage ("remove", "interpolate", "clip", "median_window", "kalman", "savgol")
        filter_type: Type de filtre à appliquer après nettoyage ("kalman", "savgol", None)
        filter_params: Paramètres du filtre si applicable
        use_case: Cas d'utilisation pour auto-sélection du filtre
        handle_missing: Si True, détecte et gère les valeurs manquantes
        max_gap_minutes: Écart maximum en minutes pour interpoler les valeurs manquantes
        missing_method: Méthode d'interpolation pour les valeurs manquantes
        columns: Colonnes à nettoyer (par défaut: ['open', 'high', 'low', 'close', 'volume'])
        corruption_threshold: Seuil de corruption (pourcentage de valeurs manquantes) au-delà duquel 
                             une période est considérée comme corrompue (défaut: 0.1 = 10%)
            
    Returns:
        Données de marché nettoyées
    """
    # Si un use_case est spécifié, sélectionner automatiquement la configuration
    if use_case:
        config = auto_select_filter(market_data, use_case)
        std_threshold = config["std_threshold"]
        window_size = config["window_size"]
        method = config["method"]
        filter_type = config["filter_type"]
        filter_params = config["filter_params"]
    
    # Nettoyer les données
    cleaner = DataCleaner()
    return cleaner.clean_market_data(
        market_data, 
        std_threshold=std_threshold,
        window_size=window_size,
        method=CleaningMethod(method),
        filter_type=filter_type,
        filter_params=filter_params,
        handle_missing=handle_missing,
        max_gap_minutes=max_gap_minutes,
        missing_method=missing_method,
        columns=columns,
        corruption_threshold=corruption_threshold
    )
