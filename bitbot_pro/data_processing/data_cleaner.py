"""
Module de nettoyage et préparation des données pour BitBot Pro.
Implémente des routines avancées pour détecter et corriger les valeurs aberrantes
ainsi que pour normaliser les données avant leur utilisation par les modèles.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Callable
import logging
from scipy import stats
from scipy import signal
import time
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from bitbot_pro.utils.logger import logger
from bitbot_pro.utils.performance import timeit

# Constantes pour la gestion des valeurs manquantes
MAX_SMALL_GAP_MINUTES = 5  # Durée maximale pour considérer un gap comme "petit"
CORRUPTION_THRESHOLD = 0.1  # Seuil à partir duquel une plage est considérée comme corrompue (10%)

class DataCleaner:
    """
    Classe principale pour le nettoyage et la préparation des données.
    Implémente diverses méthodes pour détecter et corriger les valeurs aberrantes,
    appliquer des filtres de lissage, et normaliser les données.
    """
    
    def __init__(self, 
                 volatility_window: int = 20, 
                 outlier_threshold: float = 3.0,
                 batch_size: int = 50,
                 max_workers: int = 4):
        """
        Initialise le nettoyeur de données avec les paramètres spécifiés.
        
        Args:
            volatility_window: Taille de la fenêtre pour calculer la volatilité (défaut: 20 périodes)
            outlier_threshold: Seuil de détection des valeurs aberrantes en écarts-types (défaut: 3.0)
            batch_size: Taille des lots pour le traitement par lots (défaut: 50)
            max_workers: Nombre maximum de workers pour le traitement parallèle (défaut: 4)
        """
        self.volatility_window = volatility_window
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
        # Paramètres du filtre de Kalman
        self.kalman_params = {
            'process_variance': 1e-5,  # Variance du processus (ajuster selon la volatilité)
            'measurement_variance': 1e-1,  # Variance de mesure (ajuster selon le bruit)
            'state_transition': 1.0  # Modèle de transition d'état simple (constant)
        }
        
        logger.info(f"DataCleaner initialisé avec: fenêtre de volatilité={volatility_window}, " +
                   f"seuil d'aberration={outlier_threshold}, taille de lot={batch_size}")
    
    @timeit
    def detect_outliers(self, 
                        data: pd.DataFrame, 
                        column: str = 'close', 
                        method: str = 'zscore') -> pd.Series:
        """
        Détecte les valeurs aberrantes dans une série temporelle.
        
        Args:
            data: DataFrame contenant les données à analyser
            column: Nom de la colonne à vérifier (défaut: 'close')
            method: Méthode de détection ('zscore', 'mad', 'iqr') (défaut: 'zscore')
            
        Returns:
            Une série booléenne indiquant les positions des valeurs aberrantes (True = aberrante)
        """
        if data.empty:
            logger.warning("Données vides fournies pour la détection d'aberrations")
            return pd.Series([], dtype=bool)
        
        # Pré-validation des données pour éviter les erreurs
        if column not in data.columns:
            logger.error(f"Colonne {column} non trouvée dans les données")
            return pd.Series([False] * len(data), index=data.index)
        
        # Supprimer les valeurs NaN pour éviter les erreurs dans les calculs
        series = data[column].copy()
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Si trop peu de données après nettoyage, retourner aucune aberration
        if len(series) < 3:
            logger.warning("Trop peu de données valides pour la détection d'aberrations")
            return pd.Series([False] * len(data), index=data.index)
        
        outliers = pd.Series(False, index=data.index)
        
        # Calculer la volatilité dynamique sur une fenêtre glissante
        rolling_std = series.rolling(window=self.volatility_window, 
                                    min_periods=3).std()
        
        if method == 'zscore':
            # Méthode Z-score basée sur la volatilité locale
            rolling_mean = series.rolling(window=self.volatility_window, 
                                         min_periods=3).mean()
            
            # Calculer les z-scores (nombre d'écarts-types par rapport à la moyenne)
            z_scores = pd.Series(index=series.index)
            
            # Utiliser une boucle de fenêtre pour un calcul plus précis
            for i in range(self.volatility_window, len(series)):
                window_indices = series.index[i-self.volatility_window:i]
                if rolling_std[series.index[i]] > 0:  # Éviter la division par zéro
                    z_scores[series.index[i]] = abs(
                        series[series.index[i]] - rolling_mean[series.index[i]]
                    ) / rolling_std[series.index[i]]
                else:
                    z_scores[series.index[i]] = 0
            
            # Marquer les valeurs comme aberrantes si elles dépassent le seuil
            outliers.loc[z_scores > self.outlier_threshold] = True
            
        elif method == 'mad':
            # Méthode basée sur la déviation absolue médiane (MAD)
            # Plus robuste que le Z-score standard face aux valeurs extrêmes
            for i in range(self.volatility_window, len(series)):
                window = series.iloc[i-self.volatility_window:i]
                median = window.median()
                mad = np.median(np.abs(window - median))
                
                # MAD normalisé (comparable à un écart-type)
                if mad > 0:  # Éviter la division par zéro
                    mad_score = abs(series.iloc[i] - median) / (mad * 1.4826)  # Facteur de normalisation
                    if mad_score > self.outlier_threshold:
                        outliers.iloc[i] = True
                
        elif method == 'iqr':
            # Méthode basée sur l'écart interquartile (IQR)
            for i in range(self.volatility_window, len(series)):
                window = series.iloc[i-self.volatility_window:i]
                q1 = window.quantile(0.25)
                q3 = window.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - (self.outlier_threshold * iqr)
                upper_bound = q3 + (self.outlier_threshold * iqr)
                
                if series.iloc[i] < lower_bound or series.iloc[i] > upper_bound:
                    outliers.iloc[i] = True
        else:
            logger.error(f"Méthode de détection d'aberrations inconnue: {method}")
        
        logger.debug(f"Détection d'aberrations terminée. {outliers.sum()} aberrations détectées sur {len(data)} points.")
        return outliers
        
    @timeit
    def apply_kalman_filter(self, 
                           data: pd.DataFrame, 
                           column: str = 'close',
                           process_variance: Optional[float] = None,
                           measurement_variance: Optional[float] = None) -> pd.Series:
        """
        Applique un filtre de Kalman pour lisser une série temporelle.
        Le filtre de Kalman est particulièrement efficace pour réduire le bruit
        tout en préservant les tendances significatives.
        
        Args:
            data: DataFrame contenant les données à filtrer
            column: Nom de la colonne à filtrer (défaut: 'close')
            process_variance: Variance du processus (Q) (défaut: valeur de self.kalman_params)
            measurement_variance: Variance de mesure (R) (défaut: valeur de self.kalman_params)
            
        Returns:
            Une série contenant les valeurs filtrées
        """
        if data.empty:
            logger.warning("Données vides fournies pour le filtre de Kalman")
            return pd.Series([], dtype=float)
        
        # Pré-validation et pré-conversion des données
        if column not in data.columns:
            logger.error(f"Colonne {column} non trouvée dans les données")
            return pd.Series(data=np.nan, index=data.index)
        
        # Utiliser les paramètres par défaut si non spécifiés
        process_var = process_variance if process_variance is not None else self.kalman_params['process_variance']
        measurement_var = measurement_variance if measurement_variance is not None else self.kalman_params['measurement_variance']
        
        # Création du filtre de Kalman
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[self.kalman_params['state_transition']]])  # Matrice de transition d'état
        kf.H = np.array([[1.0]])  # Matrice d'observation
        kf.P = np.array([[1.0]])  # Covariance d'estimation initiale
        kf.R = np.array([[measurement_var]])  # Variance de mesure
        
        # Définition manuelle de la matrice Q (bruit de processus) car dim=1 n'est pas supporté par Q_discrete_white_noise
        kf.Q = np.array([[process_var]])
        
        # Extraction de la série à filtrer
        series = data[column].replace([np.inf, -np.inf], np.nan)
        
        # Initialisation de la série de résultats
        filtered_values = np.zeros(len(series))
        filtered_values[:] = np.nan
        
        # Application du filtre de Kalman par lots pour optimiser les performances
        valid_indices = np.where(~np.isnan(series.values))[0]
        
        if len(valid_indices) == 0:
            logger.warning("Aucune donnée valide pour l'application du filtre de Kalman")
            return pd.Series(filtered_values, index=data.index)
        
        # Initialisation de l'état avec la première valeur non-NaN
        kf.x = np.array([[series.iloc[valid_indices[0]]]])
        
        # Traitement par lots pour améliorer les performances
        for i in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[i:i + self.batch_size]
            
            for idx in batch_indices:
                # Prédiction
                kf.predict()
                
                # Mise à jour avec la mesure actuelle
                kf.update(np.array([[series.iloc[idx]]]))
                
                # Stockage de l'estimation
                filtered_values[idx] = kf.x[0, 0]
        
        logger.debug(f"Filtrage de Kalman terminé pour {len(valid_indices)} points de données valides")
        return pd.Series(filtered_values, index=data.index)
    
    @timeit
    def apply_smoothing_filter(self, 
                              data: pd.DataFrame, 
                              column: str = 'close',
                              filter_type: str = 'kalman',
                              window_size: int = 5,
                              **filter_params) -> pd.Series:
        """
        Applique un filtre de lissage à une série temporelle.
        
        Args:
            data: DataFrame contenant les données à filtrer
            column: Nom de la colonne à filtrer (défaut: 'close')
            filter_type: Type de filtre ('kalman', 'savgol', 'ewm', 'ma')
            window_size: Taille de la fenêtre pour les filtres à fenêtre (défaut: 5)
            **filter_params: Paramètres supplémentaires spécifiques au filtre
            
        Returns:
            Une série contenant les valeurs filtrées
        """
        if data.empty:
            logger.warning("Données vides fournies pour le lissage")
            return pd.Series([], dtype=float)
        
        if column not in data.columns:
            logger.error(f"Colonne {column} non trouvée dans les données")
            return pd.Series(data=np.nan, index=data.index)
        
        # Extraction de la série à filtrer avec pré-validation
        series = data[column].replace([np.inf, -np.inf], np.nan).copy()
        
        if filter_type == 'kalman':
            # Filtre de Kalman (optimal pour réduire le bruit aléatoire)
            return self.apply_kalman_filter(data, column, 
                                           process_variance=filter_params.get('process_variance'),
                                           measurement_variance=filter_params.get('measurement_variance'))
            
        elif filter_type == 'savgol':
            # Filtre de Savitzky-Golay (bon pour préserver les tendances locales)
            polyorder = filter_params.get('polyorder', 2)
            if polyorder >= window_size:
                polyorder = window_size - 1
                if polyorder < 1:
                    polyorder = 1
                logger.warning(f"L'ordre du polynôme ajusté à {polyorder} pour être compatible avec la taille de fenêtre {window_size}")
            
            # Application du filtre uniquement sur les données valides
            valid_mask = ~np.isnan(series.values)
            result = series.copy()
            
            # S'il y a suffisamment de données valides
            if np.sum(valid_mask) > window_size:
                valid_values = series.values[valid_mask]
                filtered_valid = signal.savgol_filter(
                    valid_values, 
                    window_size, 
                    polyorder,
                    mode='nearest'
                )
                result.values[valid_mask] = filtered_valid
            
            return result
            
        elif filter_type == 'ewm':
            # Moyenne mobile exponentielle (réactivité ajustable)
            alpha = filter_params.get('alpha', 0.2)
            return series.ewm(alpha=alpha, min_periods=1).mean()
            
        elif filter_type == 'ma':
            # Moyenne mobile simple (lissage uniforme)
            return series.rolling(window=window_size, min_periods=1).mean()
            
        else:
            logger.error(f"Type de filtre inconnu: {filter_type}")
            return series
    
    @timeit
    def handle_missing_values(self, 
                            data: pd.DataFrame, 
                            timeframe: str = '1m',
                            columns: List[str] = None,
                            max_gap_minutes: int = MAX_SMALL_GAP_MINUTES,
                            corruption_threshold: float = CORRUPTION_THRESHOLD) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans un DataFrame temporel.
        
        Stratégie :
        1. Interpolation linéaire pour les petits gaps (<= max_gap_minutes)
        2. Suppression des plages de données trop corrompues (> corruption_threshold)
        
        Args:
            data: DataFrame contenant les données avec index temporel
            timeframe: Intervalle de temps des données ('1m', '5m', '1h', etc.)
            columns: Liste des colonnes à traiter (si None, toutes les colonnes numériques)
            max_gap_minutes: Durée maximale en minutes pour l'interpolation (défaut: constante MAX_SMALL_GAP_MINUTES)
            corruption_threshold: Seuil de corruption pour la suppression (défaut: constante CORRUPTION_THRESHOLD)
            
        Returns:
            DataFrame nettoyé avec les valeurs manquantes traitées
        """
        if data.empty:
            logger.warning("Données vides fournies pour le traitement des valeurs manquantes")
            return data.copy()
        
        # Vérifier que l'index est bien temporel
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("L'index du DataFrame doit être de type DatetimeIndex pour gérer les valeurs manquantes")
            return data.copy()
        
        # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
        
        if not columns:
            logger.warning("Aucune colonne numérique trouvée pour le traitement des valeurs manquantes")
            return data.copy()
        
        # Convertir le timeframe en minutes pour les calculs
        tf_minutes = self._timeframe_to_minutes(timeframe)
        if tf_minutes is None:
            logger.error(f"Timeframe non reconnu: {timeframe}")
            return data.copy()
        
        # Calculer la durée maximale acceptable pour l'interpolation
        max_gap_steps = max(1, int(max_gap_minutes / tf_minutes))
        
        # Un gap d'exactement 1 pas de temps (5 min pour tf de 5 min) devrait être considéré comme petit
        # Assurons-nous que cette valeur est suffisante pour capturer les petits gaps
        if max_gap_steps == 1 and max_gap_minutes >= tf_minutes:
            max_gap_steps = 1  # Explicite pour clarté
            logger.debug(f"Configuration pour interpoler les gaps d'exactement {tf_minutes} minutes")
        
        # Créer une copie du DataFrame pour éviter de modifier l'original
        result = data.copy()
        
        # Convertir timeframe en format de fréquence pandas
        pandas_freq = self._convert_timeframe_to_freq(timeframe)
        
        # Détecter les index manquants dans la série temporelle
        full_idx = pd.date_range(start=data.index.min(), end=data.index.max(), freq=pandas_freq)
        missing_idx = full_idx.difference(data.index)
        
        if len(missing_idx) == 0:
            logger.info("Aucune valeur manquante détectée dans la série temporelle")
            return result
        
        logger.info(f"Détection de {len(missing_idx)} points temporels manquants sur {len(full_idx)} ({len(missing_idx)/len(full_idx)*100:.2f}%)")
        
        # Si plus de X% des données sont manquantes, avertir l'utilisateur
        overall_missing_ratio = len(missing_idx) / len(full_idx)
        if overall_missing_ratio > corruption_threshold:
            logger.warning(f"Attention: {overall_missing_ratio*100:.2f}% des données sont manquantes (>{corruption_threshold*100}%)")
        
        # 1. Identifier les segments de données manquantes consécutives
        gaps = self._identify_temporal_gaps(data.index, full_idx)
        
        # Compteurs pour le rapport
        interpolated_points = 0
        
        # 2. Traiter chaque segment selon sa taille
        for gap_start, gap_end, gap_size in gaps:
            # Petits gaps : interpolation linéaire
            if gap_size <= max_gap_steps:
                # Créer un DataFrame temporaire avec les points manquants
                missing_range = pd.date_range(start=gap_start, end=gap_end, freq=pandas_freq)[1:-1]
                if len(missing_range) > 0:
                    logger.debug(f"Interpolation d'un gap de {gap_size} points ({gap_size * tf_minutes} minutes)")
                    missing_df = pd.DataFrame(index=missing_range, columns=result.columns)
                    # Fusionner avec les données existantes
                    temp_df = pd.concat([result, missing_df]).sort_index()
                    # Interpolation linéaire sur les colonnes numériques
                    for col in columns:
                        temp_df.loc[missing_range, col] = temp_df[col].interpolate(method='linear', limit=gap_size)
                    # Mettre à jour le résultat sans les NaN
                    mask = temp_df.index.isin(missing_range)
                    interpolated_df = temp_df.loc[mask, columns].dropna(how='all')
                    result = pd.concat([result, interpolated_df])
                    result = result.sort_index()
                    interpolated_points += len(interpolated_df)
                    logger.debug(f"Interpolation linéaire effectuée pour {len(interpolated_df)} points manquants")
            
            # Grands gaps : identification des plages corrompues
            else:
                missing_range = pd.date_range(start=gap_start, end=gap_end, freq=pandas_freq)[1:-1]
                if len(missing_range) > 0:
                    logger.info(f"Gap important détecté: {len(missing_range)} points manquants consécutifs " +
                              f"de {missing_range[0]} à {missing_range[-1]}")
        
        if interpolated_points > 0:
            logger.info(f"Total de {interpolated_points} points interpolés")
        
        # 3. Vérifier les plages de données corrompues à supprimer
        corrupted_ranges = self._identify_corrupted_ranges(result, columns, corruption_threshold)
        
        if corrupted_ranges:
            for range_start, range_end in corrupted_ranges:
                logger.warning(f"Suppression d'une plage de données corrompue: {range_start} à {range_end}")
                mask = (result.index < range_start) | (result.index > range_end)
                result = result.loc[mask].copy()
        
        return result
    
    def _timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        """
        Convertit un timeframe en nombre de minutes.
        
        Args:
            timeframe: Format du timeframe ('1m', '5m', '1h', '4h', '1d', etc.)
            
        Returns:
            Nombre de minutes ou None si format non reconnu
        """
        if not timeframe:
            return None
        
        # Normaliser le timeframe
        tf = timeframe.lower().strip()
        
        # Extraire le nombre et l'unité
        import re
        match = re.match(r'(\d+)([mhdw])', tf)
        if not match:
            return None
        
        value, unit = int(match.group(1)), match.group(2)
        
        # Convertir en minutes
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        elif unit == 'w':
            return value * 60 * 24 * 7
        else:
            return None
    
    def _convert_timeframe_to_freq(self, timeframe: str) -> str:
        """
        Convertit le format de timeframe de trading en format de fréquence pandas.
        
        Args:
            timeframe: Format de timeframe (ex: '1m', '5m', '1h', '1d', etc.)
            
        Returns:
            Format de fréquence pandas (ex: 'min', '5min', 'H', 'D', etc.)
        """
        # Si c'est déjà un format pandas, le retourner tel quel
        pandas_freqs = ['S', 'T', 'min', 'H', 'D', 'W', 'M', 'Q', 'A', 'Y']
        if any(timeframe.endswith(freq) for freq in pandas_freqs):
            return timeframe
        
        # Sinon, convertir
        value = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 1
        unit = timeframe[-1].lower()
        
        if unit == 'm':
            return f"{value}min"
        elif unit == 'h':
            return f"{value}H"
        elif unit == 'd':
            return f"{value}D"
        elif unit == 'w':
            return f"{value}W"
        else:
            logger.warning(f"Format de timeframe non reconnu: {timeframe}, utilisation par défaut")
            return timeframe
    
    def _identify_temporal_gaps(self, 
                               existing_idx: pd.DatetimeIndex, 
                               full_idx: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        """
        Identifie les segments de données manquantes consécutives.
        
        Args:
            existing_idx: Index temporel des données existantes
            full_idx: Index temporel complet attendu
            
        Returns:
            Liste de tuples (début du gap, fin du gap, taille du gap)
        """
        if len(existing_idx) == 0 or len(full_idx) == 0:
            return []
        
        # Obtenir les points manquants
        missing_idx = full_idx.difference(existing_idx).sort_values()
        if len(missing_idx) == 0:
            return []
        
        # Identifier les segments consécutifs
        gaps = []
        gap_start = None
        prev_ts = None
        expected_diff = full_idx[1] - full_idx[0]  # Diff attendue entre deux points consécutifs
        
        # Ajouter les points existants au début et à la fin pour faciliter l'identification des segments
        all_idx = existing_idx.union(missing_idx).sort_values()
        
        for i, ts in enumerate(all_idx):
            if ts in missing_idx:
                # Début d'un nouveau gap
                if gap_start is None:
                    # Trouver le dernier point existant avant le gap
                    prev_existing = ts - expected_diff if i == 0 else all_idx[i-1]
                    gap_start = prev_existing
                
                # Le gap continue
                prev_ts = ts
            else:
                # Fin d'un gap
                if gap_start is not None and prev_ts is not None:
                    # Calculer la taille du gap en nombre de points
                    gap_size = len(full_idx[full_idx.slice_indexer(gap_start, prev_ts)]) - 1
                    if gap_size > 0:  # S'assurer que le gap est réellement présent
                        gaps.append((gap_start, ts, gap_size))
                    gap_start = None
                    prev_ts = None
        
        # Gérer le cas où le gap se termine à la fin des données
        if gap_start is not None and prev_ts is not None:
            next_existing = prev_ts + expected_diff
            gap_size = len(full_idx[full_idx.slice_indexer(gap_start, prev_ts)]) - 1
            if gap_size > 0:  # S'assurer que le gap est réellement présent
                gaps.append((gap_start, next_existing, gap_size))
        
        return gaps
    
    def _identify_corrupted_ranges(self, 
                                 data: pd.DataFrame, 
                                 columns: List[str],
                                 corruption_threshold: float) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Identifie les plages de données trop corrompues pour être utilisables.
        
        Args:
            data: DataFrame contenant les données
            columns: Colonnes à vérifier
            corruption_threshold: Seuil à partir duquel une plage est considérée comme corrompue
            
        Returns:
            Liste de tuples (début de plage, fin de plage) à supprimer
        """
        if data.empty or not columns:
            return []
        
        # Créer un masque pour les valeurs manquantes dans toutes les colonnes spécifiées
        missing_mask = data[columns].isna().any(axis=1)
        
        # Si pas de valeurs manquantes, retourner une liste vide
        if not missing_mask.any():
            return []
        
        # Utiliser une fenêtre glissante pour évaluer la corruption locale
        window_size = max(20, int(len(data) * 0.05))  # 5% des données ou au moins 20 points
        
        corrupted_ranges = []
        i = 0
        
        while i < len(data):
            # Définir la fenêtre courante
            end_idx = min(i + window_size, len(data))
            window = data.iloc[i:end_idx]
            
            # Calculer le taux de corruption dans la fenêtre
            window_corruption = missing_mask.iloc[i:end_idx].mean()
            
            # Si la corruption dépasse le seuil, marquer la plage comme corrompue
            if window_corruption > corruption_threshold:
                range_start = window.index[0]
                range_end = window.index[-1]
                corrupted_ranges.append((range_start, range_end))
                
                # Passer à la fenêtre suivante après cette plage
                i = end_idx
            else:
                # Avancer d'un point
                i += 1
        
        # Fusionner les plages adjacentes ou qui se chevauchent
        if len(corrupted_ranges) > 1:
            merged_ranges = [corrupted_ranges[0]]
            for current_start, current_end in corrupted_ranges[1:]:
                prev_start, prev_end = merged_ranges[-1]
                
                # Si les plages se chevauchent ou sont adjacentes
                if current_start <= prev_end or (current_start - prev_end).total_seconds() < 300:  # 5 minutes
                    merged_ranges[-1] = (prev_start, max(prev_end, current_end))
                else:
                    merged_ranges.append((current_start, current_end))
            
            return merged_ranges
        
        return corrupted_ranges
    
    @timeit
    def correct_outliers(self, 
                        data: pd.DataFrame, 
                        column: str = 'close',
                        detection_method: str = 'zscore',
                        correction_method: str = 'kalman',
                        **params) -> pd.DataFrame:
        """
        Détecte et corrige les valeurs aberrantes dans un DataFrame.
        
        Args:
            data: DataFrame contenant les données à corriger
            column: Nom de la colonne à corriger (défaut: 'close')
            detection_method: Méthode de détection des aberrations ('zscore', 'mad', 'iqr')
            correction_method: Méthode de correction ('kalman', 'interpolation', 'winsorize', 'median')
            **params: Paramètres supplémentaires pour les méthodes de détection et correction
            
        Returns:
            DataFrame avec les valeurs aberrantes corrigées
        """
        # Créer une copie pour éviter de modifier les données d'entrée
        result = data.copy()
        
        # Vérifier si la colonne existe
        if column not in result.columns:
            logger.error(f"Colonne {column} non trouvée dans les données")
            return result
        
        # Détecter les valeurs aberrantes
        outliers = self.detect_outliers(result, column, method=detection_method)
        
        # Si aucune valeur aberrante n'est détectée, retourner les données originales
        if not outliers.any():
            logger.debug("Aucune valeur aberrante détectée. Aucune correction nécessaire.")
            return result
        
        # Extraire les indices des valeurs aberrantes
        outlier_indices = outliers[outliers].index
        
        # Appliquer la méthode de correction
        if correction_method == 'kalman':
            # Appliquer le filtre de Kalman à l'ensemble des données
            filtered_series = self.apply_kalman_filter(result, column)
            
            # Remplacer uniquement les valeurs aberrantes par leurs équivalents filtrés
            result.loc[outlier_indices, column] = filtered_series.loc[outlier_indices]
            
        elif correction_method == 'interpolation':
            # Marquer temporairement les valeurs aberrantes comme NaN
            temp_series = result[column].copy()
            temp_series.loc[outlier_indices] = np.nan
            
            # Interpolation linéaire des valeurs manquantes
            interpolated = temp_series.interpolate(method='linear', limit_direction='both')
            
            # Remplacer les valeurs aberrantes par les valeurs interpolées
            result.loc[outlier_indices, column] = interpolated.loc[outlier_indices]
            
        elif correction_method == 'winsorize':
            # Winsorisation: remplacer les valeurs extrêmes par des valeurs aux percentiles
            lower_percentile = params.get('lower_percentile', 0.05)
            upper_percentile = params.get('upper_percentile', 0.95)
            
            # Calculer les seuils sur les données non aberrantes
            non_outlier_values = result.loc[~outliers, column]
            lower_bound = non_outlier_values.quantile(lower_percentile)
            upper_bound = non_outlier_values.quantile(upper_percentile)
            
            # Remplacer les valeurs aberrantes par les seuils
            for idx in outlier_indices:
                if result.loc[idx, column] < lower_bound:
                    result.loc[idx, column] = lower_bound
                elif result.loc[idx, column] > upper_bound:
                    result.loc[idx, column] = upper_bound
                    
        elif correction_method == 'median':
            # Remplacer par la médiane locale
            window_size = params.get('window_size', self.volatility_window)
            
            for idx in outlier_indices:
                # Trouver l'indice de position dans le DataFrame
                pos = result.index.get_loc(idx)
                
                # Calculer les limites de la fenêtre
                start = max(0, pos - window_size // 2)
                end = min(len(result), pos + window_size // 2 + 1)
                
                # Extraire les valeurs non aberrantes de la fenêtre
                window_values = result.iloc[start:end][column].copy()
                window_values = window_values[~outliers.iloc[start:end]]
                
                if not window_values.empty:
                    # Remplacer par la médiane des valeurs non aberrantes
                    result.loc[idx, column] = window_values.median()
        
        logger.info(f"{len(outlier_indices)} valeurs aberrantes corrigées avec la méthode {correction_method}")
        return result

    @timeit
    def process_batch(self, 
                     data_batch: pd.DataFrame,
                     columns: List[str] = None,
                     detection_method: str = 'zscore',
                     correction_method: str = 'kalman',
                     **params) -> pd.DataFrame:
        """
        Traite un lot de données en appliquant la détection et correction des valeurs aberrantes
        pour toutes les colonnes spécifiées.
        
        Args:
            data_batch: Lot de données à traiter
            columns: Liste des colonnes à traiter (si None, traite 'open', 'high', 'low', 'close')
            detection_method: Méthode de détection des aberrations
            correction_method: Méthode de correction
            **params: Paramètres supplémentaires
            
        Returns:
            DataFrame avec les valeurs aberrantes corrigées pour toutes les colonnes spécifiées
        """
        # Vérifier si le lot est vide
        if data_batch.empty:
            logger.warning("Lot de données vide. Aucun traitement effectué.")
            return data_batch
        
        # Colonnes par défaut pour les données OHLC
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            # Filtrer les colonnes qui existent dans le DataFrame
            columns = [col for col in columns if col in data_batch.columns]
            
            if not columns:
                logger.warning("Aucune colonne de prix trouvée dans les données. Aucun traitement effectué.")
                return data_batch
        
        # Créer une copie pour le résultat final
        result = data_batch.copy()
        
        # Traiter chaque colonne
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(columns))) as executor:
            # Créer les tâches pour le traitement parallèle
            futures = {
                executor.submit(
                    self.correct_outliers, 
                    result, 
                    column, 
                    detection_method, 
                    correction_method, 
                    **params
                ): column for column in columns
            }
            
            # Collecter les résultats à mesure qu'ils sont disponibles
            for future in futures:
                column = futures[future]
                try:
                    # Récupérer le DataFrame avec la colonne corrigée
                    cleaned = future.result()
                    # Mettre à jour uniquement la colonne traitée
                    result[column] = cleaned[column]
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la colonne {column}: {str(e)}")
        
        return result
    
    @timeit
    def process_data(self,
                    data: pd.DataFrame,
                    columns: List[str] = None,
                    detection_method: str = 'zscore',
                    correction_method: str = 'kalman',
                    **params) -> pd.DataFrame:
        """
        Traite toutes les données en les divisant en lots pour optimiser les performances.
        
        Args:
            data: DataFrame contenant toutes les données à traiter
            columns: Liste des colonnes à traiter
            detection_method: Méthode de détection des aberrations
            correction_method: Méthode de correction
            **params: Paramètres supplémentaires
            
        Returns:
            DataFrame avec les valeurs aberrantes corrigées
        """
        if data.empty:
            logger.warning("Aucune donnée à traiter.")
            return data
        
        # Créer une copie pour éviter de modifier les données d'entrée
        result = data.copy()
        
        # Traiter les données par lots pour optimiser les performances
        for i in range(0, len(data), self.batch_size):
            # Extraire un lot
            batch = result.iloc[i:i + self.batch_size].copy()
            
            # Traiter le lot
            cleaned_batch = self.process_batch(
                batch,
                columns=columns,
                detection_method=detection_method,
                correction_method=correction_method,
                **params
            )
            
            # Mettre à jour les résultats
            result.iloc[i:i + self.batch_size] = cleaned_batch
            
            # Petite pause entre les lots pour éviter de bloquer les threads
            if i + self.batch_size < len(data):
                time.sleep(0.001)
        
        return result

    @timeit
    def standardize_timestamps(self, 
                           data_frames: Dict[str, pd.DataFrame], 
                           target_timeframe: str = '1m',
                           align_to_intervals: bool = True,
                           ensure_utc: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Uniformise les timestamps dans plusieurs DataFrames pour assurer leur synchronisation.
        
        Fonctionnalités:
        1. Convertit tous les timestamps au fuseau horaire UTC si nécessaire
        2. Aligne les timestamps sur des intervalles réguliers (ex: minutes exactes)
        3. Assure que tous les DataFrames partagent la même résolution temporelle
        
        Args:
            data_frames: Dictionnaire de DataFrames indexés par nom/source
            target_timeframe: Timeframe cible pour l'uniformisation ('1m', '5m', etc.)
            align_to_intervals: Si True, aligne les timestamps sur des intervalles réguliers
            ensure_utc: Si True, convertit tous les timestamps en UTC
            
        Returns:
            Dictionnaire de DataFrames avec timestamps uniformisés
        """
        if not data_frames:
            logger.warning("Aucun DataFrame fourni pour l'uniformisation des timestamps")
            return {}
        
        # Convertir le timeframe en format de fréquence pandas
        pandas_freq = self._convert_timeframe_to_freq(target_timeframe)
        
        # Résultat final
        result = {}
        
        # 1. Convertir en UTC si nécessaire
        if ensure_utc:
            for source, df in data_frames.items():
                if df.empty:
                    result[source] = df.copy()
                    continue
                
                # Vérifier si l'index est un DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.error(f"Source {source}: L'index n'est pas un DatetimeIndex")
                    result[source] = df.copy()
                    continue
                
                # Vérifier et convertir le fuseau horaire
                if df.index.tz is None:
                    logger.info(f"Source {source}: Timestamps sans fuseau horaire, conversion en UTC")
                    temp_df = df.copy()
                    temp_df.index = temp_df.index.tz_localize('UTC')
                    result[source] = temp_df
                else:
                    # Vérifier si le fuseau horaire est UTC
                    is_utc = False
                    try:
                        if hasattr(df.index.tz, 'zone'):
                            is_utc = df.index.tz.zone == 'UTC'
                        elif hasattr(df.index.tz, 'tzname'):
                            tz_name = df.index.tz.tzname(None)
                            is_utc = tz_name == 'UTC' or tz_name == 'GMT'
                        else:
                            tz_str = str(df.index.tz)
                            is_utc = 'UTC' in tz_str or 'GMT' in tz_str
                    except Exception:
                        is_utc = False
                    
                    if not is_utc:
                        logger.info(f"Source {source}: Conversion du fuseau horaire en UTC")
                        temp_df = df.copy()
                        temp_df.index = temp_df.index.tz_convert('UTC')
                        result[source] = temp_df
                    else:
                        # Déjà en UTC
                        result[source] = df.copy()
        else:
            # Pas de conversion de fuseau horaire
            for source, df in data_frames.items():
                result[source] = df.copy()
        
        # 2. Aligner sur des intervalles réguliers si demandé
        if align_to_intervals:
            for source, df in result.items():
                if df.empty:
                    continue
                
                # Vérifier si l'index est un DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    continue
                
                # Trouver la plage temporelle globale
                min_time = df.index.min()
                max_time = df.index.max()
                
                # Créer un index aligné sur des intervalles réguliers
                aligned_idx = pd.date_range(
                    start=min_time.floor(pandas_freq), 
                    end=max_time.ceil(pandas_freq), 
                    freq=pandas_freq
                )
                
                # Réindexer et interpoler si nécessaire
                if len(aligned_idx) > 0:
                    # Créer un nouveau DataFrame avec l'index aligné
                    aligned_df = pd.DataFrame(index=aligned_idx)
                    
                    # Fusionner avec les données originales
                    merged_df = pd.merge_asof(
                        aligned_df,
                        df,
                        left_index=True,
                        right_index=True,
                        direction='nearest',
                        tolerance=pd.Timedelta(target_timeframe)
                    )
                    
                    result[source] = merged_df
                    logger.info(f"Source {source}: Timestamps alignés sur des intervalles de {target_timeframe}")
        
        # 3. Vérifier la cohérence des index entre les sources
        if len(result) > 1:
            # Collecter tous les timestamps uniques
            all_timestamps = set()
            for df in result.values():
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    all_timestamps.update(df.index)
            
            # Calculer l'intersection des timestamps
            common_timestamps = None
            for df in result.values():
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    if common_timestamps is None:
                        common_timestamps = set(df.index)
                    else:
                        common_timestamps &= set(df.index)
            
            if common_timestamps:
                coverage = len(common_timestamps) / len(all_timestamps) * 100
                logger.info(f"Couverture temporelle commune: {coverage:.2f}% ({len(common_timestamps)}/{len(all_timestamps)} timestamps)")
        
        return result
