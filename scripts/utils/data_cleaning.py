import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def clean_kline_data(df: pd.DataFrame, 
                    remove_outliers: bool = True,
                    fill_missing: bool = True,
                    max_std_dev: float = 3.0,
                    min_volume: float = 0.0) -> pd.DataFrame:
    """
    Nettoie les données de bougies (klines)
    
    Args:
        df: DataFrame contenant les données de bougies
        remove_outliers: Si True, supprime les valeurs aberrantes
        fill_missing: Si True, interpole les valeurs manquantes
        max_std_dev: Nombre d'écarts-types pour la détection des valeurs aberrantes
        min_volume: Volume minimum acceptable
    
    Returns:
        DataFrame nettoyé
    """
    try:
        # Copie pour ne pas modifier l'original
        df = df.copy()
        
        # Vérifier les colonnes requises
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Colonnes manquantes. Requis: {required_columns}")
        
        # Supprimer les doublons
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Trier par timestamp
        df = df.sort_values('timestamp')
        
        # Vérifier la cohérence des données OHLC
        df = validate_ohlc(df)
        
        # Supprimer les volumes nuls ou trop faibles
        if min_volume > 0:
            df = df[df['volume'] >= min_volume]
        
        # Supprimer les valeurs aberrantes
        if remove_outliers:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df = remove_outliers_zscore(df, col, max_std_dev)
        
        # Gérer les valeurs manquantes
        if fill_missing:
            df = handle_missing_values(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        raise

def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide et corrige les incohérences dans les données OHLC
    """
    # Copie pour ne pas modifier l'original
    df = df.copy()
    
    # Vérifier que high >= low
    invalid_hl = df['high'] < df['low']
    if invalid_hl.any():
        logger.warning(f"Correction de {invalid_hl.sum()} lignes où high < low")
        df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
    
    # Vérifier que high >= open et high >= close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    
    # Vérifier que low <= open et low <= close
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df

def remove_outliers_zscore(df: pd.DataFrame, 
                          column: str, 
                          max_std_dev: float = 3.0) -> pd.DataFrame:
    """
    Supprime les valeurs aberrantes en utilisant le score Z
    """
    # Calculer le score Z
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    
    # Marquer les outliers
    outliers = z_scores > max_std_dev
    if outliers.any():
        logger.warning(f"Suppression de {outliers.sum()} valeurs aberrantes dans {column}")
        return df[~outliers]
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans les données
    """
    # Copie pour ne pas modifier l'original
    df = df.copy()
    
    # Vérifier s'il y a des valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Valeurs manquantes détectées: {missing[missing > 0]}")
        
        # Interpolation linéaire pour OHLCV
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        
        # Forward fill pour les valeurs restantes
        df = df.ffill()
        
        # Backward fill pour les premières lignes si nécessaire
        df = df.bfill()
    
    return df

def calculate_returns(df: pd.DataFrame, 
                     price_col: str = 'close',
                     periods: Optional[list] = None) -> pd.DataFrame:
    """
    Calcule les rendements sur différentes périodes
    """
    if periods is None:
        periods = [1, 5, 15, 30, 60]  # périodes par défaut
    
    df = df.copy()
    
    for period in periods:
        col_name = f'return_{period}'
        # Calculer les rendements et remplacer NaN par 0
        df[col_name] = df[price_col].pct_change(period).fillna(0)
    
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des caractéristiques techniques aux données
    """
    df = df.copy()
    
    # Volatilité
    df['volatility'] = (df['high'] - df['low']) / df['open']
    
    # Volume moyen - on remplit les premières valeurs avec la moyenne simple
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Rendements avec gestion des premières valeurs
    df = calculate_returns(df)
    
    # Remplir les valeurs manquantes avec 0 pour les rendements
    return_columns = ['return_1', 'return_5', 'return_15', 'return_30', 'return_60']
    df[return_columns] = df[return_columns].fillna(0)
    
    return df
