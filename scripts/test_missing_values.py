#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour tester la détection et l'interpolation des valeurs manquantes dans les données de marché.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ajouter le répertoire parent au path pour pouvoir importer bitbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.models.market_data import MarketData, Kline
from bitbot.data.binance_client import BinanceClient
from bitbot.utils.data_cleaner import detect_missing_values, fill_missing_values, clean_market_data, remove_corrupted_periods

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "missing_values")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_data_with_gaps(base_data: MarketData, gap_probability: float = 0.05) -> pd.DataFrame:
    """
    Crée un jeu de données de test avec des gaps artificiels.
    
    Args:
        base_data: Données de marché de base
        gap_probability: Probabilité de créer un gap à chaque point
        
    Returns:
        DataFrame avec des gaps
    """
    df = base_data.ohlcv.copy()
    
    # Déterminer le nombre de gaps à créer
    n_gaps = int(len(df) * gap_probability)
    print(f"Création d'un jeu de données avec {n_gaps} gaps ({gap_probability:.1%} de {len(df)} points)")
    
    # Sélectionner aléatoirement des indices pour les gaps
    gap_indices = sorted(random.sample(range(len(df)), n_gaps))
    
    # Créer un masque pour ces indices
    mask = np.zeros(len(df), dtype=bool)
    mask[gap_indices] = True
    
    # DataFrame avec les gaps
    df_with_gaps = df.copy()
    
    # Pour les tests, on remplace les valeurs par NaN, mais on garde les originales pour la vérification
    original_values = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_with_gaps.columns:
            original_values[col] = df_with_gaps.loc[mask, col].copy()
            df_with_gaps.loc[mask, col] = np.nan
    
    # Enregistrer les valeurs originales pour la vérification
    df_with_gaps['original_values'] = False
    df_with_gaps.loc[mask, 'original_values'] = True
    
    return df_with_gaps

def test_gap_detection(base_data: MarketData):
    """
    Teste la détection des gaps dans les données.
    
    Args:
        base_data: Données de marché à analyser
        
    Returns:
        DataFrame avec une colonne 'missing' qui indique où sont les gaps
    """
    # Créer un jeu de données avec des gaps
    df_with_gaps = create_test_data_with_gaps(base_data)
    
    print("\nDétection des valeurs manquantes...")
    result_df = detect_missing_values(df_with_gaps, base_data.timeframe)
    missing_count = result_df['missing'].sum() if 'missing' in result_df.columns else 0
    print(f"Nombre de valeurs manquantes détectées : {missing_count}")
    
    # Visualisation
    plt.figure(figsize=(15, 7))
    plt.title(f"Détection des valeurs manquantes - {base_data.symbol} ({base_data.timeframe})")
    
    # Tracer le prix de clôture
    plt.plot(result_df.index, result_df['close'], 'b-', alpha=0.7, label="Prix")
    
    # Marquer les valeurs manquantes
    if 'missing' in result_df.columns and result_df['missing'].any():
        missing_idx = result_df[result_df['missing']].index
        for idx in missing_idx:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.3)
        
        # Mettre en évidence les premiers gaps pour la lisibilité
        first_gaps = missing_idx[:min(10, len(missing_idx))]
        plt.scatter(first_gaps, result_df.loc[first_gaps, 'close'], 
                    color='red', marker='x', s=100, label="Valeurs manquantes")
    
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_data.symbol}_{base_data.timeframe}_missing_detection.png"))
    plt.close()
    
    print(f"Graphique sauvegardé: {base_data.symbol}_{base_data.timeframe}_missing_detection.png")
    
    return result_df

def test_gap_filling(base_data: MarketData, df_original_with_gaps: pd.DataFrame):
    """
    Teste le remplissage des gaps dans les données.
    
    Args:
        base_data: Données de marché originales
        df_original_with_gaps: DataFrame avec des gaps
    """
    print("\nRemplissage des valeurs manquantes...")
    
    # Préparer le DataFrame avec des valeurs manquantes
    df_with_missing = detect_missing_values(df_original_with_gaps, base_data.timeframe)
    
    # Obtenir les valeurs originales pour calculer l'erreur
    original_df = base_data.ohlcv.copy()

    # Test de différentes méthodes d'interpolation
    plt.figure(figsize=(15, 10))
    plt.title(f"Comparaison des méthodes d'interpolation - {base_data.symbol} ({base_data.timeframe})")
    
    # Dessiner les données originales
    plt.plot(df_with_missing.index, df_with_missing['close'], 'b-', alpha=0.3, label="Données avec gaps")
    
    # Tester différentes méthodes d'interpolation
    methods = {
        "linear": "Linéaire"
    }
    
    plt.figure(figsize=(15, 10))
    
    # Sous-figure 1: Vue d'ensemble
    plt.subplot(2, 1, 1)
    plt.title("Vue d'ensemble des méthodes d'interpolation")
    
    # Dessiner les données originales
    plt.plot(df_with_missing.index, df_with_missing['close'], 'b-', alpha=0.5, label="Données avec gaps")
    
    # Analyser chaque méthode
    colors = ['r', 'g', 'm', 'c']
    metrics = {}
    
    for i, (method_name, method_label) in enumerate(methods.items()):
        # Remplir les valeurs manquantes
        filled_df = fill_missing_values(df_with_missing, max_gap_minutes=60, method=method_name)
        
        # Calculer l'erreur pour les points qui étaient manquants
        common_idx = filled_df.index.intersection(original_df.index)
        if len(common_idx) > 0:
            mse = ((filled_df.loc[common_idx, 'close'] - original_df.loc[common_idx, 'close']) ** 2).mean()
            mae = abs(filled_df.loc[common_idx, 'close'] - original_df.loc[common_idx, 'close']).mean()
            
            metrics[method_name] = {
                'MSE': mse,
                'MAE': mae
            }
            
            # Afficher les métriques
            print(f"Méthode {method_label} - MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            # Tracer la courbe
            plt.plot(filled_df.index, filled_df['close'], f'{colors[i]}-', 
                    label=f"{method_label} (MAE: {mae:.4f})")
    
    plt.legend()
    
    # Sous-figure 2: Zoom sur une région avec des gaps
    plt.subplot(2, 1, 2)
    plt.title("Zoom sur une région avec valeurs manquantes")
    
    # Trouver une région avec des gaps pour zoomer
    missing_indices = df_with_missing[df_with_missing['missing']].index
    if len(missing_indices) > 0:
        zoom_center = missing_indices[0]
        zoom_window = timedelta(hours=2)
        zoom_start = zoom_center - zoom_window
        zoom_end = zoom_center + zoom_window
        
        # Dessiner les données originales dans la région zoomée
        mask = (df_with_missing.index >= zoom_start) & (df_with_missing.index <= zoom_end)
        plt.plot(df_with_missing.index[mask], df_with_missing.loc[mask, 'close'], 
                'b-', alpha=0.5, label="Données avec gaps")
        
        # Tracer chaque méthode dans la région zoomée
        for i, (method_name, method_label) in enumerate(methods.items()):
            filled_df = fill_missing_values(df_with_missing, max_gap_minutes=60, method=method_name)
            mask = (filled_df.index >= zoom_start) & (filled_df.index <= zoom_end)
            plt.plot(filled_df.index[mask], filled_df.loc[mask, 'close'], 
                    f'{colors[i]}-', label=f"{method_label}")
            
            # Marquer les points interpolés
            if 'missing' in df_with_missing.columns:
                interpolated = df_with_missing.loc[mask & df_with_missing['missing']].index
                if len(interpolated) > 0:
                    plt.scatter(interpolated, filled_df.loc[interpolated, 'close'], 
                            color=colors[i], marker='o', s=50)
    
    plt.legend()
    
    # Sauvegarder la figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_data.symbol}_{base_data.timeframe}_interpolation_comparison.png"))
    plt.close()
    
    print(f"Graphique sauvegardé: {base_data.symbol}_{base_data.timeframe}_interpolation_comparison.png")

def test_integrated_cleaning(base_data: MarketData):
    """
    Teste le nettoyage intégré des données avec gestion des valeurs manquantes.
    
    Args:
        base_data: Données de marché de base
    """
    # Créer un jeu de données avec gaps
    df_with_gaps = create_test_data_with_gaps(base_data, gap_probability=0.02)
    
    # Créer une version modifiée de MarketData avec les gaps
    data_with_gaps = MarketData(base_data.symbol, base_data.timeframe)
    data_with_gaps.ohlcv = df_with_gaps
    
    # Nettoyer les données avec différentes configurations
    print("\nTest du nettoyage intégré avec gestion des valeurs manquantes...")
    
    # 1. Sans gestion des valeurs manquantes
    cleaned_no_missing = clean_market_data(
        copy_market_data(data_with_gaps),
        handle_missing=False
    )
    
    # 2. Avec gestion des valeurs manquantes - petits gaps uniquement (5min)
    cleaned_small_gaps = clean_market_data(
        copy_market_data(data_with_gaps),
        handle_missing=True,
        max_gap_minutes=5
    )
    
    # 3. Avec gestion des valeurs manquantes - gaps moyens (20min)
    cleaned_medium_gaps = clean_market_data(
        copy_market_data(data_with_gaps),
        handle_missing=True,
        max_gap_minutes=20
    )
    
    # 4. Avec gestion des valeurs manquantes - tous les gaps (60min)
    cleaned_all_gaps = clean_market_data(
        copy_market_data(data_with_gaps),
        handle_missing=True,
        max_gap_minutes=60
    )
    
    # Comparer les résultats
    comparison = pd.DataFrame({
        'Original': base_data.ohlcv['close'],
        'Avec gaps': data_with_gaps.ohlcv['close'],
        'Sans gestion': cleaned_no_missing.ohlcv['close'],
        'Petits gaps (≤5min)': cleaned_small_gaps.ohlcv['close'],
        'Gaps moyens (≤20min)': cleaned_medium_gaps.ohlcv['close'],
        'Tous gaps (≤60min)': cleaned_all_gaps.ohlcv['close']
    })
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(comparison['Original'], label='Données originales', color='green')
    plt.plot(comparison['Avec gaps'], label='Données avec gaps', color='red')
    plt.title(f"Données originales vs données avec gaps - {base_data.symbol} ({base_data.timeframe})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(comparison['Original'], label='Données originales', color='green')
    plt.plot(comparison['Sans gestion'], label='Sans gestion des valeurs manquantes', color='red')
    plt.plot(comparison['Petits gaps (≤5min)'], label='Gestion des petits gaps', color='blue')
    plt.plot(comparison['Gaps moyens (≤20min)'], label='Gestion des gaps moyens', color='purple')
    plt.plot(comparison['Tous gaps (≤60min)'], label='Gestion de tous les gaps', color='orange')
    plt.title(f"Comparaison des méthodes de nettoyage - {base_data.symbol} ({base_data.timeframe})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_data.symbol}_{base_data.timeframe}_cleaning_comparison.png"))
    print(f"Graphique sauvegardé: {base_data.symbol}_{base_data.timeframe}_cleaning_comparison.png")
    
    # Calculer et afficher les métriques d'erreur
    print("\nMétriques d'erreur pour les différentes méthodes de nettoyage:")
    metrics = []
    
    for method in ['Avec gaps', 'Sans gestion', 'Petits gaps (≤5min)', 'Gaps moyens (≤20min)', 'Tous gaps (≤60min)']:
        common_idx = comparison['Original'].dropna().index.intersection(comparison[method].dropna().index)
        if len(common_idx) > 0:
            mse = ((comparison.loc[common_idx, 'Original'] - comparison.loc[common_idx, method]) ** 2).mean()
            mae = abs(comparison.loc[common_idx, 'Original'] - comparison.loc[common_idx, method]).mean()
            
            data_coverage = len(comparison[method].dropna()) / len(comparison['Original'].dropna()) * 100
            
            metrics.append({
                'Méthode': method,
                'MSE': mse,
                'MAE': mae,
                'Couverture (%)': data_coverage
            })
    
    metrics_df = pd.DataFrame(metrics).set_index('Méthode')
    print(metrics_df)
    
    # Sauvegarder les métriques
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"{base_data.symbol}_{base_data.timeframe}_cleaning_metrics.csv"))
    print(f"Métriques sauvegardées: {base_data.symbol}_{base_data.timeframe}_cleaning_metrics.csv")

def create_market_data_from_klines(symbol: str, timeframe: str, klines_data: list) -> MarketData:
    """
    Convertit les données brutes de klines en objet MarketData.
    
    Args:
        symbol: Symbole de la paire
        timeframe: Timeframe des données
        klines_data: Données brutes des klines
        
    Returns:
        Objet MarketData
    """
    market_data = MarketData(symbol, timeframe)
    
    # Conversion des klines en dataframe avec les bonnes colonnes
    df = pd.DataFrame(klines_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    # Conversion des types
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
        df[col] = pd.to_numeric(df[col])
    
    # Conversion des timestamps en datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Utiliser open_time comme index
    df.set_index('open_time', inplace=True)
    
    # Garder seulement les colonnes OHLCV
    market_data.ohlcv = df[['open', 'high', 'low', 'close', 'volume']]
    
    return market_data

def get_timedelta(timeframe: str) -> timedelta:
    """
    Convertit un timeframe en timedelta.
    
    Args:
        timeframe: Chaîne représentant le timeframe (ex: '1m', '1h', '1d')
        
    Returns:
        Objet timedelta correspondant
    """
    if timeframe.endswith('m'):
        return timedelta(minutes=int(timeframe[:-1]))
    elif timeframe.endswith('h'):
        return timedelta(hours=int(timeframe[:-1]))
    elif timeframe.endswith('d'):
        return timedelta(days=int(timeframe[:-1]))
    else:
        raise ValueError(f"Format de timeframe non reconnu: {timeframe}")

def copy_market_data(market_data: MarketData) -> MarketData:
    """
    Crée une copie profonde d'un objet MarketData.
    
    Args:
        market_data: Objet MarketData à copier
        
    Returns:
        Copie de l'objet MarketData
    """
    copied = MarketData(market_data.symbol, market_data.timeframe)
    copied.ohlcv = market_data.ohlcv.copy()
    copied.indicators = {k: v.copy() if hasattr(v, 'copy') else v 
                          for k, v in market_data.indicators.items()}
    copied.metadata = market_data.metadata.copy() if market_data.metadata else {}
    copied.signals = market_data.signals.copy() if hasattr(market_data.signals, 'copy') else []
    
    return copied

def test_corrupted_periods(base_data: MarketData):
    """
    Teste la détection et la suppression des périodes corrompues dans les données.
    
    Args:
        base_data: Données de marché de base
    """
    print("\nTest de la détection et suppression des périodes corrompues...")
    
    # Créer un jeu de données avec de fortes densités de gaps
    df_original = base_data.ohlcv.copy()
    
    # Sélectionner une période pour la corrompre fortement
    start_idx = int(len(df_original) * 0.3)
    end_idx = int(len(df_original) * 0.4)
    start_date = df_original.index[start_idx]
    end_date = df_original.index[end_idx]
    
    print(f"Corruption de la période du {start_date} au {end_date}")
    
    # Sélectionner les indices à supprimer dans cette période pour créer des gaps temporels réels
    period_indices = df_original.loc[start_date:end_date].index
    corruption_rate = 0.7  # 70% de gaps dans cette période
    n_gaps = int(len(period_indices) * corruption_rate)
    print(f"Suppression de {n_gaps} points ({corruption_rate:.1%} de {len(period_indices)} points dans la période)")
    
    # Sélectionner aléatoirement des indices à supprimer
    indices_to_drop = random.sample(list(period_indices), n_gaps)
    
    # Créer un DataFrame corrompu en supprimant ces indices
    df_corrupted = df_original.drop(indices_to_drop)
    
    # Détecter les valeurs manquantes
    with_missing = detect_missing_values(df_corrupted, base_data.timeframe)
    
    # Vérifier que les gaps ont bien été détectés
    missing_count = with_missing['missing'].sum() if 'missing' in with_missing.columns else 0
    print(f"Nombre de gaps détectés: {missing_count} ({missing_count/len(with_missing):.1%} du dataset total)")
    
    # Tester la suppression des périodes corrompues avec différents seuils
    thresholds = [0.05, 0.1, 0.2, 0.5]
    results = {}
    
    for threshold in thresholds:
        cleaned_df = remove_corrupted_periods(with_missing, base_data.timeframe, threshold)
        
        # Vérifier si la période corrompue a été supprimée
        removed_period = len(with_missing) - len(cleaned_df)
        
        results[threshold] = {
            'points_removed': removed_period,
            'period_removed': removed_period > 0
        }
    
    # Afficher les résultats
    print("\nRésultats de la suppression des périodes corrompues:")
    print(f"{'Seuil':<10} | {'Points supprimés':<20} | {'Période supprimée':<20}")
    print(f"{'-'*10} | {'-'*20} | {'-'*20}")
    
    for threshold, result in results.items():
        print(f"{threshold:<10.2f} | {result['points_removed']:<20} | {result['period_removed']:<20}")
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    plt.title(f"Détection des périodes corrompues - {base_data.symbol} ({base_data.timeframe})")
    
    # Période originale et corrompue
    plt.subplot(2, 1, 1)
    plt.title("Période corrompue (rouge = gaps)")
    
    # Tracer les données originales
    plt.plot(df_original.index, df_original['close'], 'b-', alpha=0.3, label="Données complètes")
    plt.plot(df_corrupted.index, df_corrupted['close'], 'g-', alpha=0.7, label="Données avec gaps")
    
    # Mettre en évidence la période analysée
    plt.axvspan(start_date, end_date, color='yellow', alpha=0.3, label="Période corrompue")
    
    # Mettre en évidence les valeurs manquantes
    missing_idx = with_missing[with_missing['missing']].index
    if len(missing_idx) > 0:
        # Pour visualiser les points manquants, nous devons interpoler les valeurs
        interp_values = np.interp(
            [pd.Timestamp(t).timestamp() for t in missing_idx], 
            [pd.Timestamp(t).timestamp() for t in df_original.index], 
            df_original['close'].values
        )
        plt.scatter(missing_idx, interp_values, 
                    color='red', marker='x', s=50, label="Valeurs manquantes")
    
    plt.legend()
    
    # Période après suppression
    plt.subplot(2, 1, 2)
    plt.title(f"Après suppression (seuil = 0.1)")
    
    cleaned_df = remove_corrupted_periods(with_missing, base_data.timeframe, 0.1)
    
    plt.plot(cleaned_df.index, cleaned_df['close'], 'g-', label="Prix après nettoyage")
    
    # Tracer une ligne où était la période supprimée
    plt.axvspan(start_date, end_date, color='red', alpha=0.1, label="Période potentiellement supprimée")
    
    plt.legend()
    
    # Sauvegarder la figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_data.symbol}_{base_data.timeframe}_corrupted_periods.png"))
    plt.close()
    
    print(f"Graphique sauvegardé: {base_data.symbol}_{base_data.timeframe}_corrupted_periods.png")

if __name__ == "__main__":
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Test de la gestion des valeurs manquantes")
    print("=======================================")
    print(f"Paires analysées: BTCUSDT, ETHUSDT")
    print(f"Timeframes: 5m, 1h")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print("=======================================\n")
    
    # Boucler sur les symboles et timeframes
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["5m", "1h"]
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n--- Test pour {symbol} ({timeframe}) ---")
            
            # Récupérer les données historiques
            client = BinanceClient()
            klines_data = client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                limit=1000,
                start_str="5 days ago UTC"
            )
            
            # Convertir les données en objet MarketData
            market_data = create_market_data_from_klines(symbol, timeframe, klines_data)
            
            # Créer un jeu de données avec gaps
            df_with_missing = create_test_data_with_gaps(market_data, gap_probability=0.03)
            
            # 1. Tester la détection des valeurs manquantes
            test_gap_detection(market_data)
            
            # 2. Tester le remplissage des gaps
            test_gap_filling(market_data, df_with_missing)
            
            # 3. Tester le nettoyage intégré
            market_data_with_gaps = copy_market_data(market_data)
            market_data_with_gaps.ohlcv = create_test_data_with_gaps(market_data, gap_probability=0.02)
            test_integrated_cleaning(market_data_with_gaps)
            
            # 4. Tester la détection et suppression des périodes corrompues
            if timeframe == '5m':  # N'exécuter que pour le timeframe 5m pour gagner du temps
                test_corrupted_periods(market_data)
    
    print("\nTests terminés. Tous les résultats ont été sauvegardés dans", OUTPUT_DIR)
