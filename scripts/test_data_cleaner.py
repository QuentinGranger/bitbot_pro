#!/usr/bin/env python3
"""
Script de test pour démontrer l'utilisation de l'utilitaire de nettoyage de données.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Ajouter le répertoire parent au chemin pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.binance_client import BinanceClient
from bitbot.models.market_data import MarketData, Kline
from bitbot.utils.data_cleaner import DataCleaner, CleaningMethod
from bitbot.utils.logger import logger


def create_test_data_with_outliers(symbol="BTCUSDT", timeframe="1h", num_points=100):
    """
    Crée des données de test avec des valeurs aberrantes artificielles.
    
    Args:
        symbol: Symbole pour les données de test
        timeframe: Timeframe pour les données de test
        num_points: Nombre de points de données
        
    Returns:
        MarketData avec des valeurs aberrantes
    """
    # Créer une série temporelle
    end_date = datetime.now()
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        start_date = end_date - timedelta(minutes=minutes * num_points)
        freq = f"{minutes}min"
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        start_date = end_date - timedelta(hours=hours * num_points)
        freq = f"{hours}H"
    else:  # daily
        start_date = end_date - timedelta(days=num_points)
        freq = "1D"
    
    # Générer l'index de temps
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Générer les données de prix avec une tendance et une composante aléatoire
    base_price = 30000  # Prix de base
    trend = np.linspace(0, 5000, num=len(date_range))  # Tendance linéaire
    noise = np.random.normal(0, 500, size=len(date_range))  # Bruit normal
    seasonality = 1000 * np.sin(np.linspace(0, 4*np.pi, num=len(date_range)))  # Composante saisonnière
    
    # Créer les prix
    close_prices = base_price + trend + noise + seasonality
    
    # Ajouter des valeurs aberrantes (5% des données)
    num_outliers = int(len(date_range) * 0.05)
    outlier_indices = random.sample(range(len(date_range)), num_outliers)
    
    for idx in outlier_indices:
        # Décider si l'outlier est positif ou négatif
        if random.random() > 0.5:
            close_prices[idx] = close_prices[idx] * (1 + random.uniform(0.1, 0.3))  # Augmentation de 10-30%
        else:
            close_prices[idx] = close_prices[idx] * (1 - random.uniform(0.1, 0.3))  # Diminution de 10-30%
    
    # Créer les autres colonnes de prix
    open_prices = close_prices.copy()
    np.random.shuffle(open_prices)  # Mélanger pour simuler des prix d'ouverture
    
    high_prices = np.maximum(close_prices, open_prices) + np.random.uniform(0, 200, size=len(date_range))
    low_prices = np.minimum(close_prices, open_prices) - np.random.uniform(0, 200, size=len(date_range))
    
    # Ajouter quelques valeurs aberrantes aux prix hauts et bas
    for idx in random.sample(outlier_indices, num_outliers // 2):
        high_prices[idx] = high_prices[idx] * (1 + random.uniform(0.2, 0.5))
    
    for idx in random.sample(outlier_indices, num_outliers // 2):
        low_prices[idx] = low_prices[idx] * (1 - random.uniform(0.2, 0.5))
    
    # Générer le volume
    base_volume = 1000
    volume = base_volume + np.random.exponential(scale=500, size=len(date_range))
    
    # Ajouter des pics de volume
    for idx in random.sample(range(len(date_range)), num_outliers):
        volume[idx] = volume[idx] * random.uniform(3, 10)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=date_range)
    
    # Créer l'objet MarketData
    market_data = MarketData(symbol, timeframe)
    market_data.ohlcv = df
    
    logger.info(f"Données de test créées pour {symbol} ({timeframe}) avec {len(df)} points dont {num_outliers} outliers")
    
    return market_data, outlier_indices


def test_with_synthetic_data():
    """Test du nettoyage avec des données synthétiques."""
    
    # Créer des données de test avec des outliers
    symbol = "TEST"
    timeframe = "1h"
    market_data, outlier_indices = create_test_data_with_outliers(symbol, timeframe, num_points=100)
    
    # Créer une instance du nettoyeur
    cleaner = DataCleaner()
    
    # Tester différentes méthodes de nettoyage
    methods = [
        CleaningMethod.INTERPOLATE,
        CleaningMethod.CLIP,
        CleaningMethod.MEDIAN_WINDOW
    ]
    
    plt.figure(figsize=(15, 15))
    
    # Données originales
    plt.subplot(4, 1, 1)
    plt.plot(market_data.ohlcv.index, market_data.ohlcv['close'], label='Original')
    plt.scatter(
        [market_data.ohlcv.index[i] for i in outlier_indices], 
        [market_data.ohlcv['close'].iloc[i] for i in outlier_indices],
        color='red', label='Outliers'
    )
    plt.title('Données originales avec outliers')
    plt.legend()
    
    # Tester chaque méthode
    for i, method in enumerate(methods, 2):
        cleaned_data = cleaner.clean_market_data(
            market_data, 
            std_threshold=3.0,
            window_size=10,
            method=method
        )
        
        # Afficher les résultats
        plt.subplot(4, 1, i)
        plt.plot(market_data.ohlcv.index, market_data.ohlcv['close'], 
                 'gray', alpha=0.5, label='Original')
        plt.plot(cleaned_data.ohlcv.index, cleaned_data.ohlcv['close'], 
                 label=f'Nettoyé ({method.value})')
        plt.title(f'Méthode: {method.value}')
        plt.legend()
        
        # Afficher les statistiques
        stats = cleaner.get_cleaning_stats(symbol)
        print(f"\nStatistiques - Méthode {method.value}:")
        print(f"- Outliers détectés: {stats.get('outliers_detected', 0)}")
        print(f"- Outliers corrigés: {stats.get('outliers_corrected', 0)}")
        print(f"- Pourcentage d'outliers: {stats.get('outlier_percentage', 0):.2f}%")
    
    # Afficher les résultats
    plt.tight_layout()
    plt.savefig("outputs/data_cleaning_comparison.png")
    print(f"Graphique sauvegardé dans outputs/data_cleaning_comparison.png")
    
    # Tester la vérification d'intégrité
    integrity_result = cleaner.verify_ohlc_integrity(market_data)
    print("\nRésultat de la vérification d'intégrité:")
    for key, value in integrity_result.items():
        if key == 'issues':
            print(f"- Issues:")
            for issue in value:
                print(f"  * {issue['type']}: {issue['count']} occurrences")
        else:
            print(f"- {key}: {value}")


def test_with_real_data(symbol="BTCUSDT", timeframe="1h", days=30):
    """Test du nettoyage avec des données réelles."""
    
    # Récupérer des données réelles
    client = BinanceClient()
    
    # Calculer le nombre de bougies à récupérer
    limit = 1000
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        candles_per_day = 24 * 60 / minutes
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        candles_per_day = 24 / hours
    else:  # daily
        candles_per_day = 1
    
    candles_needed = int(days * candles_per_day)
    limit = min(limit, candles_needed)
    
    # Récupérer les données
    start_str = f"{days} days ago UTC"
    klines_data = client.get_historical_klines(symbol, timeframe, limit=limit, start_str=start_str)
    
    if not klines_data:
        logger.error(f"Impossible de récupérer les données pour {symbol}")
        return
    
    # Créer le MarketData
    market_data = MarketData(symbol, timeframe)
    
    # Convertir les données en objets Kline
    klines = []
    for k in klines_data:
        kline = Kline(
            timestamp=datetime.fromtimestamp(k[0]/1000),
            open=k[1],
            high=k[2],
            low=k[3],
            close=k[4],
            volume=k[5],
            close_time=datetime.fromtimestamp(k[6]/1000),
            quote_volume=k[7],
            trades=k[8],
            taker_buy_volume=k[9],
            taker_buy_quote_volume=k[10],
            interval=timeframe
        )
        klines.append(kline)
    
    # Mettre à jour le MarketData
    market_data.update_from_klines(klines)
    
    # Créer une instance du nettoyeur
    cleaner = DataCleaner()
    
    # Nettoyer les données
    cleaned_data = cleaner.clean_market_data(
        market_data, 
        std_threshold=3.0,
        window_size=20,
        method=CleaningMethod.INTERPOLATE
    )
    
    # Afficher les résultats
    stats = cleaner.get_cleaning_stats(symbol)
    print(f"\nStatistiques de nettoyage pour {symbol} ({timeframe}):")
    print(f"- Points de données: {len(market_data.ohlcv)}")
    print(f"- Outliers détectés: {stats.get('outliers_detected', 0)}")
    print(f"- Outliers corrigés: {stats.get('outliers_corrected', 0)}")
    print(f"- Pourcentage d'outliers: {stats.get('outlier_percentage', 0):.2f}%")
    
    # Créer des graphiques pour visualiser les données nettoyées
    plt.figure(figsize=(15, 10))
    
    # Prix de clôture
    plt.subplot(2, 1, 1)
    plt.plot(market_data.ohlcv.index, market_data.ohlcv['close'], 'gray', alpha=0.7, label='Original')
    plt.plot(cleaned_data.ohlcv.index, cleaned_data.ohlcv['close'], 'blue', label='Nettoyé')
    plt.title(f'{symbol} - Prix de clôture')
    plt.legend()
    
    # Volume
    plt.subplot(2, 1, 2)
    plt.plot(market_data.ohlcv.index, market_data.ohlcv['volume'], 'gray', alpha=0.7, label='Original')
    plt.plot(cleaned_data.ohlcv.index, cleaned_data.ohlcv['volume'], 'green', label='Nettoyé')
    plt.title(f'{symbol} - Volume')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/{symbol}_{timeframe}_cleaning.png")
    print(f"Graphique sauvegardé dans outputs/{symbol}_{timeframe}_cleaning.png")
    
    # Vérifier l'intégrité des données
    integrity_result = cleaner.verify_ohlc_integrity(market_data)
    print("\nRésultat de la vérification d'intégrité:")
    for key, value in integrity_result.items():
        if key == 'issues':
            print(f"- Issues:")
            for issue in value:
                print(f"  * {issue['type']}: {issue['count']} occurrences")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs("outputs", exist_ok=True)
    
    print("Test avec des données synthétiques...")
    test_with_synthetic_data()
    
    print("\nTest avec des données réelles...")
    test_with_real_data(symbol="BTCUSDT", timeframe="5m", days=3)
