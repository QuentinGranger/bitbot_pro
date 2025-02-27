#!/usr/bin/env python3
"""
Script de test pour le module SMA.
Ce script télécharge des données de marché et affiche les moyennes mobiles calculées.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.binance_client import BinanceClient
from bitbot.models.market_data import MarketData
from bitbot.strategie.base.SMA import SMAIndicator, TimeFrame, TrendType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module SMA."""
    logger.info("Test du module SMA")
    
    # Initialiser le client Binance
    binance_client = BinanceClient()
    
    # Télécharger les données pour BTC/USDT
    symbol = "BTCUSDT"
    
    logger.info(f"Téléchargement des données pour {symbol} (30 derniers jours)")
    klines_data = binance_client.get_historical_klines(
        symbol=symbol,
        interval="1h",
        start_str="30 days ago UTC"
    )
    
    # Convertir les données en DataFrame
    df = pd.DataFrame(klines_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    
    # Conversion des types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df.set_index("timestamp", inplace=True)
    
    # Créer un objet MarketData
    market_data = MarketData(symbol=symbol, timeframe="1h")
    market_data.ohlcv = df
    
    # Initialiser l'indicateur SMA
    sma_indicator = SMAIndicator()
    
    # Calculer les SMA
    logger.info("Calcul des moyennes mobiles simples (SMA)")
    df_with_sma = sma_indicator.calculate_all(market_data)
    
    # Afficher les dernières lignes avec les SMA
    logger.info("\nDernières lignes du DataFrame avec SMA:")
    print(df_with_sma.tail().to_string())
    
    # Identifier la tendance actuelle
    for timeframe in TimeFrame:
        trend = sma_indicator.identify_trend(market_data, timeframe)
        logger.info(f"\nTendance identifiée pour {timeframe.name}: {trend.value}")
    
    # Détecter les croisements
    golden_cross, death_cross = sma_indicator.detect_crossover(market_data)
    logger.info(f"\nDétection de croisement:")
    logger.info(f"Golden Cross (SMA9 croise SMA20 vers le haut): {golden_cross}")
    logger.info(f"Death Cross (SMA9 croise SMA20 vers le bas): {death_cross}")
    
    # Calculer la distance par rapport à la SMA20
    distance = sma_indicator.get_ma_distance(market_data, period=20)
    logger.info(f"\nDistance du prix par rapport à la SMA20: {distance:.2f}%")
    
    # Créer un graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df_with_sma.index, df_with_sma['close'], label='Prix de clôture')
    plt.plot(df_with_sma.index, df_with_sma['sma_9'], label='SMA 9')
    plt.plot(df_with_sma.index, df_with_sma['sma_20'], label='SMA 20')
    plt.plot(df_with_sma.index, df_with_sma['sma_50'], label='SMA 50')
    plt.plot(df_with_sma.index, df_with_sma['sma_200'], label='SMA 200')
    
    plt.title(f'Analyse SMA pour {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'sma_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_sma_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
