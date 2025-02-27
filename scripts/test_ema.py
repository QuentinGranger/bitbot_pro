#!/usr/bin/env python3
"""
Script de test pour le module EMA.
Ce script télécharge des données de marché et affiche les moyennes mobiles exponentielles calculées.
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
from bitbot.strategie.base.EMA import EMAIndicator, TimeFrame, TrendType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module EMA."""
    logger.info("Test du module EMA")
    
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
    
    # Initialiser l'indicateur EMA
    ema_indicator = EMAIndicator()
    
    # Calculer les EMA
    logger.info("Calcul des moyennes mobiles exponentielles (EMA)")
    df_with_ema = ema_indicator.calculate_all(market_data)
    
    # Afficher les dernières lignes avec les EMA
    logger.info("\nDernières lignes du DataFrame avec EMA:")
    print(df_with_ema.tail().to_string())
    
    # Identifier la tendance actuelle
    for timeframe in TimeFrame:
        trend = ema_indicator.identify_trend(market_data, timeframe)
        logger.info(f"\nTendance identifiée pour {timeframe.name}: {trend.value}")
    
    # Détecter les croisements
    golden_cross, death_cross = ema_indicator.detect_crossover(market_data)
    logger.info(f"\nDétection de croisement:")
    logger.info(f"Golden Cross (EMA12 croise EMA26 vers le haut): {golden_cross}")
    logger.info(f"Death Cross (EMA12 croise EMA26 vers le bas): {death_cross}")
    
    # Calculer la distance par rapport à la EMA26
    distance = ema_indicator.get_ma_distance(market_data, period=26)
    logger.info(f"\nDistance du prix par rapport à la EMA26: {distance:.2f}%")
    
    # Calculer le MACD
    df_with_macd = ema_indicator.calculate_macd(market_data)
    logger.info("\nDernières lignes du DataFrame avec MACD:")
    print(df_with_macd[['close', 'macd', 'macd_signal', 'macd_hist']].tail().to_string())
    
    # Détecter les signaux MACD
    buy_signal, sell_signal = ema_indicator.detect_macd_signal(market_data)
    logger.info(f"\nSignaux MACD:")
    logger.info(f"Signal d'achat: {buy_signal}")
    logger.info(f"Signal de vente: {sell_signal}")
    
    # Créer un graphique pour les EMA
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix et EMA
    plt.subplot(2, 1, 1)
    plt.plot(df_with_ema.index, df_with_ema['close'], label='Prix de clôture')
    plt.plot(df_with_ema.index, df_with_ema['ema_9'], label='EMA 9')
    plt.plot(df_with_ema.index, df_with_ema['ema_12'], label='EMA 12')
    plt.plot(df_with_ema.index, df_with_ema['ema_26'], label='EMA 26')
    plt.plot(df_with_ema.index, df_with_ema['ema_50'], label='EMA 50')
    plt.plot(df_with_ema.index, df_with_ema['ema_200'], label='EMA 200')
    
    plt.title(f'Analyse EMA pour {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour le MACD
    plt.subplot(2, 1, 2)
    plt.plot(df_with_macd.index, df_with_macd['macd'], label='MACD')
    plt.plot(df_with_macd.index, df_with_macd['macd_signal'], label='Signal')
    
    # Histogramme du MACD
    colors = ['g' if val >= 0 else 'r' for val in df_with_macd['macd_hist']]
    plt.bar(df_with_macd.index, df_with_macd['macd_hist'], label='Histogramme', color=colors, alpha=0.5)
    
    plt.title('MACD (12, 26, 9)')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'ema_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_ema_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
