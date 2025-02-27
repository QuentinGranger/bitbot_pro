#!/usr/bin/env python3
"""
Script de test pour le module MACD.
Ce script télécharge des données de marché et affiche les résultats du MACD.
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
from bitbot.strategie.base.MACD import MACDIndicator, MACDSignalType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module MACD."""
    logger.info("Test du module MACD")
    
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
    
    # Initialiser l'indicateur MACD
    macd_indicator = MACDIndicator()
    
    # Calculer le MACD
    logger.info("Calcul du MACD")
    df_with_macd = macd_indicator.calculate_macd(market_data)
    
    # Afficher les dernières lignes avec le MACD
    logger.info("\nDernières lignes du DataFrame avec MACD:")
    print(df_with_macd[['close', 'macd', 'macd_signal', 'macd_hist']].tail().to_string())
    
    # Détecter les signaux
    signal = macd_indicator.detect_signal(market_data)
    logger.info(f"\nSignal MACD détecté: {signal.value}")
    
    # Détecter les croisements
    bullish_crossover, bearish_crossover = macd_indicator.detect_crossover(market_data)
    logger.info(f"\nDétection de croisement:")
    logger.info(f"Croisement haussier (MACD croise Signal vers le haut): {bullish_crossover}")
    logger.info(f"Croisement baissier (MACD croise Signal vers le bas): {bearish_crossover}")
    
    # Détecter les divergences
    divergence = macd_indicator.detect_divergence(market_data)
    logger.info(f"\nDivergence détectée: {divergence.value}")
    
    # Calculer la force de l'histogramme
    strength = macd_indicator.get_histogram_strength(market_data)
    logger.info(f"\nForce de l'histogramme MACD: {strength:.4f}")
    
    # Analyse complète
    analysis = macd_indicator.analyze(market_data)
    logger.info("\nAnalyse complète du MACD:")
    for key, value in analysis.items():
        if isinstance(value, MACDSignalType):
            logger.info(f"{key}: {value.value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Créer un graphique pour le MACD
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix
    plt.subplot(2, 1, 1)
    plt.plot(df_with_macd.index, df_with_macd['close'], label='Prix de clôture')
    
    plt.title(f'Analyse MACD pour {symbol}')
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
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'macd_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_macd_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
