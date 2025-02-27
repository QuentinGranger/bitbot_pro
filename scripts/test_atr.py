#!/usr/bin/env python3
"""
Script de test pour le module ATR.
Ce script télécharge des données de marché et affiche les résultats de l'ATR.
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
from bitbot.strategie.base.ATR import ATRIndicator, VolatilityLevel
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module ATR."""
    logger.info("Test du module ATR")
    
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
    
    # Initialiser l'indicateur ATR
    atr_indicator = ATRIndicator(period=14)
    
    # Calculer l'ATR
    logger.info("Calcul de l'ATR")
    df_with_atr = atr_indicator.calculate_atr(market_data)
    
    # Afficher les dernières lignes avec l'ATR
    logger.info("\nDernières lignes du DataFrame avec ATR:")
    print(df_with_atr[['close', 'tr', 'atr', 'atr_pct']].tail().to_string())
    
    # Déterminer le niveau de volatilité
    volatility_level = atr_indicator.get_volatility_level(market_data)
    logger.info(f"\nNiveau de volatilité actuel: {volatility_level.value}")
    
    # Calculer le changement de volatilité
    volatility_change = atr_indicator.calculate_volatility_change(market_data, lookback_period=5)
    logger.info(f"\nChangement de volatilité (5 périodes): {volatility_change:.2f}%")
    
    # Vérifier s'il y a un breakout de volatilité
    is_breakout = atr_indicator.is_volatility_breakout(market_data)
    logger.info(f"\nBreakout de volatilité détecté: {is_breakout}")
    
    # Calculer les bandes ATR
    df_with_bands = atr_indicator.calculate_atr_bands(market_data, multiplier=2.0)
    logger.info("\nDernières lignes du DataFrame avec bandes ATR:")
    print(df_with_bands[['close', 'atr', 'upper_band', 'lower_band']].tail().to_string())
    
    # Calculer les niveaux de stop loss
    long_stop = atr_indicator.calculate_atr_stop_loss(market_data, is_long=True)
    short_stop = atr_indicator.calculate_atr_stop_loss(market_data, is_long=False)
    logger.info(f"\nNiveaux de stop loss basés sur l'ATR:")
    logger.info(f"Stop loss pour position longue: {long_stop:.2f}")
    logger.info(f"Stop loss pour position courte: {short_stop:.2f}")
    
    # Calculer la taille de position recommandée
    position_size, stop_price = atr_indicator.calculate_position_size(
        market_data, risk_pct=1.0, account_size=10000.0
    )
    logger.info(f"\nTaille de position recommandée (compte de 10000, risque 1%):")
    logger.info(f"Taille: {position_size:.4f} {symbol}")
    logger.info(f"Prix de stop loss: {stop_price:.2f}")
    
    # Analyse complète
    analysis = atr_indicator.analyze(market_data)
    logger.info("\nAnalyse complète de l'ATR:")
    for key, value in analysis.items():
        if isinstance(value, VolatilityLevel):
            logger.info(f"{key}: {value.value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Créer un graphique pour l'ATR
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix et bandes ATR
    plt.subplot(2, 1, 1)
    plt.plot(df_with_bands.index, df_with_bands['close'], label='Prix de clôture')
    plt.plot(df_with_bands.index, df_with_bands['upper_band'], 'r--', label='Bande supérieure (ATR x2)')
    plt.plot(df_with_bands.index, df_with_bands['lower_band'], 'g--', label='Bande inférieure (ATR x2)')
    
    plt.title(f'Analyse ATR pour {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour l'ATR
    plt.subplot(2, 1, 2)
    plt.plot(df_with_atr.index, df_with_atr['atr'], label='ATR (14)')
    plt.plot(df_with_atr.index, df_with_atr['tr'], 'r.', alpha=0.3, label='True Range')
    
    plt.title('Average True Range (14)')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'atr_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_atr_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
