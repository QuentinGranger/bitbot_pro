#!/usr/bin/env python3
"""
Script de test pour le module BollingerBands.
Ce script télécharge des données de marché et affiche les résultats des bandes de Bollinger.
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
from bitbot.strategie.base.BollingerBands import BollingerBandsIndicator, MarketCondition
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module BollingerBands."""
    logger.info("Test du module BollingerBands")
    
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
    
    # Initialiser l'indicateur BollingerBands
    bb_indicator = BollingerBandsIndicator(period=20, num_std=2.0)
    
    # Calculer les bandes de Bollinger
    logger.info("Calcul des bandes de Bollinger")
    df_with_bb = bb_indicator.calculate_bollinger_bands(market_data)
    
    # Afficher les dernières lignes avec les bandes de Bollinger
    logger.info("\nDernières lignes du DataFrame avec bandes de Bollinger:")
    print(df_with_bb[['close', 'middle_band', 'upper_band', 'lower_band', 'bandwidth', 'percent_b']].tail().to_string())
    
    # Identifier la condition du marché
    market_condition = bb_indicator.identify_market_condition(market_data)
    logger.info(f"\nCondition du marché actuelle: {market_condition.value}")
    
    # Détecter les breakouts
    upper_breakout, lower_breakout = bb_indicator.detect_bollinger_breakout(market_data)
    logger.info(f"\nBreakouts des bandes de Bollinger:")
    logger.info(f"Breakout de la bande supérieure: {upper_breakout}")
    logger.info(f"Breakout de la bande inférieure: {lower_breakout}")
    
    # Détecter les rebonds
    upper_bounce, lower_bounce = bb_indicator.detect_bollinger_bounce(market_data)
    logger.info(f"\nRebonds sur les bandes de Bollinger:")
    logger.info(f"Rebond sur la bande supérieure: {upper_bounce}")
    logger.info(f"Rebond sur la bande inférieure: {lower_bounce}")
    
    # Détecter une compression
    squeeze = bb_indicator.detect_bollinger_squeeze(market_data)
    logger.info(f"\nCompression des bandes (squeeze): {squeeze}")
    
    # Déterminer la tendance
    trend = bb_indicator.detect_bollinger_trend(market_data)
    logger.info(f"\nTendance basée sur les bandes de Bollinger: {trend}")
    
    # Calculer les signaux
    df_with_signals = bb_indicator.calculate_bollinger_signals(market_data)
    signal_count = df_with_signals['bb_signal'].value_counts()
    logger.info(f"\nDistribution des signaux:")
    logger.info(f"Signaux d'achat (1): {signal_count.get(1, 0)}")
    logger.info(f"Signaux neutres (0): {signal_count.get(0, 0)}")
    logger.info(f"Signaux de vente (-1): {signal_count.get(-1, 0)}")
    
    # Calculer la force du signal
    signal_strength = bb_indicator.calculate_bollinger_strength(market_data)
    logger.info(f"\nForce du signal des bandes de Bollinger: {signal_strength:.2f}")
    
    # Analyse complète
    analysis = bb_indicator.analyze(market_data)
    logger.info("\nAnalyse complète des bandes de Bollinger:")
    for key, value in analysis.items():
        if isinstance(value, MarketCondition):
            logger.info(f"{key}: {value.value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Créer un graphique pour les bandes de Bollinger
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix et bandes de Bollinger
    plt.subplot(2, 1, 1)
    plt.plot(df_with_bb.index, df_with_bb['close'], label='Prix de clôture')
    plt.plot(df_with_bb.index, df_with_bb['middle_band'], 'b-', label='Moyenne mobile (20)')
    plt.plot(df_with_bb.index, df_with_bb['upper_band'], 'r--', label='Bande supérieure (2σ)')
    plt.plot(df_with_bb.index, df_with_bb['lower_band'], 'g--', label='Bande inférieure (2σ)')
    
    # Colorer les zones de surachat et de survente
    for i in range(len(df_with_bb)):
        if df_with_bb['percent_b'].iloc[i] > 1:  # Prix au-dessus de la bande supérieure
            plt.axvspan(df_with_bb.index[i], df_with_bb.index[i+1] if i < len(df_with_bb)-1 else df_with_bb.index[i], 
                        alpha=0.2, color='red')
        elif df_with_bb['percent_b'].iloc[i] < 0:  # Prix en-dessous de la bande inférieure
            plt.axvspan(df_with_bb.index[i], df_with_bb.index[i+1] if i < len(df_with_bb)-1 else df_with_bb.index[i], 
                        alpha=0.2, color='green')
    
    plt.title(f'Bandes de Bollinger pour {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour la largeur des bandes et %B
    plt.subplot(2, 1, 2)
    
    # Créer un axe secondaire pour %B
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Tracer la largeur des bandes sur l'axe principal
    ax1.plot(df_with_bb.index, df_with_bb['bandwidth'], 'b-', label='Largeur des bandes (%)')
    ax1.set_ylabel('Largeur des bandes (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Tracer %B sur l'axe secondaire
    ax2.plot(df_with_bb.index, df_with_bb['percent_b'], 'r-', label='%B')
    ax2.set_ylabel('%B', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Ajouter des lignes horizontales pour %B
    ax2.axhline(y=0.0, color='g', linestyle='--', alpha=0.3)  # Ligne à 0% (bande inférieure)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)  # Ligne à 50% (moyenne mobile)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)  # Ligne à 100% (bande supérieure)
    
    plt.title('Largeur des bandes et %B')
    plt.xlabel('Date')
    
    # Ajouter une légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'bollinger_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_bollinger_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
