#!/usr/bin/env python3
"""
Script de test pour le module On-Balance Volume (OBV).
Ce script télécharge des données de marché et affiche les résultats de l'indicateur OBV.
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
from bitbot.strategie.base.OBV import OBVIndicator, OBVSignal
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module OBV."""
    logger.info("Test du module On-Balance Volume (OBV)")
    
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
    
    # Initialiser l'indicateur OBV
    obv_indicator = OBVIndicator(ema_period=20, signal_period=9)
    
    # Calculer l'OBV
    logger.info("Calcul de l'indicateur OBV")
    df_with_obv = obv_indicator.calculate_obv(market_data)
    
    # Afficher les dernières lignes avec l'OBV
    logger.info("\nDernières lignes du DataFrame avec OBV:")
    print(df_with_obv[['close', 'volume', 'OBV', 'OBV_EMA', 'OBV_Signal', 'OBV_Histogram']].tail().to_string())
    
    # Obtenir le signal
    signal = obv_indicator.get_signal(market_data)
    logger.info(f"\nSignal de l'indicateur OBV: {signal.value}")
    
    # Vérifier si l'OBV est en augmentation ou en diminution
    is_increasing = obv_indicator.is_increasing(market_data)
    is_decreasing = obv_indicator.is_decreasing(market_data)
    logger.info(f"\nTendance de l'OBV:")
    logger.info(f"En augmentation: {is_increasing}")
    logger.info(f"En diminution: {is_decreasing}")
    
    # Détecter les divergences
    bullish_divergence, bearish_divergence = obv_indicator.detect_divergence(market_data)
    logger.info(f"\nDivergences détectées:")
    logger.info(f"Divergence haussière: {bullish_divergence}")
    logger.info(f"Divergence baissière: {bearish_divergence}")
    
    # Calculer le momentum
    df_momentum = obv_indicator.calculate_obv_momentum(market_data)
    logger.info(f"\nMomentum OBV actuel: {df_momentum['OBV_Momentum'].iloc[-1]:.2f}%")
    
    # Calculer le VPT
    df_vpt = obv_indicator.calculate_volume_price_trend(market_data)
    logger.info("\nDernières lignes du DataFrame avec VPT:")
    print(df_vpt[['close', 'volume', 'VPT', 'VPT_EMA']].tail().to_string())
    
    # Analyse complète
    analysis = obv_indicator.analyze(market_data)
    logger.info("\nAnalyse complète de l'indicateur OBV:")
    for key, value in analysis.items():
        if isinstance(value, OBVSignal):
            logger.info(f"{key}: {value.value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Créer un graphique pour l'OBV
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix et le volume
    plt.subplot(3, 1, 1)
    
    # Créer un axe secondaire pour le volume
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Tracer le prix
    ax1.plot(df_with_obv.index, df_with_obv['close'], 'b-', label='Prix de clôture')
    ax1.set_ylabel('Prix', color='b')
    
    # Tracer le volume
    ax2.bar(df_with_obv.index, df_with_obv['volume'], alpha=0.3, color='g', label='Volume')
    ax2.set_ylabel('Volume', color='g')
    
    # Marquer les divergences si détectées
    if bullish_divergence:
        plt.axvspan(df_with_obv.index[-20], df_with_obv.index[-1], alpha=0.2, color='green', label='Divergence haussière')
    if bearish_divergence:
        plt.axvspan(df_with_obv.index[-20], df_with_obv.index[-1], alpha=0.2, color='red', label='Divergence baissière')
    
    plt.title(f'Analyse OBV pour {symbol}')
    
    # Combiner les légendes des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True)
    
    # Subplot pour l'OBV et ses lignes
    plt.subplot(3, 1, 2)
    plt.plot(df_with_obv.index, df_with_obv['OBV'], 'b-', label='OBV')
    plt.plot(df_with_obv.index, df_with_obv['OBV_EMA'], 'r-', label='OBV EMA (20)')
    plt.plot(df_with_obv.index, df_with_obv['OBV_Signal'], 'g-', label='OBV Signal (9)')
    
    plt.title('On-Balance Volume (OBV)')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour l'histogramme OBV
    plt.subplot(3, 1, 3)
    
    # Colorer l'histogramme en fonction de sa valeur (positif ou négatif)
    colors = ['green' if x > 0 else 'red' for x in df_with_obv['OBV_Histogram']]
    plt.bar(df_with_obv.index, df_with_obv['OBV_Histogram'], color=colors, alpha=0.6, label='OBV Histogram')
    
    # Ajouter une ligne horizontale à zéro
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title('OBV Histogram (OBV - Signal)')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'obv_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_obv_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
