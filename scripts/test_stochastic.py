#!/usr/bin/env python3
"""
Script de test pour le module StochasticOscillator.
Ce script télécharge des données de marché et affiche les résultats de l'oscillateur stochastique.
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
from bitbot.strategie.base.StochasticOscillator import StochasticOscillatorIndicator, StochasticSignal
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module StochasticOscillator."""
    logger.info("Test du module StochasticOscillator")
    
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
    
    # Initialiser l'indicateur StochasticOscillator
    stoch_indicator = StochasticOscillatorIndicator(k_period=14, d_period=3, slowing=3)
    
    # Calculer l'oscillateur stochastique
    logger.info("Calcul de l'oscillateur stochastique")
    df_with_stoch = stoch_indicator.calculate_stochastic(market_data)
    
    # Afficher les dernières lignes avec l'oscillateur stochastique
    logger.info("\nDernières lignes du DataFrame avec oscillateur stochastique:")
    print(df_with_stoch[['close', '%K', '%D']].tail().to_string())
    
    # Obtenir le signal
    signal = stoch_indicator.get_signal(market_data)
    logger.info(f"\nSignal de l'oscillateur stochastique: {signal.value}")
    
    # Vérifier les conditions de surachat/survente
    is_overbought = stoch_indicator.is_overbought(market_data)
    is_oversold = stoch_indicator.is_oversold(market_data)
    logger.info(f"\nConditions de marché:")
    logger.info(f"Surachat: {is_overbought}")
    logger.info(f"Survente: {is_oversold}")
    
    # Détecter les divergences
    bullish_divergence, bearish_divergence = stoch_indicator.detect_divergence(market_data)
    logger.info(f"\nDivergences détectées:")
    logger.info(f"Divergence haussière: {bullish_divergence}")
    logger.info(f"Divergence baissière: {bearish_divergence}")
    
    # Calculer le momentum
    df_momentum = stoch_indicator.calculate_stochastic_momentum(market_data)
    logger.info(f"\nMomentum stochastique actuel: {df_momentum['stoch_momentum'].iloc[-1]:.2f}")
    
    # Calculer le Stochastic RSI
    df_stoch_rsi = stoch_indicator.calculate_stochastic_rsi(market_data)
    logger.info("\nDernières lignes du DataFrame avec Stochastic RSI:")
    print(df_stoch_rsi[['close', 'RSI', 'StochRSI_K', 'StochRSI_D']].tail().to_string())
    
    # Analyse complète
    analysis = stoch_indicator.analyze(market_data)
    logger.info("\nAnalyse complète de l'oscillateur stochastique:")
    for key, value in analysis.items():
        if isinstance(value, StochasticSignal):
            logger.info(f"{key}: {value.value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Créer un graphique pour l'oscillateur stochastique
    plt.figure(figsize=(12, 10))
    
    # Subplot pour les prix
    plt.subplot(3, 1, 1)
    plt.plot(df_with_stoch.index, df_with_stoch['close'], label='Prix de clôture')
    
    # Marquer les divergences si détectées
    if bullish_divergence:
        plt.axvspan(df_with_stoch.index[-20], df_with_stoch.index[-1], alpha=0.2, color='green', label='Divergence haussière')
    if bearish_divergence:
        plt.axvspan(df_with_stoch.index[-20], df_with_stoch.index[-1], alpha=0.2, color='red', label='Divergence baissière')
    
    plt.title(f'Analyse Stochastique pour {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour l'oscillateur stochastique
    plt.subplot(3, 1, 2)
    plt.plot(df_with_stoch.index, df_with_stoch['%K'], 'b-', label='%K')
    plt.plot(df_with_stoch.index, df_with_stoch['%D'], 'r-', label='%D')
    
    # Ajouter des lignes horizontales pour les niveaux de surachat/survente
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.3)
    
    # Colorer les zones de surachat et de survente
    plt.axhspan(80, 100, alpha=0.2, color='red', label='Zone de surachat')
    plt.axhspan(0, 20, alpha=0.2, color='green', label='Zone de survente')
    
    plt.title('Oscillateur Stochastique (14,3,3)')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour le Stochastic RSI
    plt.subplot(3, 1, 3)
    plt.plot(df_stoch_rsi.index, df_stoch_rsi['StochRSI_K'], 'b-', label='StochRSI %K')
    plt.plot(df_stoch_rsi.index, df_stoch_rsi['StochRSI_D'], 'r-', label='StochRSI %D')
    
    # Ajouter des lignes horizontales pour les niveaux de surachat/survente
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.3)
    
    # Colorer les zones de surachat et de survente
    plt.axhspan(80, 100, alpha=0.2, color='red')
    plt.axhspan(0, 20, alpha=0.2, color='green')
    
    plt.title('Stochastic RSI (14,3,3)')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'stochastic_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_stochastic_analysis.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
