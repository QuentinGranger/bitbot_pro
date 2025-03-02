#!/usr/bin/env python3
"""
Exemple d'utilisation de la stratégie de croisement SMA avec filtre ATR.

Ce script montre comment utiliser la stratégie de croisement des moyennes mobiles
simples avec un filtre ATR pour éviter les faux signaux en période de faible volatilité.
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
from bitbot.strategie.indicators.sma_crossover_strategy import SMACrossoverStrategy
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester la stratégie de croisement SMA."""
    logger.info("Test de la stratégie de croisement SMA avec filtre ATR")
    
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
    
    # Initialiser la stratégie SMA Crossover
    # Paramètres standards: SMA9 vs SMA21, filtre ATR à 0.5%
    strategy = SMACrossoverStrategy(
        fast_period=9,
        slow_period=21,
        atr_period=14,
        atr_threshold_pct=0.5,
        use_atr_filter=True
    )
    
    # Générer les signaux
    logger.info("Génération des signaux de trading")
    signals_df = strategy.generate_signals(market_data)
    
    # Afficher les informations sur la stratégie
    logger.info(f"Configuration de la stratégie:\n{strategy}")
    
    # Compter les signaux
    raw_signals = signals_df['signal'].abs().sum()
    valid_signals = signals_df['valid_signal'].abs().sum()
    
    logger.info(f"Nombre total de signaux bruts: {raw_signals}")
    logger.info(f"Nombre de signaux valides après filtre ATR: {valid_signals}")
    if raw_signals > 0:
        filtered_percentage = (raw_signals - valid_signals) / raw_signals * 100
        logger.info(f"Signaux filtrés: {raw_signals - valid_signals} ({filtered_percentage:.2f}%)")
    
    # Afficher les derniers signaux
    logger.info("\nDerniers signaux générés:")
    recent_signals = signals_df[signals_df['signal'] != 0].tail(5)
    if not recent_signals.empty:
        for idx, row in recent_signals.iterrows():
            signal_type = "ACHAT" if row['signal'] == 1 else "VENTE"
            valid = "VALIDE" if row['valid_signal'] != 0 else "FILTRÉ"
            logger.info(f"{idx}: Signal {signal_type} - {valid} (ATR: {row['atr_pct']:.2f}%)")
    else:
        logger.info("Aucun signal récent")
    
    # Créer un graphique pour visualiser les signaux
    plt.figure(figsize=(14, 10))
    
    # Subplot pour les prix et SMA
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(signals_df.index, signals_df['close'], label='Prix', alpha=0.7)
    plt.plot(signals_df.index, signals_df[f'sma_{strategy.fast_period}'], label=f'SMA{strategy.fast_period}')
    plt.plot(signals_df.index, signals_df[f'sma_{strategy.slow_period}'], label=f'SMA{strategy.slow_period}')
    
    # Ajouter les signaux d'achat et de vente valides
    buy_signals = signals_df[signals_df['valid_signal'] == 1]
    sell_signals = signals_df[signals_df['valid_signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Achat')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Vente')
    
    # Ajouter les signaux filtrés
    filtered_signals = signals_df[(signals_df['signal'] != 0) & (signals_df['valid_signal'] == 0)]
    plt.scatter(filtered_signals.index, filtered_signals['close'], marker='x', color='gray', s=80, label='Filtré')
    
    plt.title(f'Stratégie de croisement SMA pour {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Subplot pour l'ATR
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(signals_df.index, signals_df['atr_pct'], label='ATR %', color='purple')
    plt.axhline(y=strategy.atr_threshold_pct, color='r', linestyle='--', label=f'Seuil ATR ({strategy.atr_threshold_pct}%)')
    
    plt.title('Average True Range (ATR %)')
    plt.xlabel('Date')
    plt.ylabel('ATR %')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'sma_strategy')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_sma_crossover_strategy.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Analyser les dernières données pour obtenir un signal actuel
    current_analysis = strategy.analyze(market_data)
    
    logger.info("\nAnalyse du marché actuel:")
    if current_analysis['signal'] == 1:
        logger.info("SIGNAL D'ACHAT détecté!")
    elif current_analysis['signal'] == -1:
        logger.info("SIGNAL DE VENTE détecté!")
    else:
        logger.info("Pas de signal actuellement")
    
    logger.info(f"Message: {current_analysis['message']}")
    logger.info(f"Force du signal: {current_analysis['signal_strength']:.2f}")
    
    if strategy.use_atr_filter and 'atr_pct' in current_analysis:
        logger.info(f"ATR actuel: {current_analysis['atr_pct']:.2f}% (seuil: {strategy.atr_threshold_pct:.2f}%)")
        logger.info(f"Niveau de volatilité: {current_analysis['volatility_level']}")
    
    # Calculer les niveaux de stop loss et take profit si un signal est détecté
    if current_analysis['signal'] != 0:
        stop_loss, take_profit, risk_reward = strategy.calculate_risk_reward(
            market_data, current_analysis['signal']
        )
        
        last_close = signals_df['close'].iloc[-1]
        
        logger.info("\nNiveaux de gestion de risque:")
        logger.info(f"Prix actuel: {last_close:.2f}")
        logger.info(f"Stop Loss: {stop_loss:.2f} ({abs(stop_loss - last_close) / last_close * 100:.2f}% du prix actuel)")
        logger.info(f"Take Profit: {take_profit:.2f} ({abs(take_profit - last_close) / last_close * 100:.2f}% du prix actuel)")
        logger.info(f"Ratio Risk/Reward: 1:{risk_reward:.2f}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
