#!/usr/bin/env python3
"""
Script de test pour la stratégie basée sur l'indicateur On-Balance Volume (OBV).
Ce script télécharge des données de marché, applique la stratégie et affiche les résultats.
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
from bitbot.strategie.indicators.obv_strategy import OBVStrategy
from bitbot.models.trade_signal import SignalType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester la stratégie basée sur l'OBV."""
    logger.info("Test de la stratégie basée sur l'indicateur On-Balance Volume (OBV)")
    
    # Initialiser le client Binance
    binance_client = BinanceClient()
    
    # Télécharger les données pour BTC/USDT
    symbol = "BTCUSDT"
    
    logger.info(f"Téléchargement des données pour {symbol} (60 derniers jours)")
    klines_data = binance_client.get_historical_klines(
        symbol=symbol,
        interval="4h",
        start_str="60 days ago UTC"
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
    market_data = MarketData(symbol=symbol, timeframe="4h")
    market_data.ohlcv = df
    
    # Initialiser la stratégie
    strategy = OBVStrategy(
        ema_period=20,
        signal_period=9,
        use_divergence=True,
        use_vpt=False
    )
    
    # Générer des signaux
    logger.info("Génération des signaux de trading")
    signals = strategy.generate_signals(market_data)
    
    # Afficher les signaux
    if signals:
        logger.info(f"\nSignaux générés ({len(signals)}):")
        for signal in signals:
            logger.info(f"Type: {signal.signal_type.name}, "
                       f"Timestamp: {signal.timestamp}, "
                       f"Prix: {signal.price}, "
                       f"Confiance: {signal.confidence}")
    else:
        logger.info("Aucun signal généré pour les données actuelles")
    
    # Effectuer un backtest
    logger.info("\nExécution du backtest")
    backtest_results = strategy.backtest(market_data, initial_capital=10000.0)
    
    # Afficher les résultats du backtest
    logger.info("\nRésultats du backtest:")
    logger.info(f"Capital initial: ${backtest_results['initial_capital']:.2f}")
    logger.info(f"Capital final: ${backtest_results['final_capital']:.2f}")
    logger.info(f"Rendement total: {backtest_results['total_return_pct']:.2f}%")
    logger.info(f"Nombre de trades: {backtest_results['num_trades']}")
    logger.info(f"Ratio de Sharpe: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"Drawdown maximum: {backtest_results['max_drawdown_pct']:.2f}%")
    
    # Créer un graphique pour visualiser les résultats du backtest
    plt.figure(figsize=(14, 12))
    
    # Sous-graphique pour le prix et le volume
    plt.subplot(4, 1, 1)
    
    # Créer un axe secondaire pour le volume
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Tracer le prix
    ax1.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['close'], 'b-', label='Prix de clôture')
    ax1.set_ylabel('Prix', color='b')
    
    # Tracer le volume
    ax2.bar(backtest_results['backtest_data'].index, backtest_results['backtest_data']['volume'], alpha=0.3, color='g', label='Volume')
    ax2.set_ylabel('Volume', color='g')
    
    # Ajouter les signaux d'achat et de vente
    buy_signals = backtest_results['backtest_data'][backtest_results['backtest_data']['signal'] == 1]
    sell_signals = backtest_results['backtest_data'][backtest_results['backtest_data']['signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Achat')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Vente')
    
    plt.title(f'Stratégie OBV - Backtest sur {symbol}')
    
    # Combiner les légendes des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True)
    
    # Sous-graphique pour l'OBV et ses lignes
    plt.subplot(4, 1, 2)
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['OBV'], 'b-', label='OBV')
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['OBV_EMA'], 'r-', label='OBV EMA (20)')
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['OBV_Signal'], 'g-', label='OBV Signal (9)')
    
    # Ajouter les signaux d'achat et de vente
    plt.scatter(buy_signals.index, buy_signals['OBV'], marker='^', color='green', s=100)
    plt.scatter(sell_signals.index, sell_signals['OBV'], marker='v', color='red', s=100)
    
    plt.title('On-Balance Volume (OBV)')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    # Sous-graphique pour l'histogramme OBV
    plt.subplot(4, 1, 3)
    
    # Colorer l'histogramme en fonction de sa valeur (positif ou négatif)
    colors = ['green' if x > 0 else 'red' for x in backtest_results['backtest_data']['OBV_Histogram']]
    plt.bar(backtest_results['backtest_data'].index, backtest_results['backtest_data']['OBV_Histogram'], color=colors, alpha=0.6, label='OBV Histogram')
    
    # Ajouter une ligne horizontale à zéro
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title('OBV Histogram (OBV - Signal)')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    
    # Sous-graphique pour la valeur du portefeuille
    plt.subplot(4, 1, 4)
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['total_value'], label='Valeur du portefeuille')
    
    # Ajouter une ligne pour le capital initial
    plt.axhline(y=backtest_results['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='Capital initial')
    
    plt.title('Performance du portefeuille')
    plt.xlabel('Date')
    plt.ylabel('Valeur ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'obv_strategy')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_obv_strategy_backtest.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
