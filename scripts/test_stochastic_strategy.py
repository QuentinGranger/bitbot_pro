#!/usr/bin/env python3
"""
Script de test pour la stratégie basée sur l'oscillateur stochastique.
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
from bitbot.strategie.indicators.stochastic_strategy import StochasticStrategy
from bitbot.models.trade_signal import SignalType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester la stratégie basée sur l'oscillateur stochastique."""
    logger.info("Test de la stratégie basée sur l'oscillateur stochastique")
    
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
    strategy = StochasticStrategy(
        k_period=14,
        d_period=3,
        slowing=3,
        overbought=80,
        oversold=20,
        use_stoch_rsi=False,
        use_divergence=True
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
    plt.figure(figsize=(14, 10))
    
    # Sous-graphique pour le prix et les signaux
    plt.subplot(3, 1, 1)
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['close'], label='Prix de clôture')
    
    # Ajouter les signaux d'achat et de vente
    buy_signals = backtest_results['backtest_data'][backtest_results['backtest_data']['signal'] == 1]
    sell_signals = backtest_results['backtest_data'][backtest_results['backtest_data']['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Achat')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Vente')
    
    plt.title(f'Stratégie Stochastique - Backtest sur {symbol}')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    
    # Sous-graphique pour l'oscillateur stochastique
    plt.subplot(3, 1, 2)
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['%K'], 'b-', label='%K')
    plt.plot(backtest_results['backtest_data'].index, backtest_results['backtest_data']['%D'], 'r-', label='%D')
    
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
    
    # Sous-graphique pour la valeur du portefeuille
    plt.subplot(3, 1, 3)
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
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'stochastic_strategy')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f'{symbol}_stochastic_strategy_backtest.png')
    plt.savefig(output_file)
    logger.info(f"\nGraphique enregistré: {output_file}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
