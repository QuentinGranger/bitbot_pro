#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour tester l'agrégateur de signaux sur un scénario de flash crash.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Chemin absolu vers le répertoire parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from bitbot.strategie.aggregation.signal_aggregator import (
    SignalAggregator, Signal, SignalCategory, AggregatedSignal
)

def generate_flash_crash_market(days=30, start_price=10000.0, crash_size=0.3, 
                               recovery_days=10, random_seed=44):
    """
    Générer des données de marché avec un flash crash suivi d'une récupération.
    
    Args:
        days: Nombre de jours de données
        start_price: Prix initial
        crash_size: Pourcentage de la baisse de prix pendant le crash (0.3 = 30%)
        recovery_days: Nombre de jours pour récupérer après le crash
        random_seed: Graine aléatoire pour la reproductibilité
    
    Returns:
        DataFrame avec les données OHLCV
    """
    np.random.seed(random_seed)
    
    # Date de début
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Déterminer le jour du crash (au tiers du total)
    crash_day = days // 3
    
    # Générer le prix
    price = start_price
    closes = []
    
    for i in range(days):
        if i < crash_day:
            # Avant le crash, marché normal avec légère tendance haussière
            change = np.random.normal(0.001, 0.02)
            price *= (1 + change)
        elif i == crash_day:
            # Jour du crash
            price *= (1 - crash_size)
        elif i < crash_day + recovery_days:
            # Période de récupération
            recovery_percent = (1 / (1 - crash_size) - 1) / recovery_days
            change = np.random.normal(recovery_percent, 0.03)
            price *= (1 + change)
        else:
            # Après la récupération, marché normal
            change = np.random.normal(0.0005, 0.015)
            price *= (1 + change)
        
        closes.append(price)
    
    # Générer high, low, open à partir des close
    volatility = 0.02
    highs = [close * (1 + abs(np.random.normal(0, volatility/2))) for close in closes]
    lows = [close * (1 - abs(np.random.normal(0, volatility/2))) for close in closes]
    
    # Jour du crash avec plus d'écart
    highs[crash_day] = closes[crash_day-1] * 0.95
    lows[crash_day] = closes[crash_day] * 0.9
    
    opens = [low + (high - low) * np.random.random() for high, low in zip(highs, lows)]
    
    # Générer le volume (plus élevé pendant le crash)
    volumes = []
    for i in range(days):
        if i == crash_day or i == crash_day + 1:
            volumes.append(np.random.gamma(5.0, 3000000))
        else:
            volumes.append(np.random.gamma(2.0, 1000000))
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    data.set_index('date', inplace=True)
    return data

def run_flash_crash_backtest(days=60, plot=False):
    """
    Exécuter un backtest avec un scénario de flash crash.
    
    Args:
        days: Nombre de jours dans le backtest
        plot: Si True, affiche un graphique des résultats
        
    Returns:
        Tuple de (market_data, signals, performance_metrics)
    """
    # Générer les données de marché
    market_data = generate_flash_crash_market(days=days)
    
    # Générer des signaux techniques de base pour le test
    market_data = generate_signals(market_data)
    
    # Créer l'agrégateur de signaux
    signal_aggregator = SignalAggregator(
        default_weights={
            SignalCategory.TECHNICAL: 0.7,    # Donner plus de poids aux signaux techniques
            SignalCategory.SENTIMENT: 0.1,
            SignalCategory.ON_CHAIN: 0.1,
            SignalCategory.ORDER_BOOK: 0.05,
            SignalCategory.VOLATILITY: 0.05
        },
        signal_threshold_buy=65,    # Seuil pour générer un signal d'achat
        signal_threshold_sell=32    # Seuil pour générer un signal de vente
    )
    
    # Désactiver temporairement la méthode clean_expired_signals pour les backtests
    signal_aggregator.clean_expired_signals = lambda: None
    
    # Ajouter les signaux et agréger
    signals = []
    trades = []
    
    for idx, row in market_data.iterrows():
        # Ajouter les signaux disponibles pour cette date
        for signal_type in ['ema_crossover', 'sma_crossover', 'rsi', 'market_sentiment', 
                           'volatility_low', 'whale_transaction', 'exchange_netflow', 'macd']:
            signal_col = f"{signal_type}_signal"
            if signal_col in row and not pd.isna(row[signal_col]):
                signal = Signal(
                    name=signal_type,
                    score=row[signal_col],
                    category=SignalCategory.TECHNICAL,
                    timestamp=idx.timestamp(),
                    confidence=0.8,
                    metadata={"source": "backtest"}
                )
                signal_aggregator.add_signal(signal)
        
        # Agréger les signaux
        agg_signal = signal_aggregator.aggregate_signals()
        signals.append(agg_signal)
        
        # Simuler des trades sur les signaux forts
        if agg_signal and agg_signal.score >= 65:  # Signal d'achat fort (seuil abaissé à 65)
            trades.append({
                'date': idx,
                'type': 'buy',
                'price': row['close']
            })
        elif agg_signal and agg_signal.score <= 32:  # Signal de vente fort (seuil augmenté à 32)
            trades.append({
                'date': idx,
                'type': 'sell',
                'price': row['close']
            })
    
    # Calculer les métriques de performance
    performance = calculate_performance(trades, market_data)
    
    # Afficher le graphique si demandé
    if plot:
        plot_results(market_data, signals, trades)
    
    # Filtrer les signaux None
    valid_signals = [s for s in signals if s is not None and s.score is not None]
    
    buy_signals = sum(1 for s in valid_signals if s.score >= 65)
    sell_signals = sum(1 for s in valid_signals if s.score <= 32)
    neutral_signals = sum(1 for s in valid_signals if 32 < s.score < 65)
    
    print(f"\nNombre total de points d'agrégation: {len(valid_signals)}")
    print(f"Signaux d'achat (score >= 65): {buy_signals}")
    print(f"Signaux de vente (score <= 32): {sell_signals}")
    print(f"Signaux neutres: {neutral_signals}")
    
    return market_data, signals, performance

def generate_signals(data):
    """
    Générer des signaux techniques de base pour le test.
    """
    # Calculer les EMA pour EMA crossover
    data['ema_fast'] = data['close'].ewm(span=9).mean()
    data['ema_slow'] = data['close'].ewm(span=21).mean()
    data['ema_fast_gt_slow'] = data['ema_fast'] > data['ema_slow']
    data['prev_ema_fast_gt_slow'] = data['ema_fast_gt_slow'].shift(1).fillna(False)
    
    # Signal EMA crossover
    data['ema_crossover_signal'] = None
    # Signal d'achat quand fast croise slow vers le haut
    buy_crossover = (~data['prev_ema_fast_gt_slow'] & data['ema_fast_gt_slow'])
    data.loc[buy_crossover, 'ema_crossover_signal'] = 75
    # Signal de vente quand fast croise slow vers le bas
    sell_crossover = (data['prev_ema_fast_gt_slow'] & ~data['ema_fast_gt_slow'])
    data.loc[sell_crossover, 'ema_crossover_signal'] = 25
    
    # Calculer les SMA pour SMA crossover
    data['sma_fast'] = data['close'].rolling(window=5).mean()
    data['sma_slow'] = data['close'].rolling(window=20).mean()
    data['sma_fast_gt_slow'] = data['sma_fast'] > data['sma_slow']
    data['prev_sma_fast_gt_slow'] = data['sma_fast_gt_slow'].shift(1).fillna(False)
    
    # Signal SMA crossover
    data['sma_crossover_signal'] = None
    # Signal d'achat quand fast croise slow vers le haut
    buy_crossover = (~data['prev_sma_fast_gt_slow'] & data['sma_fast_gt_slow'])
    data.loc[buy_crossover, 'sma_crossover_signal'] = 70
    # Signal de vente quand fast croise slow vers le bas
    sell_crossover = (data['prev_sma_fast_gt_slow'] & ~data['sma_fast_gt_slow'])
    data.loc[sell_crossover, 'sma_crossover_signal'] = 30
    
    # Calculer le RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Signal RSI
    data['rsi_signal'] = None
    data.loc[data['rsi'] < 30, 'rsi_signal'] = 80  # Survendu - signal d'achat
    data.loc[data['rsi'] > 70, 'rsi_signal'] = 20  # Suracheté - signal de vente
    
    # Renforcer le signal RSI pour les périodes de crash
    flash_crash_start = int(len(data) * 0.3)  # Le crash commence plus tôt
    flash_crash_end = int(len(data) * 0.5)    # Et finit plus tôt (avant la reprise)
    
    for i in range(len(data)):
        if i >= flash_crash_start and i <= flash_crash_end:  # Pendant le crash, pas pendant la reprise
            data.iloc[i, data.columns.get_loc('rsi_signal')] = 5  # Signal de vente très fort
    
    # Générer un signal MACD pendant le flash crash
    data['macd_signal'] = None
    for i in range(len(data)):
        if i >= flash_crash_start and i <= flash_crash_end:
            data.iloc[i, data.columns.get_loc('macd_signal')] = 10  # Signal de vente très fort
    
    # Ajouter un signal de tendance ema_crossover
    data['ema_crossover_signal'] = None
    for i in range(len(data)):
        if i >= flash_crash_start and i <= flash_crash_end:
            data.iloc[i, data.columns.get_loc('ema_crossover_signal')] = 15  # Signal de vente fort
    
    # Générer les signaux de sentiment de marché (plus baissier pendant un flash crash)
    market_sentiment = []
    for i in range(len(data)):
        if i < int(len(data) * 0.3):
            # Phase initiale - sentiment légèrement positif
            sentiment = 50 + 10 * np.random.randn()
        elif i < int(len(data) * 0.5):
            # Pendant le crash - sentiment fortement négatif
            sentiment = 10 + 15 * np.random.randn()  # Sentiment très négatif pendant le crash
        elif i < int(len(data) * 0.8):
            # Juste après le crash - sentiment incertain mais en amélioration
            sentiment = 30 + 20 * np.random.randn()
        else:
            # Phase de reprise - sentiment positif
            sentiment = 60 + 15 * np.random.randn()
            
        sentiment = max(0, min(100, sentiment))  # Limiter entre 0 et 100
        market_sentiment.append(sentiment)
    
    data['market_sentiment_signal'] = market_sentiment
    
    # Simuler des signaux d'alerte de faible volatilité
    # (indiquant une possible explosion de volatilité)
    data['volatility'] = data['close'].pct_change().rolling(window=10).std() * 100
    data['volatility_low_signal'] = None
    data.loc[data['volatility'] < 1.5, 'volatility_low_signal'] = 55  # Signal neutre à légèrement haussier
    
    # Simuler les transactions de baleines (plus de ventes pendant le crash)
    whale_transactions = []
    
    for i in range(len(data)):
        if i < int(len(data) * 0.3):
            # Phase normale - activité équilibrée avec léger biais d'achat
            transaction_sentiment = 55 + 20 * np.random.randn()
        elif i < int(len(data) * 0.5):
            # Crash - les baleines vendent massivement
            transaction_sentiment = 20 + 15 * np.random.randn()
        elif i < int(len(data) * 0.7):
            # Accumulation après le crash - certaines baleines commencent à acheter
            transaction_sentiment = 60 + 25 * np.random.randn()
        else:
            # Reprise - activité d'achat dominante
            transaction_sentiment = 70 + 20 * np.random.randn()
        
        whale_transactions.append(max(0, min(100, transaction_sentiment)))
    
    data['whale_transaction_signal'] = whale_transactions
    
    # Simuler des signaux de flux de fonds sur les exchanges (inflows/outflows)
    exchange_signals = []
    for i in range(len(data)):
        if i < int(len(data) * 0.3):
            # Phase normale
            signal = random.normalvariate(50, 10)
        elif i < int(len(data) * 0.5):
            # Début du crash - entrée de fonds sur les exchanges (pour vendre)
            signal = random.normalvariate(30, 15)  # Plus d'entrées = baissier
        elif i < int(len(data) * 0.7):
            # Pendant le crash - beaucoup de mouvements
            signal = random.normalvariate(20, 20)  # Signal très baissier, volatil
        else:
            # Fin du crash - sortie des exchanges (achat et transfert)
            signal = random.normalvariate(60, 15)  # Plus de sorties = haussier
        exchange_signals.append(max(0, min(100, signal)))
    data['exchange_netflow_signal'] = exchange_signals
    
    return data

def calculate_performance(trades, market_data):
    """
    Calculer les métriques de performance à partir des trades et des données de marché.
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_profit': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0
        }
    
    # Analyser les trades
    trade_pairs = []
    current_buy = None
    
    for trade in trades:
        if trade['type'] == 'buy' and current_buy is None:
            current_buy = trade
        elif trade['type'] == 'sell' and current_buy is not None:
            trade_pairs.append({
                'buy_date': current_buy['date'],
                'buy_price': current_buy['price'],
                'sell_date': trade['date'],
                'sell_price': trade['price'],
                'profit_pct': (trade['price'] - current_buy['price']) / current_buy['price'] * 100
            })
            current_buy = None
    
    # Si le dernier achat n'a pas été vendu, calculer le profit jusqu'au dernier prix
    if current_buy is not None:
        last_price = market_data.iloc[-1]['close']
        trade_pairs.append({
            'buy_date': current_buy['date'],
            'buy_price': current_buy['price'],
            'sell_date': market_data.index[-1],
            'sell_price': last_price,
            'profit_pct': (last_price - current_buy['price']) / current_buy['price'] * 100
        })
    
    if not trade_pairs:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_profit': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0
        }
    
    # Calculer les métriques
    profits = [trade['profit_pct'] for trade in trade_pairs]
    wins = sum(1 for p in profits if p > 0)
    losses = sum(1 for p in profits if p <= 0)
    
    total_profit = sum(p for p in profits if p > 0)
    total_loss = sum(abs(p) for p in profits if p <= 0)
    
    metrics = {
        'total_trades': len(trade_pairs),
        'win_rate': wins / len(trade_pairs) if len(trade_pairs) > 0 else 0,
        'avg_profit': sum(profits) / len(profits) if profits else 0,
        'max_profit': max(profits) if profits else 0,
        'max_loss': min(profits) if profits else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf')
    }
    
    # Calculer le ratio de Sharpe
    if profits:
        returns_mean = sum(profits) / len(profits)
        returns_std = np.std(profits) if len(profits) > 1 else 1
        metrics['sharpe_ratio'] = returns_mean / returns_std if returns_std > 0 else 0
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics

def plot_results(market_data, signals, trades):
    """
    Afficher un graphique des résultats du backtest.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Tracer le prix
    ax1.plot(market_data.index, market_data['close'], label='Prix de clôture')
    
    # Ajouter les trades au graphique
    for trade in trades:
        if trade['type'] == 'buy':
            ax1.scatter(trade['date'], trade['price'], marker='^', color='green', s=100)
        else:  # sell
            ax1.scatter(trade['date'], trade['price'], marker='v', color='red', s=100)
    
    # Configurer l'axe du prix
    ax1.set_title('Backtest sur scénario Flash Crash')
    ax1.set_ylabel('Prix')
    ax1.legend()
    ax1.grid(True)
    
    # Tracer les scores des signaux agrégés
    signal_dates = [datetime.fromtimestamp(s.timestamp) for s in signals]
    signal_scores = [s.score for s in signals]
    
    ax2.plot(signal_dates, signal_scores, label='Score du signal agrégé', color='purple')
    ax2.axhline(y=65, color='green', linestyle='--', alpha=0.7, label='Seuil d\'achat')
    ax2.axhline(y=32, color='red', linestyle='--', alpha=0.7, label='Seuil de vente')
    
    # Configurer l'axe des signaux
    ax2.set_ylabel('Score du signal')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Fonction principale pour exécuter le backtest"""
    print("Test de l'agrégateur de signaux sur un scénario de flash crash...")
    
    # Exécuter le backtest
    market_data, signals, performance = run_flash_crash_backtest(days=60, plot=True)
    
    # Afficher les résultats de performance
    print("\nRésultats de performance:")
    for key, value in performance.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
