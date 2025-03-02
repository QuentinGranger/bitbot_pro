"""
Exemple d'utilisation de la stratégie RSI.

Ce script montre comment utiliser la stratégie RSI pour détecter
les situations de surachat/survente et générer des signaux d'achat/vente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
import sys

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.strategie.indicators.rsi_strategy import RSIStrategy
from bitbot.strategie.base.RSI import TrendType
from bitbot.models.trade_signal import SignalType


def generate_sample_price_data(days=200, with_trend_changes=True):
    """
    Génère des données de prix synthétiques avec des changements de tendance.
    
    Args:
        days: Nombre de jours de données à générer
        with_trend_changes: Si True, inclut des changements de tendance prononcés
        
    Returns:
        DataFrame avec les données OHLCV
    """
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Générer un prix avec une tendance globale mais des fluctuations aléatoires
    base_price = 100.0
    prices = [base_price]
    
    # Paramètres pour les tendances
    trend = 0.1  # Tendance initiale (haussière)
    volatility = 0.5  # Volatilité de base
    
    for i in range(1, days):
        # Changer de tendance à certains points si requis
        if with_trend_changes:
            if i % 40 == 0:
                trend = -trend * 1.5  # Inverser et amplifier la tendance
                volatility = volatility * 1.2  # Augmenter la volatilité lors des changements
            elif i % 20 == 0 and i % 40 != 0:
                trend = trend * 0.2  # Réduire la tendance (période de range)
                volatility = volatility * 0.8  # Réduire la volatilité
        
        # Ajouter du bruit aléatoire à la tendance
        random_change = (random.random() - 0.5) * volatility
        new_price = prices[-1] * (1 + trend/100 + random_change/100)
        prices.append(new_price)
        
        # Réduire progressivement la volatilité
        if i % 10 == 0 and volatility > 0.5:
            volatility = volatility * 0.95
    
    # Créer les colonnes OHLCV
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'] * 0.99)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.rand(len(df)) * 0.01)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.rand(len(df)) * 0.01)
    df['volume'] = np.random.randint(1000, 100000, size=len(df))
    
    return df


def plot_strategy_signals(data, title="Stratégie RSI"):
    """
    Visualise les signaux générés par la stratégie.
    
    Args:
        data: DataFrame avec les données de prix et signaux
        title: Titre du graphique
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1.5]})
    
    # Graphique des prix
    ax1.plot(data.index, data['close'], label='Prix de clôture', color='blue')
    
    # Ajouter les signaux d'achat et de vente
    buy_signals = data[data['signal'] == SignalType.BUY]
    strong_buy_signals = data[data['signal'] == SignalType.STRONG_BUY]
    sell_signals = data[data['signal'] == SignalType.SELL]
    strong_sell_signals = data[data['signal'] == SignalType.STRONG_SELL]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=80, label='Achat')
    ax1.scatter(strong_buy_signals.index, strong_buy_signals['close'], marker='^', color='green', s=120, label='Achat fort')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=80, label='Vente')
    ax1.scatter(strong_sell_signals.index, strong_sell_signals['close'], marker='v', color='red', s=120, label='Vente forte')
    
    # Ajouter les annotations de tendance
    bull_periods = data[data['trend'] == TrendType.BULL.value]
    bear_periods = data[data['trend'] == TrendType.BEAR.value]
    range_periods = data[data['trend'] == TrendType.RANGE.value]
    
    # Colorer le fond en fonction de la tendance
    if not bull_periods.empty:
        ax1.fill_between(bull_periods.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         color='green', alpha=0.1)
    if not bear_periods.empty:
        ax1.fill_between(bear_periods.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         color='red', alpha=0.1)
    if not range_periods.empty:
        ax1.fill_between(range_periods.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         color='gray', alpha=0.1)
    
    ax1.set_title(title)
    ax1.set_ylabel('Prix')
    ax1.legend()
    ax1.grid(True)
    
    # Graphique du RSI
    ax2.plot(data.index, data['rsi'], label='RSI', color='purple')
    
    # Ajouter les lignes de seuil
    ax2.axhline(y=70, color='red', linestyle='--', label='Surachat 70')
    ax2.axhline(y=30, color='green', linestyle='--', label='Survente 30')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    # Ajouter le score composite
    ax2.plot(data.index, (data['composite_score'] * 50) + 50, label='Score composite', color='orange', alpha=0.7)
    
    ax2.set_ylabel('RSI / Score')
    ax2.set_ylim([0, 100])
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale."""
    print("Exemple de la stratégie RSI")
    
    # Générer des données de test
    print("Génération de données de prix synthétiques...")
    data = generate_sample_price_data(days=200, with_trend_changes=True)
    
    # Initialiser la stratégie avec seuils dynamiques
    print("Initialisation de la stratégie RSI avec seuils dynamiques...")
    strategy = RSIStrategy(
        period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        strong_overbought_threshold=80,
        strong_oversold_threshold=20,
        use_dynamic_thresholds=True,
        trend_weight=1.0,
        range_weight=0.5,
        lookback_period=50
    )
    
    # Calcul des signaux historiques
    print("Calcul des signaux historiques...")
    results = strategy.calculate_signals_historical(data)
    
    # Afficher quelques statistiques
    buy_signals = results[results['signal'].isin([SignalType.BUY, SignalType.STRONG_BUY])]
    sell_signals = results[results['signal'].isin([SignalType.SELL, SignalType.STRONG_SELL])]
    
    print(f"Nombre de signaux d'achat: {len(buy_signals)}")
    print(f"Nombre de signaux de vente: {len(sell_signals)}")
    
    # Visualiser les résultats
    print("Affichage des résultats...")
    plot_strategy_signals(results, title="Stratégie RSI - Avec seuils dynamiques")
    
    # Tester sans seuils dynamiques pour comparaison
    print("Calcul des signaux sans seuils dynamiques pour comparaison...")
    strategy.set_parameters(use_dynamic_thresholds=False)
    results_static = strategy.calculate_signals_historical(data)
    
    buy_signals = results_static[results_static['signal'].isin([SignalType.BUY, SignalType.STRONG_BUY])]
    sell_signals = results_static[results_static['signal'].isin([SignalType.SELL, SignalType.STRONG_SELL])]
    
    print(f"Nombre de signaux d'achat (sans seuils dynamiques): {len(buy_signals)}")
    print(f"Nombre de signaux de vente (sans seuils dynamiques): {len(sell_signals)}")
    
    plot_strategy_signals(results_static, title="Stratégie RSI - Sans seuils dynamiques")
    
    # Comparer les performances (nombre de signaux)
    trend_counts = results['trend'].value_counts()
    print("\nAnalyse des tendances détectées:")
    for trend, count in trend_counts.items():
        print(f"  {trend}: {count} périodes")
    
    # Montrer un exemple où le poids du RSI est réduit dans les marchés en range
    range_periods = results[results['trend'] == TrendType.RANGE.value]
    if not range_periods.empty:
        range_signals = range_periods[range_periods['signal'].isin([SignalType.BUY, SignalType.SELL, SignalType.STRONG_BUY, SignalType.STRONG_SELL])]
        print(f"\nNombre de signaux générés pendant les périodes sans tendance forte: {len(range_signals)}")
        print("Avec un poids réduit du RSI pour éviter les faux signaux.")
    
    print("\nExemple terminé.")


if __name__ == "__main__":
    main()
