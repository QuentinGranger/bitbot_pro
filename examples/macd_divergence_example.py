"""
Exemple d'utilisation de la stratégie de divergence MACD.

Ce script montre comment utiliser la stratégie de divergence MACD pour détecter
les retournements de tendance et générer des signaux d'achat/vente.
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

from bitbot.strategie.indicators.macd_divergence_strategy import MACDDivergenceStrategy, DivergenceType
from bitbot.strategie.base.MACD import MACDSignalType


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
        if with_trend_changes and i % 40 == 0:
            trend = -trend * 1.5  # Inverser et amplifier la tendance
            volatility = volatility * 1.2  # Augmenter la volatilité lors des changements
        
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


def plot_strategy_signals(data, title="Stratégie de Divergence MACD"):
    """
    Visualise les signaux générés par la stratégie.
    
    Args:
        data: DataFrame avec les données de prix et signaux
        title: Titre du graphique
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1.5, 1]})
    
    # Graphique des prix
    ax1.plot(data.index, data['close'], label='Prix de clôture', color='blue')
    
    # Ajouter les signaux d'achat et de vente
    buy_signals = data[data['signal'] == MACDSignalType.BUY]
    sell_signals = data[data['signal'] == MACDSignalType.SELL]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Achat')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Vente')
    
    ax1.set_title(title)
    ax1.set_ylabel('Prix')
    ax1.legend()
    ax1.grid(True)
    
    # Graphique du MACD
    ax2.plot(data.index, data['macd'], label='MACD', color='blue')
    ax2.plot(data.index, data['signal_line'], label='Signal Line', color='red')
    ax2.fill_between(data.index, data['histogram'], 0, 
                     where=(data['histogram'] >= 0), color='green', alpha=0.3)
    ax2.fill_between(data.index, data['histogram'], 0, 
                     where=(data['histogram'] < 0), color='red', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # Graphique de l'ATR en pourcentage
    ax3.plot(data.index, data['atr_pct'], label='ATR %', color='purple')
    ax3.axhline(y=0.5, color='orange', linestyle='--', label='Seuil ATR 0.5%')
    ax3.set_ylabel('ATR %')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale."""
    print("Exemple de la stratégie de divergence MACD")
    
    # Générer des données de test
    print("Génération de données de prix synthétiques...")
    data = generate_sample_price_data(days=200, with_trend_changes=True)
    
    # Initialiser la stratégie
    print("Initialisation de la stratégie de divergence MACD...")
    strategy = MACDDivergenceStrategy(
        macd_fast_period=12,
        macd_slow_period=26,
        macd_signal_period=9,
        lookback_period=30,
        divergence_threshold=0.05,
        use_volatility_filter=True,
        atr_period=14,
        atr_threshold_pct=0.5
    )
    
    # Calcul des signaux historiques
    print("Calcul des signaux historiques...")
    results = strategy.calculate_signals_historical(data)
    
    # Afficher quelques statistiques
    buy_signals = results[results['signal'] == MACDSignalType.BUY]
    sell_signals = results[results['signal'] == MACDSignalType.SELL]
    
    print(f"Nombre de signaux d'achat: {len(buy_signals)}")
    print(f"Nombre de signaux de vente: {len(sell_signals)}")
    
    # Visualiser les résultats
    print("Affichage des résultats...")
    plot_strategy_signals(results, title="Stratégie de Divergence MACD - Avec filtre de volatilité")
    
    # Tester sans filtre de volatilité pour comparaison
    print("Calcul des signaux sans filtre de volatilité pour comparaison...")
    strategy.set_parameters(use_volatility_filter=False)
    results_no_filter = strategy.calculate_signals_historical(data)
    
    buy_signals = results_no_filter[results_no_filter['signal'] == MACDSignalType.BUY]
    sell_signals = results_no_filter[results_no_filter['signal'] == MACDSignalType.SELL]
    
    print(f"Nombre de signaux d'achat (sans filtre): {len(buy_signals)}")
    print(f"Nombre de signaux de vente (sans filtre): {len(sell_signals)}")
    
    plot_strategy_signals(results_no_filter, title="Stratégie de Divergence MACD - Sans filtre de volatilité")
    
    print("Exemple terminé.")


if __name__ == "__main__":
    main()
