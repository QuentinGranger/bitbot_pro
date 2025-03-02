"""
Tests pour la stratégie de croisement des moyennes mobiles simples (SMA).

Ce module teste la stratégie de croisement des SMA en vérifiant:
1. La détection des inversions de tendance à court terme
2. Les croisements rapides (SMA9 vs SMA21)
3. L'utilisation du filtre ATR pour éviter les faux signaux en période de faible volatilité
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Ajouter le répertoire parent au chemin de recherche des modules si nécessaire
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.strategie.base.SMA import SMAIndicator
from bitbot.strategie.base.ATR import ATRIndicator
from bitbot.strategie.indicators.sma_crossover_strategy import SMACrossoverStrategy
from bitbot.models.market_data import MarketData

@pytest.fixture
def sample_data():
    """Génère des données synthétiques pour les tests."""
    # Créer des données avec une tendance puis un renversement
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    
    # Tendance haussière initiale
    prices_up = np.linspace(100, 150, 40)
    
    # Consolidation
    prices_flat = np.linspace(150, 152, 20)
    
    # Tendance baissière
    prices_down = np.linspace(152, 120, 40)
    
    prices = np.concatenate([prices_up, prices_flat, prices_down])
    
    # Ajouter du bruit
    np.random.seed(42)
    noise = np.random.normal(0, 1, 100)
    prices = prices + noise
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    return df

@pytest.fixture
def low_volatility_data():
    """Génère des données synthétiques avec faible volatilité."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    
    # Prix presque plat avec de petites oscillations
    base_price = 100
    np.random.seed(42)
    small_noise = np.random.normal(0, 0.2, 100)
    prices = base_price + small_noise
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.2,
        'low': prices - 0.2,
        'close': prices,
        'volume': np.random.randint(100, 200, 100)
    }, index=dates)
    
    return df

@pytest.fixture
def real_price_movement_data():
    """Génère des données synthétiques imitant un mouvement de prix réel."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
    
    # Créer un mouvement de prix plus réaliste
    # Phase 1: Tendance haussière
    trend1 = np.linspace(100, 130, 50) + np.random.normal(0, 1, 50)
    
    # Phase 2: Consolidation avec un petit pullback
    trend2a = np.linspace(130, 125, 20) + np.random.normal(0, 0.8, 20)
    trend2b = np.linspace(125, 132, 30) + np.random.normal(0, 0.8, 30)
    trend2 = np.concatenate([trend2a, trend2b])
    
    # Phase 3: Tendance haussière forte
    trend3 = np.linspace(132, 160, 50) + np.random.normal(0, 1.5, 50)
    
    # Phase 4: Retournement et tendance baissière
    trend4 = np.linspace(160, 140, 50) + np.random.normal(0, 2, 50)
    
    # Combiner les phases
    prices = np.concatenate([trend1, trend2, trend3, trend4])
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0.5, 2, 200),
        'low': prices - np.random.uniform(0.5, 2, 200),
        'close': prices,
        'volume': np.random.randint(100, 1000, 200)
    }, index=dates)
    
    return df

def test_sma_crossover_detection(sample_data):
    """Test de la détection des croisements de SMA."""
    strategy = SMACrossoverStrategy(fast_period=9, slow_period=21)
    result = strategy.generate_signals(sample_data)
    
    # Vérifier qu'il y a au moins un signal de croisement
    assert result['signal'].abs().sum() > 0
    
    # Vérifier les signaux aux points de croisement
    crossover_points = result[result['signal'] != 0]
    
    # Pour chaque croisement, vérifier que les SMA se sont effectivement croisées
    for idx in crossover_points.index:
        i = result.index.get_loc(idx)
        if i > 0:
            fast_col = f'sma_{strategy.fast_period}'
            slow_col = f'sma_{strategy.slow_period}'
            
            # Si signal haussier
            if result.loc[idx, 'signal'] == 1:
                assert result[fast_col].iloc[i-1] <= result[slow_col].iloc[i-1]
                assert result[fast_col].iloc[i] > result[slow_col].iloc[i]
            
            # Si signal baissier
            elif result.loc[idx, 'signal'] == -1:
                assert result[fast_col].iloc[i-1] >= result[slow_col].iloc[i-1]
                assert result[fast_col].iloc[i] < result[slow_col].iloc[i]

def test_atr_filter(low_volatility_data):
    """Test du filtre ATR pour éviter les faux signaux en faible volatilité."""
    # Stratégie avec filtre ATR
    strategy_with_filter = SMACrossoverStrategy(atr_threshold_pct=0.5)
    result_with_filter = strategy_with_filter.generate_signals(low_volatility_data)
    
    # Stratégie sans filtre ATR (seuil très bas)
    strategy_no_filter = SMACrossoverStrategy(atr_threshold_pct=0.0)
    result_no_filter = strategy_no_filter.generate_signals(low_volatility_data)
    
    # Vérifier que le filtre ATR réduit le nombre de signaux
    signals_with_filter = result_with_filter['valid_signal'].abs().sum()
    signals_no_filter = result_no_filter['signal'].abs().sum()
    
    # Il devrait y avoir moins ou le même nombre de signaux valides avec le filtre
    assert signals_with_filter <= signals_no_filter
    
    # Si des signaux sont filtrés, vérifier qu'ils l'ont été correctement
    if signals_with_filter < signals_no_filter:
        # Vérifier que les signaux valides ont tous une volatilité suffisante
        valid_signals = result_with_filter[result_with_filter['valid_signal'] != 0]
        for idx in valid_signals.index:
            assert result_with_filter.loc[idx, 'atr_pct'] >= strategy_with_filter.atr_threshold_pct

def test_trend_reversal_detection(sample_data):
    """Test de la détection des inversions de tendance."""
    strategy = SMACrossoverStrategy()
    result = strategy.generate_signals(sample_data)
    
    # Vérifier si les signaux correspondent aux inversions de tendance
    # Dans notre ensemble de données, nous avons une tendance haussière suivie d'une tendance baissière
    
    # Rechercher des signaux significatifs près des points de retournement
    # La tendance change après le point 60 (40 + 20 points)
    pre_reversal = result.iloc[30:50]
    post_reversal = result.iloc[60:80]
    
    # Vérifier la présence d'un signal baissier après le retournement
    assert post_reversal['signal'].min() == -1

def test_historical_consistency(real_price_movement_data):
    """
    Test de la cohérence des signaux générés sur des données historiques.
    Vérifie si les signaux sont bien alignés avec les mouvements de prix.
    """
    strategy = SMACrossoverStrategy(fast_period=9, slow_period=21)
    result = strategy.generate_signals(real_price_movement_data)
    
    # Identifier les points avec des signaux valides
    buy_signals = result[result['valid_signal'] == 1]
    sell_signals = result[result['valid_signal'] == -1]
    
    # Vérifier qu'il y a au moins quelques signaux d'achat et de vente
    assert len(buy_signals) > 0, "Aucun signal d'achat détecté"
    assert len(sell_signals) > 0, "Aucun signal de vente détecté"
    
    # Vérifier la performance après les signaux
    successful_buys = 0
    for idx in buy_signals.index:
        i = result.index.get_loc(idx)
        if i < len(result) - 10:  # S'assurer qu'il y a assez de données après
            # Calculer la performance 10 périodes après le signal d'achat
            future_return = (result['close'].iloc[i+10] - result['close'].iloc[i]) / result['close'].iloc[i] * 100
            if future_return > 0:
                successful_buys += 1
    
    successful_sells = 0
    for idx in sell_signals.index:
        i = result.index.get_loc(idx)
        if i < len(result) - 10:
            # Calculer la performance 10 périodes après le signal de vente
            future_return = (result['close'].iloc[i] - result['close'].iloc[i+10]) / result['close'].iloc[i] * 100
            if future_return > 0:
                successful_sells += 1
    
    # Calculer les taux de réussite
    buy_success_rate = successful_buys / len(buy_signals) if len(buy_signals) > 0 else 0
    sell_success_rate = successful_sells / len(sell_signals) if len(sell_signals) > 0 else 0
    
    # Vérifier que le taux de réussite est supérieur à 50%
    # Note: Ce test pourrait échouer si les données synthétiques ne reflètent pas bien les marchés réels
    assert buy_success_rate >= 0.5, f"Taux de réussite des signaux d'achat trop faible: {buy_success_rate:.2f}"
    assert sell_success_rate >= 0.5, f"Taux de réussite des signaux de vente trop faible: {sell_success_rate:.2f}"

def test_close_averages_proximity_detection():
    """Test de la détection de proximité des moyennes mobiles pour éviter les faux signaux."""
    # Créer des données où les moyennes mobiles sont très proches
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    
    # Créer une série de prix avec de très petites fluctuations autour d'une valeur
    base_price = 100
    np.random.seed(42)
    prices = base_price + np.sin(np.linspace(0, 6*np.pi, 100)) * 0.5 + np.random.normal(0, 0.1, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.2,
        'low': prices - 0.2,
        'close': prices,
        'volume': np.random.randint(100, 200, 100)
    }, index=dates)
    
    # Calculer manuellement les SMA pour vérifier qu'elles sont proches
    df['sma_9'] = df['close'].rolling(window=9).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    
    # Vérifier que les moyennes sont proches à certains points
    close_mask = (abs(df['sma_9'] - df['sma_21']) < 0.2) & (df.index >= df.index[21])
    close_points = df[close_mask]
    
    assert len(close_points) > 0, "Pas de points où les moyennes sont proches"
    
    # Tester la stratégie avec et sans le filtre ATR
    strategy_with_filter = SMACrossoverStrategy(use_atr_filter=True, atr_threshold_pct=0.5)
    result_with_filter = strategy_with_filter.generate_signals(df)
    
    strategy_no_filter = SMACrossoverStrategy(use_atr_filter=False)
    result_no_filter = strategy_no_filter.generate_signals(df)
    
    # Vérifier qu'il y a moins de signaux avec le filtre ATR
    signals_with_filter = result_with_filter['valid_signal'].abs().sum()
    signals_no_filter = result_no_filter['signal'].abs().sum()
    
    assert signals_with_filter < signals_no_filter, "Le filtre ATR n'a pas réduit les signaux"

if __name__ == "__main__":
    # Cette section permet d'exécuter le test directement
    import sys
    pytest.main(["-xvs", __file__])
