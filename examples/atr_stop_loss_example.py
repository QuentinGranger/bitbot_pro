"""
Exemple d'utilisation de la stratégie de stop-loss ATR.

Cet exemple montre comment utiliser la classe ATRStopLossStrategy pour gérer
dynamiquement les stop-loss en fonction de la volatilité du marché.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Import des modules BitBot
from bitbot.models.market_data import MarketData
from bitbot.strategie.base.ATR import (
    ATRStopLossStrategy, 
    StopLossType, 
    TrailingSLMode,
    VolatilityLevel
)

# Fonction pour générer des données de test
def generate_test_data(n_days=200, volatility_change_points=None):
    """
    Génère des données OHLCV synthétiques avec des changements de volatilité.
    """
    if volatility_change_points is None:
        # Points où la volatilité change (en pourcentage du nombre total de jours)
        volatility_change_points = [0.25, 0.5, 0.75]
    
    # Date de début
    start_date = datetime.now() - timedelta(days=n_days)
    
    # Générer les dates
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Générer le prix de base (marche aléatoire)
    np.random.seed(42)  # Pour la reproductibilité
    
    # Prix initial
    price = 100.0
    prices = [price]
    
    # Volatilité de base et changements
    base_volatility = 1.0
    volatilities = [base_volatility]
    
    # Générer les prix et la volatilité
    for i in range(1, n_days):
        # Changer la volatilité aux points définis
        rel_position = i / n_days
        
        if any(abs(rel_position - p) < 0.01 for p in volatility_change_points):
            # Changement de volatilité
            if base_volatility < 1.5:
                base_volatility *= random.uniform(1.5, 3.0)
            else:
                base_volatility /= random.uniform(1.5, 2.5)
        
        volatilities.append(base_volatility)
        
        # Générer le mouvement de prix en fonction de la volatilité
        daily_return = np.random.normal(0, base_volatility / 100)
        price = price * (1 + daily_return)
        prices.append(price)
    
    # Créer les colonnes OHLCV
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + random.uniform(0, v/100)) for p, v in zip(prices, volatilities)],
        'low': [p * (1 - random.uniform(0, v/100)) for p, v in zip(prices, volatilities)],
        'close': [p * (1 + random.uniform(-v/200, v/200)) for p, v in zip(prices, volatilities)],
        'volume': [random.uniform(1000, 5000) * (1 + v/100) for v in volatilities]
    })
    
    # S'assurer que high >= open, close et low <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # Définir la date comme index
    df.set_index('date', inplace=True)
    
    # Générer des signaux d'entrée simplistes basés sur des seuils
    returns = df['close'].pct_change()
    df['entry_signal'] = (returns > 0.01) & (df['volume'] > df['volume'].rolling(5).mean())
    df['exit_signal'] = returns < -0.01
    
    return df

def main():
    # Générer des données de test
    print("Génération des données de test...")
    df = generate_test_data(n_days=200)
    
    # Créer un objet MarketData
    market_data = MarketData("TEST", df)
    
    # Créer une instance de la stratégie ATR Stop Loss
    print("\nInitialisation de la stratégie ATR Stop Loss...")
    # Test avec stop-loss volatilité-adaptatif
    atr_strategy = ATRStopLossStrategy(
        atr_period=14,
        stop_type=StopLossType.VOLATILITY_ADJUSTED,
        atr_multiplier=2.0,
        volatility_scaling=True
    )
    
    # Exécuter le backtest
    print("\nExécution du backtest...")
    results, metrics = atr_strategy.run_backtest(
        data=market_data,
        entry_signal_col='entry_signal',
        exit_signal_col='exit_signal',
        plot_results=True
    )
    
    # Afficher les métriques
    print("\nRésultats du backtest:")
    print(f"Nombre de trades: {metrics['n_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2f}%")
    print(f"Profit moyen par trade: {metrics['avg_profit']:.2f}%")
    print(f"Profit maximum: {metrics['max_profit']:.2f}%")
    print(f"Perte maximum: {metrics['max_loss']:.2f}%")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Taux de déclenchement des stops: {metrics['stop_hit_rate']:.2f}%")
    
    # Démonstration des différents types de stop-loss
    print("\nDémonstration des différents types de stop-loss...")
    
    # Créer plusieurs stratégies avec différents types de stop
    stop_types = [
        (StopLossType.FIXED, "Stop-loss fixe"),
        (StopLossType.TRAILING, "Stop-loss suiveur"),
        (StopLossType.VOLATILITY_ADJUSTED, "Stop-loss adapté à la volatilité"),
        (StopLossType.CHANDELIER, "Stop-loss chandelier"),
        (StopLossType.MULTI_LEVEL, "Stop-loss multi-niveaux")
    ]
    
    # Simuler l'évolution des différents types de stop sur une partie des données
    test_size = 50
    test_data = df.iloc[-test_size:].copy()
    
    # Préparer les résultats pour l'affichage
    stop_results = {}
    
    for stop_type, name in stop_types:
        # Créer la stratégie
        strategy = ATRStopLossStrategy(
            atr_period=14,
            stop_type=stop_type,
            atr_multiplier=2.0,
            trailing_factor=2.0
        )
        
        # Initialiser la position au début de la période de test
        entry_price = test_data['close'].iloc[0]
        strategy.initialize_position(
            entry_price=entry_price,
            is_long=True,
            data=test_data.iloc[:1]
        )
        
        # Suivre l'évolution du stop-loss
        stops = [strategy.current_stop]
        
        # Mettre à jour le stop-loss à chaque période
        for i in range(1, test_size):
            current_price = test_data['close'].iloc[i]
            strategy.update_stop_loss(
                current_price=current_price,
                data=test_data.iloc[:i+1]
            )
            stops.append(strategy.current_stop)
        
        # Stocker les résultats
        stop_results[name] = stops
    
    # Afficher l'évolution des différents types de stop
    plt.figure(figsize=(12, 8))
    plt.plot(test_data.index, test_data['close'], label='Prix', linewidth=2)
    
    for name, stops in stop_results.items():
        plt.plot(test_data.index, stops, label=name, linestyle='--')
    
    plt.title('Évolution des différents types de stop-loss')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Démonstration du stress test
    print("\nDémonstration du stress test des stop-loss...")
    
    # Créer une stratégie pour le stress test
    stress_strategy = ATRStopLossStrategy(
        atr_period=14,
        stop_type=StopLossType.VOLATILITY_ADJUSTED,
        atr_multiplier=2.0
    )
    
    # Effectuer le stress test
    stress_results = stress_strategy.stress_test_stop_loss(
        data=market_data,
        entry_price=df['close'].iloc[-1],
        initial_stop=df['close'].iloc[-1] * 0.95,  # 5% sous le prix actuel
        is_long=True,
        volatility_increase=3.0,  # Tripler la volatilité
        n_scenarios=5
    )
    
    # Afficher les résultats du stress test
    print("\nRésultats du stress test:")
    for scenario, data in stress_results.items():
        print(f"\nScénario: {scenario}")
        print(f"Facteur de volatilité: {data['volatility_factor']:.2f}x")
        print(f"ATR simulé: {data['simulated_atr']:.2f}")
        print(f"Stop-loss initial: {data['original_stop']:.2f}")
        print(f"Stop-loss stressé: {data['stressed_stop']:.2f}")
        print(f"Changement du stop: {data['stop_change_pct']:.2f}%")
        print(f"Distance prix-stop: {data['price_to_stop_pct']:.2f}%")

if __name__ == "__main__":
    main()
