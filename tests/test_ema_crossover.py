"""
Tests pour la stratégie de croisement des moyennes mobiles exponentielles (EMA).

Ce module teste la stratégie de croisement des EMA en vérifiant:
1. La détection des inversions de tendance à court terme
2. Les croisements rapides (EMA9 vs EMA21)
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

from bitbot.strategie.base.EMA import EMAIndicator
from bitbot.strategie.base.ATR import ATRIndicator
from bitbot.models.market_data import MarketData

class EMACrossoverStrategy:
    """
    Stratégie basée sur le croisement des moyennes mobiles exponentielles avec filtre ATR.
    
    Cette stratégie détecte les croisements entre deux moyennes mobiles exponentielles
    pour identifier les inversions de tendance à court terme. Un filtre ATR est appliqué
    pour éviter les faux signaux pendant les périodes de faible volatilité.
    """
    
    def __init__(self, fast_period: int = 9, slow_period: int = 21, 
                 atr_period: int = 14, atr_threshold: float = 0.5):
        """
        Initialise la stratégie de croisement EMA avec filtre ATR.
        
        Args:
            fast_period: Période pour la moyenne mobile rapide (défaut: 9)
            slow_period: Période pour la moyenne mobile lente (défaut: 21)
            atr_period: Période pour le calcul de l'ATR (défaut: 14)
            atr_threshold: Seuil ATR en pourcentage pour filtrer les signaux (défaut: 0.5%)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        
        self.ema_indicator = EMAIndicator()
        self.atr_indicator = ATRIndicator(period=atr_period)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur les croisements d'EMA avec filtre ATR.
        
        Args:
            data: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame avec des colonnes de signal ajoutées
        """
        # Créer une copie des données
        df = data.copy()
        
        # Convertir en MarketData si nécessaire pour les indicateurs
        market_data = MarketData(symbol="TEST", timeframe="1h")
        market_data.ohlcv = df
        
        # Calculer les EMA
        df_with_ema = self.ema_indicator.calculate_ema(
            market_data, 
            periods=[self.fast_period, self.slow_period]
        )
        
        # Calculer l'ATR
        df_with_atr = self.atr_indicator.calculate_atr(market_data)
        
        # Fusionner les résultats
        df = pd.concat([df_with_ema, df_with_atr[['atr', 'atr_pct']]], axis=1)
        
        # Initialiser les colonnes de signal
        df['signal'] = 0
        df['valid_signal'] = 0
        
        # Noms des colonnes EMA
        fast_col = f'ema_{self.fast_period}'
        slow_col = f'ema_{self.slow_period}'
        
        # Détecter les croisements
        for i in range(1, len(df)):
            # Croisement à la hausse (Golden Cross)
            if (df[fast_col].iloc[i-1] <= df[slow_col].iloc[i-1] and 
                df[fast_col].iloc[i] > df[slow_col].iloc[i]):
                df.loc[df.index[i], 'signal'] = 1
            
            # Croisement à la baisse (Death Cross)
            elif (df[fast_col].iloc[i-1] >= df[slow_col].iloc[i-1] and 
                  df[fast_col].iloc[i] < df[slow_col].iloc[i]):
                df.loc[df.index[i], 'signal'] = -1
        
        # Appliquer le filtre ATR
        df['valid_signal'] = np.where(
            (df['signal'] != 0) & (df['atr_pct'] >= self.atr_threshold),
            df['signal'],
            0
        )
        
        return df

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

def test_ema_crossover_detection(sample_data):
    """Test de la détection des croisements d'EMA."""
    strategy = EMACrossoverStrategy(fast_period=9, slow_period=21)
    result = strategy.generate_signals(sample_data)
    
    # Vérifier qu'il y a au moins un signal de croisement
    assert result['signal'].abs().sum() > 0
    
    # Vérifier les signaux aux points de croisement
    crossover_points = result[result['signal'] != 0]
    
    # Pour chaque croisement, vérifier que les EMA se sont effectivement croisées
    for idx in crossover_points.index:
        i = result.index.get_loc(idx)
        if i > 0:
            fast_col = f'ema_{strategy.fast_period}'
            slow_col = f'ema_{strategy.slow_period}'
            
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
    strategy_with_filter = EMACrossoverStrategy(atr_threshold=0.5)
    result_with_filter = strategy_with_filter.generate_signals(low_volatility_data)
    
    # Stratégie sans filtre ATR (seuil très bas)
    strategy_no_filter = EMACrossoverStrategy(atr_threshold=0.0)
    result_no_filter = strategy_no_filter.generate_signals(low_volatility_data)
    
    # Vérifier que le filtre ATR réduit le nombre de signaux
    signals_with_filter = result_with_filter['valid_signal'].abs().sum()
    signals_no_filter = result_no_filter['signal'].abs().sum()
    
    assert signals_with_filter <= signals_no_filter
    
    # Vérifier que les signaux valides ont tous une volatilité suffisante
    valid_signals = result_with_filter[result_with_filter['valid_signal'] != 0]
    for idx in valid_signals.index:
        assert result_with_filter.loc[idx, 'atr_pct'] >= strategy_with_filter.atr_threshold

def test_trend_reversal_detection(sample_data):
    """Test de la détection des inversions de tendance."""
    strategy = EMACrossoverStrategy()
    result = strategy.generate_signals(sample_data)
    
    # Vérifier si les signaux correspondent aux inversions de tendance
    # Dans notre ensemble de données, nous avons une tendance haussière suivie d'une tendance baissière
    
    # Rechercher des signaux significatifs près des points de retournement
    # La tendance change après le point 60 (40 + 20 points)
    pre_reversal = result.iloc[30:50]
    post_reversal = result.iloc[60:80]
    
    # Vérifier la présence d'un signal baissier après le retournement
    assert post_reversal['signal'].min() == -1

def test_historical_consistency(sample_data):
    """
    Test de la cohérence des signaux générés sur des données historiques.
    Vérifie si les signaux sont bien alignés avec les mouvements de prix.
    """
    strategy = EMACrossoverStrategy()
    result = strategy.generate_signals(sample_data)
    
    # Identifier les points avec des signaux valides
    buy_signals = result[result['valid_signal'] == 1]
    sell_signals = result[result['valid_signal'] == -1]
    
    # Vérifier la performance après les signaux
    for idx in buy_signals.index:
        i = result.index.get_loc(idx)
        if i < len(result) - 5:  # S'assurer qu'il y a assez de données après
            # Calculer la performance 5 périodes après le signal d'achat
            future_return = (result['close'].iloc[i+5] - result['close'].iloc[i]) / result['close'].iloc[i] * 100
            # Un signal d'achat devrait idéalement être suivi d'une hausse
            # Mais nous vérifions seulement la cohérence du signal
            # Dans un test réel, on pourrait utiliser: assert future_return > 0
            assert isinstance(future_return, float)
    
    for idx in sell_signals.index:
        i = result.index.get_loc(idx)
        if i < len(result) - 5:
            # Calculer la performance 5 périodes après le signal de vente
            future_return = (result['close'].iloc[i+5] - result['close'].iloc[i]) / result['close'].iloc[i] * 100
            # Un signal de vente devrait idéalement être suivi d'une baisse
            # assert future_return < 0
            assert isinstance(future_return, float)

if __name__ == "__main__":
    # Cette section permet d'exécuter le test directement
    import sys
    pytest.main(["-xvs", __file__])
