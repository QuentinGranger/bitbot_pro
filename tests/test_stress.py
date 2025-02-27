"""
Tests des simulations de conditions extrêmes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bitbot.backtest.stress_test import (
    StressTestConfig,
    MarketStressSimulator,
    NetworkStressSimulator,
    StressTestEngine,
    MarketCondition,
    NetworkCondition
)
from bitbot.backtest.engine import BacktestConfig
from bitbot.trading.strategy import BaseStrategy
from bitbot.models.signal import Signal, SignalType

class SimpleTestStrategy(BaseStrategy):
    """Stratégie simple pour les tests."""
    
    async def analyze(self, data: pd.DataFrame) -> Signal:
        if len(data) < 2:
            return None
        
        # Générer un signal si le prix augmente
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return Signal(
                timestamp=data.index[-1],
                symbol="BTC/USDT",
                signal_type=SignalType.LONG,
                price=data['close'].iloc[-1],
                strength=0.8
            )
        return None

@pytest.fixture
def sample_data():
    """Données de test."""
    return pd.DataFrame({
        'open': [10000] * 100,
        'high': [10100] * 100,
        'low': [9900] * 100,
        'close': [10050] * 100,
        'volume': [100] * 100
    }, index=pd.date_range(start='2024-01-01', periods=100, freq='1H'))

@pytest.fixture
def stress_config():
    """Configuration de test pour le stress test."""
    return StressTestConfig(
        volatility_factor=3.0,
        flash_crash_probability=0.05,
        flash_crash_magnitude=0.2,
        pump_dump_probability=0.05,
        pump_dump_magnitude=0.25,
        base_latency=100,
        max_latency=1000,
        latency_volatility=0.3,
        connection_loss_probability=0.05,
        max_offline_duration=60
    )

def test_market_stress_simulation(sample_data, stress_config):
    """Test de la simulation de stress de marché."""
    simulator = MarketStressSimulator(stress_config)
    stressed_data = simulator.apply_stress(sample_data)
    
    # Vérifier que la volatilité a augmenté
    original_volatility = sample_data['close'].std()
    stressed_volatility = stressed_data['close'].std()
    assert stressed_volatility > original_volatility
    
    # Vérifier les flash crashes
    min_price = stressed_data['low'].min()
    max_drawdown = (sample_data['close'].mean() - min_price) / sample_data['close'].mean()
    assert max_drawdown <= stress_config.flash_crash_magnitude * 1.1  # 10% de marge
    
    # Vérifier les pump & dumps
    max_price = stressed_data['high'].max()
    max_pump = (max_price - sample_data['close'].mean()) / sample_data['close'].mean()
    assert max_pump <= stress_config.pump_dump_magnitude * 1.1  # 10% de marge

def test_network_stress_simulation(stress_config):
    """Test de la simulation de stress réseau."""
    simulator = NetworkStressSimulator(stress_config)
    
    # Tester la latence
    latencies = [simulator.get_current_latency() for _ in range(100)]
    assert min(latencies) >= 1
    assert max(latencies) <= stress_config.max_latency * 1.5  # Avec jitter
    
    # Tester les déconnexions
    connections = [simulator.is_connected() for _ in range(100)]
    disconnections = connections.count(False)
    expected_disconnections = int(100 * stress_config.connection_loss_probability)
    assert abs(disconnections - expected_disconnections) <= 5  # Marge statistique

@pytest.mark.asyncio
async def test_stress_test_engine(sample_data, stress_config):
    """Test du moteur de backtest avec stress."""
    backtest_config = BacktestConfig(
        initial_balance=10000,
        commission=0.001,
        slippage=0.001
    )
    
    engine = StressTestEngine(backtest_config, stress_config)
    strategy = SimpleTestStrategy()
    
    results = await engine.run(sample_data, strategy)
    
    # Vérifier que le backtest s'est terminé
    assert results is not None
    assert len(results.trades) > 0
    
    # Vérifier l'impact du stress sur les performances
    assert results.metrics['max_drawdown'] > 0
    assert results.metrics['win_rate'] < 1.0  # Le stress devrait causer des pertes

@pytest.mark.asyncio
async def test_stress_recovery(sample_data, stress_config):
    """Test de la récupération après des conditions extrêmes."""
    engine = StressTestEngine(
        BacktestConfig(initial_balance=10000),
        stress_config
    )
    strategy = SimpleTestStrategy()
    
    # Forcer une déconnexion
    engine.network_simulator.current_condition = NetworkCondition.OFFLINE
    engine.network_simulator.offline_until = datetime.now() + timedelta(seconds=1)
    
    results = await engine.run(sample_data, strategy)
    
    # Vérifier que le système a continué après la déconnexion
    assert len(results.trades) > 0
    
    # Vérifier les logs d'erreur
    # Note: Nécessiterait un mock du logger pour être testé proprement

@pytest.mark.asyncio
async def test_extreme_conditions(sample_data, stress_config):
    """Test des conditions extrêmes combinées."""
    # Configurer des conditions très extrêmes
    stress_config.volatility_factor = 5.0
    stress_config.flash_crash_probability = 0.1
    stress_config.connection_loss_probability = 0.2
    
    engine = StressTestEngine(
        BacktestConfig(initial_balance=10000),
        stress_config
    )
    strategy = SimpleTestStrategy()
    
    results = await engine.run(sample_data, strategy)
    
    # Vérifier que le système survit aux conditions extrêmes
    assert results.metrics['total_trades'] > 0
    assert results.metrics['max_drawdown'] < 1.0  # Ne pas perdre tout le capital
