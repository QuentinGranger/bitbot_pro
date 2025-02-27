"""
Tests des stratégies de trading.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from bitbot.trading.strategy import BaseStrategy
from bitbot.trading.indicators import TechnicalIndicators
from bitbot.models.signal import Signal, SignalType
from bitbot.models.position import Position

class TestStrategy(BaseStrategy):
    """Stratégie de test basée sur le croisement de moyennes mobiles."""
    
    def __init__(self, fast_period=10, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    async def analyze(self, data: pd.DataFrame) -> Signal:
        ti = TechnicalIndicators(data)
        
        fast_ma = ti.sma(self.fast_period)
        slow_ma = ti.sma(self.slow_period)
        
        # Croisement à la hausse
        if fast_ma.iloc[-2] < slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
            return Signal(
                timestamp=data.index[-1],
                symbol="BTC/USDT",
                signal_type=SignalType.LONG,
                price=data['close'].iloc[-1],
                strength=0.8
            )
        
        # Croisement à la baisse
        if fast_ma.iloc[-2] > slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
            return Signal(
                timestamp=data.index[-1],
                symbol="BTC/USDT",
                signal_type=SignalType.SHORT,
                price=data['close'].iloc[-1],
                strength=0.8
            )
        
        return None

@pytest.fixture
def strategy():
    """Instance de la stratégie de test."""
    return TestStrategy(fast_period=2, slow_period=3)

@pytest.fixture
def mock_exchange():
    """Mock d'un exchange."""
    exchange = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock(return_value=pd.DataFrame({
        'open': [8000, 8100, 8200, 8300, 8400],
        'high': [8050, 8150, 8250, 8350, 8450],
        'low': [7950, 8050, 8150, 8250, 8350],
        'close': [8100, 8200, 8300, 8400, 8500],
        'volume': [10, 12, 15, 11, 13]
    }))
    return exchange

@pytest.mark.asyncio
async def test_strategy_long_signal(strategy, mock_exchange, sample_ohlcv_data):
    """Test de la génération d'un signal long."""
    # Créer des données avec un croisement à la hausse
    data = pd.DataFrame(sample_ohlcv_data)
    data.set_index('timestamp', inplace=True)
    
    # Simuler un croisement à la hausse
    data['close'] = [8000, 8100, 8300]  # La moyenne rapide dépasse la lente
    
    signal = await strategy.analyze(data)
    
    assert signal is not None
    assert signal.signal_type == SignalType.LONG
    assert signal.symbol == "BTC/USDT"
    assert signal.strength == 0.8

@pytest.mark.asyncio
async def test_strategy_short_signal(strategy, mock_exchange, sample_ohlcv_data):
    """Test de la génération d'un signal short."""
    # Créer des données avec un croisement à la baisse
    data = pd.DataFrame(sample_ohlcv_data)
    data.set_index('timestamp', inplace=True)
    
    # Simuler un croisement à la baisse
    data['close'] = [8300, 8200, 8000]  # La moyenne rapide passe sous la lente
    
    signal = await strategy.analyze(data)
    
    assert signal is not None
    assert signal.signal_type == SignalType.SHORT
    assert signal.symbol == "BTC/USDT"
    assert signal.strength == 0.8

@pytest.mark.asyncio
async def test_strategy_no_signal(strategy, mock_exchange, sample_ohlcv_data):
    """Test de l'absence de signal."""
    # Créer des données sans croisement
    data = pd.DataFrame(sample_ohlcv_data)
    data.set_index('timestamp', inplace=True)
    
    # Pas de croisement
    data['close'] = [8000, 8100, 8200]  # Tendance continue
    
    signal = await strategy.analyze(data)
    
    assert signal is None

@pytest.mark.asyncio
async def test_strategy_risk_management(strategy, mock_exchange):
    """Test de la gestion du risque."""
    # Créer une position
    position = Position(
        symbol="BTC/USDT",
        entry_price=8000,
        amount=1.0,
        side=SignalType.LONG,
        stop_loss=7900,
        take_profit=8200
    )
    
    # Prix actuel au-dessus du take profit
    current_price = 8300
    
    should_close = strategy.check_exit_conditions(position, current_price)
    assert should_close is True
    
    # Prix actuel en-dessous du stop loss
    current_price = 7800
    
    should_close = strategy.check_exit_conditions(position, current_price)
    assert should_close is True
    
    # Prix actuel dans la zone de trading
    current_price = 8100
    
    should_close = strategy.check_exit_conditions(position, current_price)
    assert should_close is False

@pytest.mark.asyncio
async def test_strategy_position_sizing(strategy, mock_exchange):
    """Test du calcul de la taille de position."""
    # Mock du solde du compte
    balance = {
        'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
    }
    mock_exchange.fetch_balance = AsyncMock(return_value=balance)
    
    # Test avec un risque de 1%
    risk_percent = 0.01
    entry_price = 8000
    stop_loss = 7900
    
    size = await strategy.calculate_position_size(
        mock_exchange,
        "BTC/USDT",
        entry_price,
        stop_loss,
        risk_percent
    )
    
    # Vérifier que la taille respecte le risque maximum
    risk_amount = 10000.0 * risk_percent  # 100 USDT de risque
    max_size = risk_amount / (entry_price - stop_loss)  # Taille maximale basée sur le risque
    
    assert size <= max_size

@pytest.mark.asyncio
async def test_strategy_backtest(strategy, mock_exchange, sample_ohlcv_data):
    """Test du backtest de la stratégie."""
    # Créer des données de backtest
    data = pd.DataFrame(sample_ohlcv_data)
    data.set_index('timestamp', inplace=True)
    
    # Exécuter le backtest
    results = await strategy.backtest(
        data,
        initial_balance=10000,
        commission=0.001
    )
    
    assert 'trades' in results
    assert 'equity_curve' in results
    assert 'statistics' in results
    
    # Vérifier les statistiques de base
    stats = results['statistics']
    assert 'total_trades' in stats
    assert 'win_rate' in stats
    assert 'profit_factor' in stats
    assert 'max_drawdown' in stats
