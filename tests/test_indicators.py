"""
Tests des indicateurs techniques.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict

from bitbot.trading.indicators import TechnicalIndicators

@pytest.fixture
def sample_data(sample_ohlcv_data: List[Dict]) -> pd.DataFrame:
    """Convertit les données OHLCV en DataFrame."""
    df = pd.DataFrame(sample_ohlcv_data)
    df.set_index('timestamp', inplace=True)
    return df

def test_sma(sample_data):
    """Test du calcul de la moyenne mobile simple."""
    ti = TechnicalIndicators(sample_data)
    sma = ti.sma(period=2)
    
    expected = (sample_data['close'].iloc[0] + sample_data['close'].iloc[1]) / 2
    assert np.isclose(sma.iloc[-1], expected)

def test_ema(sample_data):
    """Test du calcul de la moyenne mobile exponentielle."""
    ti = TechnicalIndicators(sample_data)
    ema = ti.ema(period=2)
    
    # Vérifier que l'EMA est plus sensible aux prix récents que la SMA
    sma = ti.sma(period=2)
    latest_close = sample_data['close'].iloc[-1]
    
    assert abs(ema.iloc[-1] - latest_close) < abs(sma.iloc[-1] - latest_close)

def test_rsi(sample_data):
    """Test du calcul du RSI."""
    ti = TechnicalIndicators(sample_data)
    rsi = ti.rsi(period=2)
    
    # Vérifier que le RSI est entre 0 et 100
    assert 0 <= rsi.iloc[-1] <= 100
    
    # Vérifier que le RSI augmente avec une tendance haussière
    assert rsi.iloc[-1] > 50  # Car notre échantillon est haussier

def test_macd(sample_data):
    """Test du calcul du MACD."""
    ti = TechnicalIndicators(sample_data)
    macd, signal, hist = ti.macd()
    
    # Vérifier que l'histogramme est la différence entre MACD et Signal
    assert np.isclose(hist.iloc[-1], macd.iloc[-1] - signal.iloc[-1])

def test_bollinger_bands(sample_data):
    """Test du calcul des bandes de Bollinger."""
    ti = TechnicalIndicators(sample_data)
    upper, middle, lower = ti.bollinger_bands(period=2)
    
    # Vérifier que la bande moyenne est la SMA
    sma = ti.sma(period=2)
    assert np.isclose(middle.iloc[-1], sma.iloc[-1])
    
    # Vérifier que les bandes supérieure et inférieure encadrent la moyenne
    assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

def test_atr(sample_data):
    """Test du calcul de l'ATR."""
    ti = TechnicalIndicators(sample_data)
    atr = ti.atr(period=2)
    
    # Vérifier que l'ATR est positif
    assert atr.iloc[-1] > 0
    
    # Vérifier que l'ATR est inférieur à la différence High-Low maximale
    max_range = max(sample_data['high'] - sample_data['low'])
    assert atr.iloc[-1] <= max_range

def test_volume_sma(sample_data):
    """Test du calcul de la moyenne mobile du volume."""
    ti = TechnicalIndicators(sample_data)
    vol_sma = ti.volume_sma(period=2)
    
    expected = (sample_data['volume'].iloc[0] + sample_data['volume'].iloc[1]) / 2
    assert np.isclose(vol_sma.iloc[-1], expected)

def test_invalid_period(sample_data):
    """Test de la gestion des périodes invalides."""
    ti = TechnicalIndicators(sample_data)
    
    with pytest.raises(ValueError):
        ti.sma(period=0)
    
    with pytest.raises(ValueError):
        ti.sma(period=-1)
    
    with pytest.raises(ValueError):
        ti.sma(period=len(sample_data) + 1)
