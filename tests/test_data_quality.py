"""
Tests pour vérifier la qualité des données OHLCV.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

def verify_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Vérifie que les données OHLCV sont valides.
    
    Règles de validation :
    1. High >= Low
    2. High >= Open
    3. High >= Close
    4. Low <= Open
    5. Low <= Close
    6. Volume >= 0
    7. Pas de valeurs NaN ou infinies
    8. Pas de valeurs négatives
    9. Index temporel trié et sans doublons
    """
    if df.empty:
        return True
        
    # Vérifier les colonnes requises
    required_columns = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Vérifier l'index temporel
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if not df.index.is_monotonic_increasing:
        return False
    if df.index.has_duplicates:
        return False
    
    # Vérifier les valeurs NaN et infinies
    if df[required_columns].isna().any().any():
        return False
    if np.isinf(df[required_columns]).any().any():
        return False
    
    # Vérifier les valeurs négatives
    if (df[required_columns] < 0).any().any():
        return False
    
    # Vérifier les relations OHLC
    if not (df["high"] >= df["low"]).all():
        return False
    if not (df["high"] >= df["open"]).all():
        return False
    if not (df["high"] >= df["close"]).all():
        return False
    if not (df["low"] <= df["open"]).all():
        return False
    if not (df["low"] <= df["close"]).all():
        return False
    
    return True

@pytest.mark.asyncio
async def test_binance_data_quality(market_data_provider):
    """Vérifie la qualité des données OHLCV de Binance."""
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=1)
    
    df, source = await market_data_provider.binance.get_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_time=start_time,
        end_time=end_time
    )
    
    assert not df.empty, "Les données ne devraient pas être vides"
    assert verify_ohlcv_data(df), "Les données OHLCV ne sont pas valides"
    
    # Vérifier la fréquence des données
    time_diffs = df.index.to_series().diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all(), "La fréquence des données n'est pas correcte"
    
    # Vérifier les ordres de grandeur
    assert df["open"].between(1000, 100000).all(), "Prix hors limites raisonnables"
    assert df["volume"].between(0.1, 100000).all(), "Volume hors limites raisonnables"

@pytest.mark.asyncio
async def test_coingecko_data_quality(market_data_provider):
    """Vérifie la qualité des données OHLCV de CoinGecko."""
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=1)
    
    df, source = await market_data_provider.coingecko.get_klines(
        symbol="BTCUSDT",
        interval="30m",
        start_time=start_time,
        end_time=end_time
    )
    
    assert not df.empty, "Les données ne devraient pas être vides"
    assert verify_ohlcv_data(df), "Les données OHLCV ne sont pas valides"
    
    # Vérifier la fréquence des données
    time_diffs = df.index.to_series().diff().dropna()
    assert (time_diffs == pd.Timedelta(minutes=30)).all(), "La fréquence des données n'est pas correcte"
    
    # Vérifier les ordres de grandeur
    assert df["open"].between(1000, 100000).all(), "Prix hors limites raisonnables"
    assert df["volume"].between(0, 100000).all(), "Volume hors limites raisonnables"

@pytest.mark.asyncio
async def test_cached_data_quality(market_data_provider):
    """Vérifie la qualité des données OHLCV en cache."""
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=1)
    
    # D'abord, récupérer des données fraîches
    df_fresh, source = await market_data_provider.get_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_time=start_time,
        end_time=end_time
    )
    
    # Ensuite, récupérer les données du cache
    df_cached, source = await market_data_provider.get_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_time=start_time,
        end_time=end_time
    )
    
    # Vérifier que les deux DataFrames sont similaires (à 0.1% près)
    pd.testing.assert_frame_equal(
        df_fresh,
        df_cached,
        check_exact=False,
        rtol=0.001  # 0.1% de tolérance
    )
    
    # Vérifier la qualité des données
    assert verify_ohlcv_data(df_cached), "Les données OHLCV en cache ne sont pas valides"
