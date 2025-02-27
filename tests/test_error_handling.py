"""
Tests pour vérifier la gestion des erreurs API.
"""

import pytest
import aiohttp
import asyncio
from datetime import datetime, timedelta
import pytz
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_binance_timeout(market_data_provider):
    """
    Vérifie que les timeouts sont correctement gérés pour Binance.
    """
    provider = market_data_provider
    
    # Simuler un timeout
    with patch('aiohttp.ClientSession.get', side_effect=asyncio.TimeoutError):
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)
        
        df, source = await provider.binance.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons un DataFrame vide en cas de timeout
        assert df.empty
        assert source == "binance"

@pytest.mark.asyncio
async def test_binance_server_error(market_data_provider):
    """
    Vérifie que les erreurs serveur sont correctement gérées pour Binance.
    """
    provider = market_data_provider
    
    # Simuler une erreur serveur (500)
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"
    
    async def mock_get(*args, **kwargs):
        return mock_response
    
    with patch('aiohttp.ClientSession.get', new=mock_get):
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)
        
        df, source = await provider.binance.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons un DataFrame vide en cas d'erreur serveur
        assert df.empty
        assert source == "binance"

@pytest.mark.asyncio
async def test_binance_rate_limit(market_data_provider):
    """
    Vérifie que les erreurs de rate limit sont correctement gérées pour Binance.
    """
    provider = market_data_provider
    
    # Simuler une erreur de rate limit (429)
    mock_response = MagicMock()
    mock_response.status = 429
    mock_response.reason = "Too Many Requests"
    mock_response.headers = {"Retry-After": "5"}
    
    async def mock_get(*args, **kwargs):
        return mock_response
    
    with patch('aiohttp.ClientSession.get', new=mock_get):
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)
        
        df, source = await provider.binance.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons un DataFrame vide en cas de rate limit
        assert df.empty
        assert source == "binance"

@pytest.mark.asyncio
async def test_coingecko_timeout(market_data_provider):
    """
    Vérifie que les timeouts sont correctement gérés pour CoinGecko.
    """
    provider = market_data_provider
    
    # Simuler un timeout
    with patch('aiohttp.ClientSession.get', side_effect=asyncio.TimeoutError):
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)
        
        df, source = await provider.coingecko.get_klines(
            symbol="BTCUSDT",
            interval="daily",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons un DataFrame vide en cas de timeout
        assert df.empty
        assert source == "coingecko"

@pytest.mark.asyncio
async def test_coingecko_server_error(market_data_provider):
    """
    Vérifie que les erreurs serveur sont correctement gérées pour CoinGecko.
    """
    provider = market_data_provider
    
    # Simuler une erreur serveur (500)
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"
    
    async def mock_get(*args, **kwargs):
        return mock_response
    
    with patch('aiohttp.ClientSession.get', new=mock_get):
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)
        
        df, source = await provider.coingecko.get_klines(
            symbol="BTCUSDT",
            interval="daily",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons un DataFrame vide en cas d'erreur serveur
        assert df.empty
        assert source == "coingecko"

@pytest.mark.asyncio
async def test_fallback_to_cache(market_data_provider):
    """
    Vérifie que le système bascule sur le cache en cas d'erreur API.
    """
    provider = market_data_provider
    
    # D'abord, récupérer des données valides et les mettre en cache
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=1)
    
    df_original, source = await provider.get_klines(
        symbol="BTCUSDT",
        interval="1d",
        start_time=start_time,
        end_time=end_time
    )
    
    # Ensuite, simuler une erreur API
    with patch('aiohttp.ClientSession.get', side_effect=asyncio.TimeoutError):
        df_cached, source = await provider.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        
        # Vérifier que nous avons récupéré les données du cache
        assert not df_cached.empty
        assert source == "cache"
