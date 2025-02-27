"""
Configuration globale des tests pour BitBot Pro.
"""

import os
import json
import pytest
import responses
from pathlib import Path
from typing import Generator

# Configuration du mode test
os.environ["BITBOT_MODE"] = "test"
os.environ["BITBOT_DEBUG"] = "1"

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Retourne le chemin vers le répertoire des données de test."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def mock_config() -> dict:
    """Configuration de test."""
    return {
        "exchanges": {
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True
            }
        },
        "trading": {
            "base_currency": "USDT",
            "trading_pairs": ["BTC/USDT", "ETH/USDT"],
            "position_size": 0.01,
            "max_positions": 3
        },
        "risk": {
            "max_drawdown": 0.02,
            "stop_loss": 0.01,
            "take_profit": 0.03
        }
    }

@pytest.fixture(scope="function")
def mock_responses() -> Generator:
    """Mock pour les réponses HTTP."""
    with responses.RequestsMock() as rsps:
        yield rsps

@pytest.fixture(scope="function")
def mock_binance_api(mock_responses):
    """Mock pour l'API Binance."""
    # Klines (OHLCV)
    mock_responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/klines",
        json=[
            [
                1499040000000,      # Open time
                "8100.0",           # Open
                "8200.0",           # High
                "8000.0",           # Low
                "8150.0",           # Close
                "10.0",             # Volume
                1499644799999,      # Close time
                "80000.0",          # Quote asset volume
                100,                # Number of trades
                "5.0",              # Taker buy base asset volume
                "40000.0",          # Taker buy quote asset volume
                "0"                 # Ignore
            ]
        ]
    )
    
    # Ticker
    mock_responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/ticker/24hr",
        json={
            "symbol": "BTCUSDT",
            "lastPrice": "8150.0",
            "volume": "10.0",
            "bidPrice": "8145.0",
            "askPrice": "8155.0"
        }
    )
    
    # Order Book
    mock_responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/depth",
        json={
            "lastUpdateId": 1027024,
            "bids": [
                ["8145.0", "1.0"],
                ["8144.0", "2.0"]
            ],
            "asks": [
                ["8155.0", "1.0"],
                ["8156.0", "2.0"]
            ]
        }
    )
    
    return mock_responses

@pytest.fixture(scope="function")
def mock_cryptopanic_api(mock_responses):
    """Mock pour l'API CryptoPanic."""
    mock_responses.add(
        responses.GET,
        "https://cryptopanic.com/api/v1/posts/",
        json={
            "results": [
                {
                    "kind": "news",
                    "title": "Bitcoin Tests $50,000",
                    "published_at": "2024-02-27T00:00:00Z",
                    "url": "https://example.com/news/1",
                    "currencies": [{"code": "BTC"}],
                    "votes": {"positive": 10, "negative": 2}
                }
            ]
        }
    )
    
    return mock_responses

@pytest.fixture(scope="function")
def sample_ohlcv_data():
    """Données OHLCV de test."""
    return [
        {
            "timestamp": 1499040000000,
            "open": 8100.0,
            "high": 8200.0,
            "low": 8000.0,
            "close": 8150.0,
            "volume": 10.0
        },
        {
            "timestamp": 1499040060000,
            "open": 8150.0,
            "high": 8300.0,
            "low": 8100.0,
            "close": 8250.0,
            "volume": 15.0
        }
    ]
