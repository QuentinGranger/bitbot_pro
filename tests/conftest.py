"""
Configuration globale des tests pour BitBot Pro.
"""

import os
import json
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from typing import Generator
from bitbot.data.market_data import MarketDataProvider
from bitbot.config import Config

# Configuration du mode test
os.environ["BITBOT_MODE"] = "test"
os.environ["BITBOT_DEBUG"] = "1"

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Retourne le chemin vers le répertoire des données de test."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def config() -> Config:
    """Fournit une configuration pour les tests."""
    return Config(
        binance_api_key="test_key",
        binance_api_secret="test_secret",
        data_dir=Path("data"),
        verify_ssl=False
    )

@pytest_asyncio.fixture
async def market_data_provider(config) -> MarketDataProvider:
    """Fournit une instance de MarketDataProvider pour les tests."""
    provider = MarketDataProvider(config)
    yield provider
    await provider.close()
