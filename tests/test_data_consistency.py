"""
Tests de cohérence des données entre les différentes sources.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import List, Tuple

from bitbot.data.market_data import MarketDataProvider, MarketDataConfig
from bitbot.utils.logger import logger

@pytest.fixture
def market_data_provider():
    """Fixture pour le fournisseur de données."""
    config = MarketDataConfig()
    provider = MarketDataProvider(config)
    yield provider
    
    # Nettoyage
    import asyncio
    asyncio.run(provider.close())

@pytest.mark.asyncio
async def test_timestamp_consistency_between_sources(market_data_provider):
    """
    Vérifie la cohérence des timestamps entre les différentes sources.

    1. Récupère les données de la même période depuis chaque source
    2. Compare les timestamps pour s'assurer qu'ils sont cohérents
    """
    # Configuration
    provider = market_data_provider

    try:
        # Période de test (dernières 24h)
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)

        # Récupérer les données depuis chaque source
        sources_data = []

        # 1. Binance
        df_binance, source = await provider.binance.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        if not df_binance.empty:
            sources_data.append(("binance", df_binance))

        # 2. CoinGecko
        df_coingecko, source = await provider.coingecko.get_klines(
            symbol="BTCUSDT",
            interval="daily",
            start_time=start_time,
            end_time=end_time
        )
        if not df_coingecko.empty:
            sources_data.append(("coingecko", df_coingecko))

        # 3. CoinMarketCap (si configuré)
        if provider.coinmarketcap:
            df_cmc, source = await provider.coinmarketcap.get_klines(
                symbol="BTCUSDT",
                interval="daily",
                start_time=start_time,
                end_time=end_time
            )
            if not df_cmc.empty:
                sources_data.append(("coinmarketcap", df_cmc))

        # Vérifier qu'on a au moins une source
        assert len(sources_data) >= 1, "Pas de source disponible"

        # Si on a plusieurs sources, comparer les timestamps
        if len(sources_data) >= 2:
            # Comparer les timestamps entre chaque paire de sources
            for i in range(len(sources_data)):
                for j in range(i + 1, len(sources_data)):
                    source1_name, df1 = sources_data[i]
                    source2_name, df2 = sources_data[j]

                    # Vérifier que les timestamps sont cohérents
                    # On tolère une différence de 1 minute entre les sources
                    tolerance = pd.Timedelta(minutes=1)

                    # Trouver les timestamps communs
                    common_timestamps = df1.index.intersection(df2.index)
                    assert len(common_timestamps) > 0, f"Pas de timestamps communs entre {source1_name} et {source2_name}"

                    # Vérifier que les timestamps sont proches
                    for ts in common_timestamps:
                        ts1 = df1.index.get_loc(ts)
                        ts2 = df2.index.get_loc(ts)
                        diff = abs(df1.index[ts1] - df2.index[ts2])
                        assert diff <= tolerance, f"Différence de timestamp trop grande entre {source1_name} et {source2_name}: {diff}"

    except Exception as e:
        pytest.fail(f"Erreur lors du test: {str(e)}")

@pytest.mark.asyncio
async def test_timestamp_consistency_in_cache(market_data_provider):
    """
    Vérifie la cohérence des timestamps lors de la lecture/écriture du cache.

    1. Récupère des données depuis une source
    2. Les sauvegarde dans le cache
    3. Les relit depuis le cache
    4. Vérifie que les timestamps sont identiques
    """
    # Configuration
    provider = market_data_provider

    try:
        # Période de test (dernières 24h)
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=1)

        # 1. Récupérer des données depuis Binance
        df_original, source = await provider.binance.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )
        assert not df_original.empty, "Pas de données depuis Binance"

        # 2. Les données sont automatiquement mises en cache par le MarketDataProvider
        # Forcer une mise à jour pour être sûr d'avoir les dernières données
        df_updated, source = await provider.get_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )

        # 3. Lire depuis le cache
        df_cached = provider._get_cached_data(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time
        )

        # 4. Vérifier que les timestamps sont identiques
        assert not df_cached.empty, "Pas de données dans le cache"
        pd.testing.assert_index_equal(df_original.index, df_cached.index, check_exact=True)

    except Exception as e:
        pytest.fail(f"Erreur lors du test: {str(e)}")
