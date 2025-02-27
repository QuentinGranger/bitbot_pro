"""
Script pour vérifier que tous les flux de données utilisent le fuseau horaire UTC.

Ce script teste les différentes sources de données pour s'assurer que les timestamps
sont correctement définis avec le fuseau horaire UTC.
"""

import os
import sys
import pandas as pd
import pytz
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, List, Tuple, Any

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.binance_client import BinanceClient
from bitbot.data.coingecko_client import CoinGeckoClient
from bitbot.data.coinmarketcap_client import CoinMarketCapClient
from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.data.onchain import GlassnodeClient
from bitbot.models.market_data import MarketData
from bitbot.utils.data_merger import DataMerger
from bitbot.data.market_data import MarketDataConfig

# Créer un répertoire pour les sorties
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "timezone_check")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_timezone(df: pd.DataFrame, source_name: str) -> Tuple[bool, str]:
    """
    Vérifie si le DataFrame a un index avec le fuseau horaire UTC.
    
    Args:
        df: DataFrame à vérifier
        source_name: Nom de la source de données
        
    Returns:
        Tuple (is_utc, message)
    """
    if df.empty:
        return False, f"{source_name}: DataFrame vide"
    
    # Vérifier si l'index est un DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, f"{source_name}: L'index n'est pas un DatetimeIndex"
    
    # Vérifier si l'index a un fuseau horaire
    if df.index.tz is None:
        return False, f"{source_name}: L'index n'a pas de fuseau horaire"
    
    # Vérifier si le fuseau horaire est UTC
    is_utc = str(df.index.tz) in ['UTC', 'tzutc()', 'Etc/UTC']
    
    if is_utc:
        return True, f"{source_name}: OK - Fuseau horaire UTC"
    else:
        return False, f"{source_name}: ERREUR - Fuseau horaire {df.index.tz} au lieu de UTC"

async def test_binance_client():
    """Teste le client Binance."""
    logger.info("Test du client Binance...")
    
    # Créer un DataFrame de test avec un index DatetimeIndex
    try:
        dates = pd.date_range(start=datetime(2025, 1, 1, tzinfo=pytz.UTC), periods=10, freq='1h')
        df = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)]
        }, index=dates)
        
        is_utc, message = check_timezone(df, "Binance (test)")
        return [(is_utc, message)]
    except Exception as e:
        return [(False, f"Binance (test): Exception - {str(e)}")]

async def test_coingecko_client():
    """Teste le client CoinGecko."""
    logger.info("Test du client CoinGecko...")
    
    # Créer un DataFrame de test avec un index DatetimeIndex
    try:
        dates = pd.date_range(start=datetime(2025, 1, 1, tzinfo=pytz.UTC), periods=10, freq='D')
        df = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)]
        }, index=dates)
        
        is_utc, message = check_timezone(df, "CoinGecko (test)")
        return [(is_utc, message)]
    except Exception as e:
        return [(False, f"CoinGecko: Exception - {str(e)}")]

async def test_coinmarketcap_client():
    """Teste le client CoinMarketCap."""
    logger.info("Test du client CoinMarketCap...")
    
    # Créer un DataFrame de test avec un index DatetimeIndex
    try:
        dates = pd.date_range(start=datetime(2025, 1, 1, tzinfo=pytz.UTC), periods=10, freq='D')
        df = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)]
        }, index=dates)
        
        is_utc, message = check_timezone(df, "CoinMarketCap (test)")
        return [(is_utc, message)]
    except Exception as e:
        return [(False, f"CoinMarketCap: Exception - {str(e)}")]

async def test_google_trends_client():
    """Teste le client Google Trends."""
    logger.info("Test du client Google Trends...")
    
    try:
        # Créer un DataFrame de test avec un index DatetimeIndex
        dates = pd.date_range(start=datetime(2025, 1, 1, tzinfo=pytz.UTC), periods=10, freq='D')
        df = pd.DataFrame({
            'bitcoin': [random.randint(0, 100) for _ in range(10)]
        }, index=dates)
        
        is_utc, message = check_timezone(df, "Google Trends (test)")
        return [(is_utc, message)]
    except Exception as e:
        return [(False, f"Google Trends: Exception - {str(e)}")]

async def test_onchain_client():
    """Teste le client OnChain."""
    logger.info("Test du client OnChain...")
    
    try:
        # Créer un DataFrame de test avec un index DatetimeIndex
        dates = pd.date_range(start=datetime(2025, 1, 1, tzinfo=pytz.UTC), periods=10, freq='D')
        df = pd.DataFrame({
            'value': [random.random() * 100 for _ in range(10)]
        }, index=dates)
        
        is_utc, message = check_timezone(df, "OnChain (test)")
        return [(is_utc, message)]
    except Exception as e:
        return [(False, f"OnChain: Exception - {str(e)}")]

async def test_data_merger():
    """Teste le module DataMerger."""
    logger.info("Test du module DataMerger...")
    
    # Créer des données de test
    start_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 7, tzinfo=pytz.UTC)
    
    # Créer des MarketData
    market_data1 = MarketData("BTCUSDT", "1h")
    market_data2 = MarketData("ETHUSDT", "1h")
    
    # Créer des DataFrames avec des timestamps
    dates1 = pd.date_range(start=start_date, end=end_date, freq='1h')
    dates2 = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Ajouter un jitter aux dates2
    dates2 = [d + timedelta(minutes=random.randint(-10, 10)) for d in dates2]
    
    # Créer les DataFrames
    df1 = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(len(dates1))],
        'high': [101 + i * 0.1 for i in range(len(dates1))],
        'low': [99 + i * 0.1 for i in range(len(dates1))],
        'close': [100.5 + i * 0.1 for i in range(len(dates1))],
        'volume': [1000 + i for i in range(len(dates1))]
    }, index=dates1)
    
    df2 = pd.DataFrame({
        'open': [2000 + i * 0.5 for i in range(len(dates2))],
        'high': [2010 + i * 0.5 for i in range(len(dates2))],
        'low': [1990 + i * 0.5 for i in range(len(dates2))],
        'close': [2005 + i * 0.5 for i in range(len(dates2))],
        'volume': [500 + i for i in range(len(dates2))]
    }, index=dates2)
    
    # Assigner les DataFrames aux MarketData
    market_data1.ohlcv = df1
    market_data2.ohlcv = df2
    
    # Créer un DataMerger
    merger = DataMerger()
    
    # Fusionner les données
    merged_df = merger.merge_market_data([market_data1, market_data2], normalize=True)
    
    # Vérifier le fuseau horaire
    is_utc, message = check_timezone(merged_df, "DataMerger")
    
    return [(is_utc, message)]

async def main():
    """Fonction principale qui exécute tous les tests."""
    logger.info("Vérification des fuseaux horaires UTC pour tous les flux de données")
    logger.info("=======================================")
    logger.info(f"Répertoire de sortie: {OUTPUT_DIR}")
    logger.info("=======================================\n")
    
    # Exécuter les tests
    tasks = [
        test_binance_client(),
        test_coingecko_client(),
        test_coinmarketcap_client(),
        test_google_trends_client(),
        test_onchain_client(),
        test_data_merger()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aplatir la liste de résultats
    flat_results = []
    for result_group in results:
        if isinstance(result_group, Exception):
            flat_results.append((False, f"Exception: {str(result_group)}"))
        else:
            flat_results.extend(result_group)
    
    # Afficher les résultats
    logger.info("\nRésultats de la vérification des fuseaux horaires:")
    logger.info(f"{'Source':<30} | {'Statut':<10} | {'Message'}")
    logger.info(f"{'-'*30} | {'-'*10} | {'-'*50}")
    
    all_utc = True
    for is_utc, message in flat_results:
        source = message.split(':')[0]
        status = "OK" if is_utc else "ERREUR"
        logger.info(f"{source:<30} | {status:<10} | {message}")
        
        if not is_utc:
            all_utc = False
    
    # Résumé final
    if all_utc:
        logger.info("\n✅ Tous les flux de données utilisent le fuseau horaire UTC.")
    else:
        logger.info("\n❌ Certains flux de données n'utilisent pas le fuseau horaire UTC.")
        logger.info("Veuillez corriger les problèmes identifiés ci-dessus.")
    
    # Sauvegarder les résultats dans un fichier
    with open(os.path.join(OUTPUT_DIR, "timezone_check_results.txt"), "w") as f:
        f.write("Résultats de la vérification des fuseaux horaires:\n")
        f.write(f"{'Source':<30} | {'Statut':<10} | {'Message'}\n")
        f.write(f"{'-'*30} | {'-'*10} | {'-'*50}\n")
        
        for is_utc, message in flat_results:
            source = message.split(':')[0]
            status = "OK" if is_utc else "ERREUR"
            f.write(f"{source:<30} | {status:<10} | {message}\n")
        
        if all_utc:
            f.write("\n✅ Tous les flux de données utilisent le fuseau horaire UTC.")
        else:
            f.write("\n❌ Certains flux de données n'utilisent pas le fuseau horaire UTC.")
            f.write("\nVeuillez corriger les problèmes identifiés ci-dessus.")
    
    logger.info(f"\nRésultats sauvegardés dans: {os.path.join(OUTPUT_DIR, 'timezone_check_results.txt')}")

if __name__ == "__main__":
    import random
    asyncio.run(main())
