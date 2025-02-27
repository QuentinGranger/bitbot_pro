"""
Script de test pour le module data_merger.py.

Ce script teste les fonctionnalités de synchronisation des timestamps
entre différentes sources de données.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import pytz

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.utils.data_merger import DataMerger
from bitbot.models.market_data import MarketData

# Créer un répertoire pour les sorties
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "data_merger")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_data(symbol: str, timeframe: str, start_date: datetime, 
                    end_date: datetime, jitter: bool = False, missing_pct: float = 0.0):
    """
    Crée des données de test avec des timestamps potentiellement décalés.
    
    Args:
        symbol: Symbole de la paire
        timeframe: Timeframe des données
        start_date: Date de début
        end_date: Date de fin
        jitter: Si True, ajoute un décalage aléatoire aux timestamps
        missing_pct: Pourcentage de données manquantes
        
    Returns:
        MarketData avec les données de test
    """
    # Convertir le timeframe en minutes
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('h'):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        minutes = int(timeframe[:-1]) * 60 * 24
    else:
        minutes = 60  # Par défaut 1h
    
    # Créer une séquence de dates
    delta = timedelta(minutes=minutes)
    dates = []
    current = start_date
    
    while current <= end_date:
        # Ajouter un jitter aléatoire si demandé
        if jitter:
            # Jitter entre -30% et +30% de l'intervalle
            jitter_seconds = random.uniform(-0.3, 0.3) * minutes * 60
            jittered_date = current + timedelta(seconds=jitter_seconds)
            dates.append(jittered_date)
        else:
            dates.append(current)
        
        current += delta
    
    # Supprimer aléatoirement des points si missing_pct > 0
    if missing_pct > 0:
        n_missing = int(len(dates) * missing_pct)
        if n_missing > 0:
            indices_to_remove = random.sample(range(len(dates)), n_missing)
            dates = [d for i, d in enumerate(dates) if i not in indices_to_remove]
    
    # Créer des données aléatoires
    n = len(dates)
    
    # Simuler un mouvement de prix réaliste
    price = 100.0
    prices = [price]
    
    for _ in range(n-1):
        change_pct = random.normalvariate(0, 0.01)  # Changement de prix avec distribution normale
        price *= (1 + change_pct)
        prices.append(price)
    
    # Créer le DataFrame
    data = {
        'open': prices,
        'high': [p * (1 + random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + random.uniform(-0.005, 0.005)) for p in prices],
        'volume': [random.uniform(100, 1000) for _ in range(n)]
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Créer l'objet MarketData
    market_data = MarketData(symbol, timeframe)
    market_data.ohlcv = df
    
    return market_data

def test_normalize_timestamps():
    """
    Teste la normalisation des timestamps.
    """
    print("\nTest de la normalisation des timestamps...")
    
    # Créer un DataMerger
    merger = DataMerger()
    
    # Créer des données avec des timestamps décalés
    start_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 2, tzinfo=pytz.UTC)
    
    # Données avec jitter
    market_data_jitter = create_test_data("BTCUSDT", "1h", start_date, end_date, jitter=True)
    
    # Normaliser les timestamps
    normalized_df = merger.normalize_timestamps(market_data_jitter.ohlcv, freq='1h')
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.title("Timestamps originaux avec jitter")
    plt.plot(market_data_jitter.ohlcv.index, market_data_jitter.ohlcv['close'], 'o-', label="Timestamps originaux")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title("Timestamps normalisés")
    plt.plot(normalized_df.index, normalized_df['close'], 'o-', label="Timestamps normalisés")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "normalize_timestamps.png"))
    plt.close()
    
    print(f"Nombre de points originaux: {len(market_data_jitter.ohlcv)}")
    print(f"Nombre de points normalisés: {len(normalized_df)}")
    print(f"Graphique sauvegardé: normalize_timestamps.png")

def test_merge_market_data():
    """
    Teste la fusion de données de marché de différentes sources.
    """
    print("\nTest de la fusion de données de marché...")
    
    # Créer un DataMerger
    merger = DataMerger()
    
    # Créer des données pour différents symboles avec des timestamps légèrement décalés
    start_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 7, tzinfo=pytz.UTC)
    
    btc_data = create_test_data("BTCUSDT", "4h", start_date, end_date, jitter=True, missing_pct=0.1)
    eth_data = create_test_data("ETHUSDT", "4h", start_date, end_date, jitter=True, missing_pct=0.15)
    
    # Fusionner les données
    merged_df = merger.merge_market_data([btc_data, eth_data], normalize=True)
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.title("BTC/USDT - Données originales")
    plt.plot(btc_data.ohlcv.index, btc_data.ohlcv['close'], 'o-', label="BTC/USDT")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title("ETH/USDT - Données originales")
    plt.plot(eth_data.ohlcv.index, eth_data.ohlcv['close'], 'o-', label="ETH/USDT")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title("Données fusionnées et synchronisées")
    plt.plot(merged_df.index, merged_df['BTCUSDT_close'], 'o-', label="BTC/USDT")
    plt.plot(merged_df.index, merged_df['ETHUSDT_close'], 'o-', label="ETH/USDT")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "merge_market_data.png"))
    plt.close()
    
    print(f"Nombre de points BTC originaux: {len(btc_data.ohlcv)}")
    print(f"Nombre de points ETH originaux: {len(eth_data.ohlcv)}")
    print(f"Nombre de points fusionnés: {len(merged_df)}")
    print(f"Graphique sauvegardé: merge_market_data.png")

def test_merge_with_alternative_data():
    """
    Teste la fusion de données de marché avec des données alternatives.
    """
    print("\nTest de la fusion avec des données alternatives...")
    
    # Créer un DataMerger
    merger = DataMerger()
    
    # Créer des données de marché
    start_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 7, tzinfo=pytz.UTC)
    
    market_data = create_test_data("BTCUSDT", "4h", start_date, end_date)
    
    # Créer des données alternatives (fréquence plus basse)
    alt_dates = []
    current = start_date
    while current <= end_date:
        alt_dates.append(current)
        current += timedelta(days=1)  # Données journalières
    
    # Données de sentiment
    sentiment_values = [random.uniform(-1, 1) for _ in range(len(alt_dates))]
    sentiment_df = pd.DataFrame({'sentiment': sentiment_values}, index=alt_dates)
    
    # Données de volume on-chain
    onchain_values = [random.uniform(1000, 10000) for _ in range(len(alt_dates))]
    onchain_df = pd.DataFrame({'volume': onchain_values}, index=alt_dates)
    
    # Fusionner les données
    alt_data = {
        'sentiment': sentiment_df,
        'onchain': onchain_df
    }
    
    merged_df = merger.merge_with_alternative_data(market_data, alt_data, normalize=True)
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 1, 1)
    plt.title("Prix BTC/USDT")
    plt.plot(market_data.ohlcv.index, market_data.ohlcv['close'], 'o-', label="Prix")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.title("Sentiment (données journalières)")
    plt.plot(sentiment_df.index, sentiment_df['sentiment'], 'o-', label="Sentiment")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.title("Volume on-chain (données journalières)")
    plt.plot(onchain_df.index, onchain_df['volume'], 'o-', label="Volume on-chain")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.title("Données fusionnées et synchronisées")
    
    # Tracer le prix
    ax1 = plt.gca()
    ax1.plot(merged_df.index, merged_df['BTCUSDT_close'], 'b-', label="Prix")
    ax1.set_ylabel('Prix', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Tracer le sentiment sur un axe secondaire
    ax2 = ax1.twinx()
    ax2.plot(merged_df.index, merged_df['sentiment_sentiment'], 'r-', label="Sentiment")
    ax2.set_ylabel('Sentiment', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Ajouter une légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "merge_with_alternative_data.png"))
    plt.close()
    
    print(f"Nombre de points de marché originaux: {len(market_data.ohlcv)}")
    print(f"Nombre de points de sentiment originaux: {len(sentiment_df)}")
    print(f"Nombre de points on-chain originaux: {len(onchain_df)}")
    print(f"Nombre de points fusionnés: {len(merged_df)}")
    print(f"Graphique sauvegardé: merge_with_alternative_data.png")

def test_align_multi_timeframe_data():
    """
    Teste l'alignement des données de différents timeframes.
    """
    print("\nTest de l'alignement multi-timeframe...")
    
    # Créer un DataMerger
    merger = DataMerger()
    
    # Créer des données pour différents timeframes
    start_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 7, tzinfo=pytz.UTC)
    
    data_1h = create_test_data("BTCUSDT", "1h", start_date, end_date)
    data_4h = create_test_data("BTCUSDT", "4h", start_date, end_date)
    data_1d = create_test_data("BTCUSDT", "1d", start_date, end_date)
    
    # Dictionnaire des données
    market_data_dict = {
        "1h": data_1h,
        "4h": data_4h,
        "1d": data_1d
    }
    
    # Aligner sur le timeframe 4h
    aligned_df = merger.align_multi_timeframe_data(market_data_dict, "4h")
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 1, 1)
    plt.title("Données 1h")
    plt.plot(data_1h.ohlcv.index, data_1h.ohlcv['close'], 'o-', label="1h")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.title("Données 4h")
    plt.plot(data_4h.ohlcv.index, data_4h.ohlcv['close'], 'o-', label="4h")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.title("Données 1d")
    plt.plot(data_1d.ohlcv.index, data_1d.ohlcv['close'], 'o-', label="1d")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.title("Données alignées sur 4h")
    plt.plot(aligned_df.index, aligned_df['1h_close'], 'o-', label="1h -> 4h")
    plt.plot(aligned_df.index, aligned_df['4h_close'], 'o-', label="4h")
    plt.plot(aligned_df.index, aligned_df['1d_close'], 'o-', label="1d -> 4h")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "align_multi_timeframe.png"))
    plt.close()
    
    print(f"Nombre de points 1h originaux: {len(data_1h.ohlcv)}")
    print(f"Nombre de points 4h originaux: {len(data_4h.ohlcv)}")
    print(f"Nombre de points 1d originaux: {len(data_1d.ohlcv)}")
    print(f"Nombre de points alignés: {len(aligned_df)}")
    print(f"Graphique sauvegardé: align_multi_timeframe.png")

def main():
    """
    Fonction principale qui exécute tous les tests.
    """
    print("Tests du module data_merger.py")
    print("=======================================")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print("=======================================\n")
    
    # Exécuter les tests
    test_normalize_timestamps()
    test_merge_market_data()
    test_merge_with_alternative_data()
    test_align_multi_timeframe_data()
    
    print("\nTests terminés. Tous les résultats ont été sauvegardés dans", OUTPUT_DIR)

if __name__ == "__main__":
    main()
