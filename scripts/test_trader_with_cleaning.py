#!/usr/bin/env python3
"""
Script de test pour vérifier le fonctionnement du nettoyage des données dans le trader.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.trader import Trader
from bitbot.models.market_data import MarketData
from bitbot.utils.logger import logger


def test_trader_with_cleaning():
    """Test du trader avec le nettoyage des données intégré."""
    
    # Créer une instance du trader
    trader = Trader()
    
    # Symboles et timeframes à tester
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    timeframes = ["5m", "15m", "1h"]
    
    results = {}
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTest avec {symbol} ({timeframe}):")
            
            # Récupérer et nettoyer les données de marché
            market_data = trader.update_market_data(symbol, timeframe, limit=200)
            
            # Vérifier si le nettoyage a été effectué
            if 'cleaned' in market_data.metadata and market_data.metadata['cleaned']:
                print(f"✅ Nettoyage des données effectué")
                
                # Afficher les statistiques de nettoyage si disponibles
                if 'cleaning_stats' in market_data.metadata:
                    stats = market_data.metadata['cleaning_stats']
                    print(f"  - Outliers détectés: {stats.get('outliers_detected', 0)}")
                    print(f"  - Outliers corrigés: {stats.get('outliers_corrected', 0)}")
                    print(f"  - Pourcentage d'outliers: {stats.get('outlier_percentage', 0):.2f}%")
            else:
                print(f"❌ Nettoyage des données NON effectué")
            
            # Générer des signaux avec les données nettoyées
            signals = trader.generate_signals(market_data)
            
            if signals:
                print(f"📊 {len(signals)} signaux générés:")
                for signal in signals:
                    print(f"  - {signal}")
            else:
                print(f"⚠️ Aucun signal généré")
            
            # Stocker les résultats pour comparaison
            results[f"{symbol}_{timeframe}"] = {
                "cleaned": market_data.metadata.get('cleaned', False),
                "data_points": len(market_data.ohlcv),
                "signals": len(signals)
            }
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES TESTS ===")
    for key, result in results.items():
        print(f"{key}: {'✅ Nettoyé' if result['cleaned'] else '❌ Non nettoyé'}, "
              f"{result['data_points']} points, {result['signals']} signaux")
    
    # Sélectionner un exemple pour visualisation
    example_symbol = "BTCUSDT"
    example_timeframe = "1h"
    
    # Charger à nouveau les données
    clean_data = trader.update_market_data(example_symbol, example_timeframe, limit=100)
    
    # Charger les données sans nettoyage (en désactivant temporairement la fonction)
    # Sauvegarde de la référence à la fonction clean_market_data
    import bitbot.utils.data_cleaner
    original_clean_func = bitbot.utils.data_cleaner.clean_market_data
    
    # Remplacer temporairement par une fonction qui retourne les données non modifiées
    def dummy_clean(*args, **kwargs):
        return args[0]
    
    bitbot.utils.data_cleaner.clean_market_data = dummy_clean
    
    # Récupérer les données non nettoyées
    raw_data = trader.update_market_data(example_symbol, example_timeframe, limit=100)
    
    # Restaurer la fonction originale
    bitbot.utils.data_cleaner.clean_market_data = original_clean_func
    
    # Créer des graphiques pour visualiser la différence
    plt.figure(figsize=(15, 10))
    
    # Prix de clôture
    plt.subplot(2, 1, 1)
    plt.plot(raw_data.ohlcv.index, raw_data.ohlcv['close'], 'gray', alpha=0.7, label='Non nettoyé')
    plt.plot(clean_data.ohlcv.index, clean_data.ohlcv['close'], 'blue', label='Nettoyé')
    plt.title(f'{example_symbol} ({example_timeframe}) - Prix de clôture')
    plt.legend()
    
    # Volume
    plt.subplot(2, 1, 2)
    plt.plot(raw_data.ohlcv.index, raw_data.ohlcv['volume'], 'gray', alpha=0.7, label='Non nettoyé')
    plt.plot(clean_data.ohlcv.index, clean_data.ohlcv['volume'], 'green', label='Nettoyé')
    plt.title(f'{example_symbol} ({example_timeframe}) - Volume')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/{example_symbol}_{example_timeframe}_trader_cleaning_comparison.png")
    print(f"\nGraphique de comparaison sauvegardé dans outputs/{example_symbol}_{example_timeframe}_trader_cleaning_comparison.png")


if __name__ == "__main__":
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs("outputs", exist_ok=True)
    
    test_trader_with_cleaning()
