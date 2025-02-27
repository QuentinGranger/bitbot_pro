#!/usr/bin/env python3
"""
Script de test pour comparer différentes méthodes de lissage sur les données de crypto-monnaies.
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
from bitbot.utils.data_cleaner import (
    apply_kalman_filter, 
    apply_savgol_filter, 
    clean_market_data
)

# Configuration des tests
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "1h"]
METHODS = ["raw", "outlier_removal", "kalman", "savgol", "combined"]


def test_filtering_methods():
    """Test et comparaison des différentes méthodes de lissage."""
    
    # Créer une instance du trader
    trader = Trader()
    
    # Configurer un répertoire pour sauvegarder les graphiques
    output_dir = os.path.join(os.getcwd(), "outputs", "filters")
    os.makedirs(output_dir, exist_ok=True)
    
    # Pour chaque symbole et timeframe
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\nTest avec {symbol} ({timeframe}):")
            
            # Récupérer les données brutes
            raw_data = trader.update_market_data(symbol, timeframe, limit=200)
            
            # Créer un dictionnaire pour stocker différentes versions des données
            data_versions = {
                "raw": raw_data.ohlcv.copy(),
            }
            
            # 1. Appliquer uniquement la détection et correction d'outliers
            cleaned_data = clean_market_data(
                raw_data,
                std_threshold=3.0,
                window_size=20,
                method="interpolate",
                filter_type=None
            )
            data_versions["outlier_removal"] = cleaned_data.ohlcv.copy()
            
            # 2. Appliquer le filtre de Kalman
            kalman_data = clean_market_data(
                raw_data,
                std_threshold=3.0,
                window_size=20,
                method="interpolate",
                filter_type="kalman",
                filter_params={"process_variance": 1e-5, "measurement_variance": 0.1}
            )
            data_versions["kalman"] = kalman_data.ohlcv.copy()
            
            # 3. Appliquer le filtre Savitzky-Golay
            savgol_data = clean_market_data(
                raw_data,
                std_threshold=3.0,
                window_size=20,
                method="interpolate",
                filter_type="savgol",
                filter_params={"window_length": 15, "polyorder": 3}
            )
            data_versions["savgol"] = savgol_data.ohlcv.copy()
            
            # 4. Combiner détection d'outliers + Kalman
            combined_data = clean_market_data(
                raw_data,
                std_threshold=2.5,  # Plus strict pour les outliers
                window_size=20,
                method="interpolate",
                filter_type="kalman",
                filter_params={"process_variance": 1e-6, "measurement_variance": 0.05}
            )
            data_versions["combined"] = combined_data.ohlcv.copy()
            
            # Comparer les résultats graphiquement
            compare_filtering_methods(symbol, timeframe, data_versions, output_dir)
            
            # Analyser la qualité des lissages
            analyze_filtering_quality(symbol, timeframe, data_versions)


def compare_filtering_methods(symbol, timeframe, data_versions, output_dir):
    """
    Crée des graphiques comparant les différentes méthodes de lissage.
    
    Args:
        symbol: Symbole de la crypto-monnaie
        timeframe: Intervalle de temps
        data_versions: Dictionnaire contenant les différentes versions des données
        output_dir: Répertoire pour sauvegarder les graphiques
    """
    # Créer un graphique pour le prix de clôture
    plt.figure(figsize=(15, 10))
    
    # Palette de couleurs
    colors = {
        "raw": "lightgray",
        "outlier_removal": "blue",
        "kalman": "green",
        "savgol": "purple",
        "combined": "red"
    }
    
    # Labels
    labels = {
        "raw": "Données brutes",
        "outlier_removal": "Nettoyage d'outliers",
        "kalman": "Filtre de Kalman",
        "savgol": "Filtre Savitzky-Golay",
        "combined": "Combiné (Outliers + Kalman)"
    }
    
    # Prix de clôture
    plt.subplot(2, 1, 1)
    
    # Commencer par les données brutes
    plt.plot(data_versions["raw"].index, data_versions["raw"]["close"], 
             color=colors["raw"], alpha=0.5, linewidth=1, label=labels["raw"])
    
    # Ajouter les autres méthodes
    for method in ["outlier_removal", "kalman", "savgol", "combined"]:
        plt.plot(data_versions[method].index, data_versions[method]["close"], 
                 color=colors[method], linewidth=1.5, label=labels[method])
    
    plt.title(f'{symbol} ({timeframe}) - Prix de clôture avec différentes méthodes de lissage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Volume
    plt.subplot(2, 1, 2)
    
    # Commencer par les données brutes
    plt.plot(data_versions["raw"].index, data_versions["raw"]["volume"], 
             color=colors["raw"], alpha=0.5, linewidth=1, label=labels["raw"])
    
    # Ajouter les autres méthodes
    for method in ["outlier_removal", "kalman", "savgol", "combined"]:
        plt.plot(data_versions[method].index, data_versions[method]["volume"], 
                 color=colors[method], linewidth=1.5, label=labels[method])
    
    plt.title(f'{symbol} ({timeframe}) - Volume avec différentes méthodes de lissage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    filename = f"{symbol}_{timeframe}_filtering_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Graphique de comparaison sauvegardé: {filepath}")
    
    # Créer un autre graphique pour zoomer sur une section (20% du milieu)
    plt.figure(figsize=(15, 10))
    
    # Calculer l'index de début et de fin pour le zoom (milieu 20%)
    num_points = len(data_versions["raw"])
    start_idx = int(num_points * 0.4)
    end_idx = int(num_points * 0.6)
    
    # Prix de clôture zoomé
    plt.subplot(2, 1, 1)
    
    for method in ["raw", "outlier_removal", "kalman", "savgol", "combined"]:
        df = data_versions[method]
        subset = df.iloc[start_idx:end_idx]
        plt.plot(subset.index, subset["close"], 
                 color=colors[method], linewidth=2 if method != "raw" else 1,
                 alpha=0.5 if method == "raw" else 1.0,
                 label=labels[method])
    
    plt.title(f'{symbol} ({timeframe}) - Zoom sur le prix de clôture')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Volume zoomé
    plt.subplot(2, 1, 2)
    
    for method in ["raw", "outlier_removal", "kalman", "savgol", "combined"]:
        df = data_versions[method]
        subset = df.iloc[start_idx:end_idx]
        plt.plot(subset.index, subset["volume"], 
                 color=colors[method], linewidth=2 if method != "raw" else 1,
                 alpha=0.5 if method == "raw" else 1.0,
                 label=labels[method])
    
    plt.title(f'{symbol} ({timeframe}) - Zoom sur le volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique zoomé
    filename = f"{symbol}_{timeframe}_filtering_zoom.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Graphique zoomé sauvegardé: {filepath}")


def analyze_filtering_quality(symbol, timeframe, data_versions):
    """
    Analyse la qualité des méthodes de lissage en calculant différentes métriques.
    
    Args:
        symbol: Symbole de la crypto-monnaie
        timeframe: Intervalle de temps
        data_versions: Dictionnaire contenant les différentes versions des données
    """
    print(f"\nAnalyse de la qualité du lissage pour {symbol} ({timeframe}):")
    
    # Utiliser les données brutes comme référence
    raw_data = data_versions["raw"]
    
    # Calculer des métriques de lissage et de fidélité pour chaque méthode
    metrics = {}
    
    for method in ["outlier_removal", "kalman", "savgol", "combined"]:
        filtered_data = data_versions[method]
        
        # 1. Mesurer la volatilité (écart-type des retours)
        raw_returns = raw_data["close"].pct_change().dropna()
        filtered_returns = filtered_data["close"].pct_change().dropna()
        
        volatility_raw = raw_returns.std()
        volatility_filtered = filtered_returns.std()
        
        # 2. Calculer l'erreur quadratique moyenne (MSE)
        mse = ((raw_data["close"] - filtered_data["close"]) ** 2).mean()
        
        # 3. Calculer la préservation des tendances (corrélation des retours)
        correlation = raw_returns.corr(filtered_returns)
        
        # 4. Calculer la réduction du bruit (ratio signal/bruit)
        # Le SNR est approximé comme le rapport entre l'amplitude du signal et l'écart-type du bruit
        signal_amplitude = filtered_data["close"].max() - filtered_data["close"].min()
        noise_std = (raw_data["close"] - filtered_data["close"]).std()
        snr = signal_amplitude / noise_std if noise_std > 0 else float('inf')
        
        # Stocker les métriques
        metrics[method] = {
            "volatility_reduction": (1 - volatility_filtered / volatility_raw) * 100,
            "mse": mse,
            "trend_preservation": correlation * 100,
            "snr": snr
        }
    
    # Afficher les résultats
    print("\nMétriques de qualité des filtres:")
    print(f"{'Méthode':<20} {'Réduction de vol. (%)':<20} {'MSE':<15} {'Préservation tendances (%)':<25} {'SNR':<10}")
    print("-" * 90)
    
    for method, values in metrics.items():
        method_label = {
            "outlier_removal": "Nettoyage outliers",
            "kalman": "Filtre Kalman",
            "savgol": "Savitzky-Golay",
            "combined": "Combiné"
        }[method]
        
        print(f"{method_label:<20} {values['volatility_reduction']:>18.2f} {values['mse']:>15.6f} "
              f"{values['trend_preservation']:>23.2f} {values['snr']:>10.2f}")


if __name__ == "__main__":
    # Exécuter le test des méthodes de lissage
    test_filtering_methods()
