#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du SignalAggregator pour agréger des signaux de trading
provenant de différentes sources et catégories.

Ce script:
1. Initialise un SignalAggregator
2. Enregistre plusieurs signaux provenant de différentes stratégies
3. Définit un contexte de marché
4. Agrège les signaux en tenant compte du contexte
5. Affiche les résultats et visualise la contribution des différentes catégories
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.strategie.aggregation.signal_aggregator import (
    SignalAggregator, Signal, SignalCategory, SignalRecommendation
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('signal_aggregator_example')

def generate_sample_market_data(days=30, volatility=0.02):
    """Générer des données de marché synthétiques pour l'exemple."""
    np.random.seed(42)  # Pour la reproductibilité
    
    # Date de début
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Générer un prix qui suit une marche aléatoire
    price = 10000.0  # Prix initial
    closes = [price]
    
    for i in range(1, days):
        # Mouvement aléatoire avec tendance légèrement positive
        change = np.random.normal(0.001, volatility)
        price *= (1 + change)
        closes.append(price)
    
    # Générer high, low, open à partir des close
    highs = [close * (1 + abs(np.random.normal(0, volatility/2))) for close in closes]
    lows = [close * (1 - abs(np.random.normal(0, volatility/2))) for close in closes]
    opens = [low + (high - low) * np.random.random() for high, low in zip(highs, lows)]
    
    # Générer le volume
    volumes = [np.random.gamma(2.0, 1000000) for _ in range(days)]
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    data.set_index('date', inplace=True)
    return data

def create_signal_from_strategy(name, category, score, confidence):
    """Créer un signal factice à partir d'une stratégie."""
    return Signal(
        name=name,
        score=score,
        timestamp=time.time(),
        source="simulated",
        category=category,
        confidence=confidence,
        metadata={
            'strategy': name,
            'version': '1.0',
            'simulated': True
        }
    )

def plot_signal_categories(signal_data):
    """Visualiser la contribution des différentes catégories de signaux."""
    # Extraire les poids des catégories
    categories = []
    weights = []
    
    for cat, weight in signal_data['category_weights'].items():
        categories.append(cat)
        weights.append(weight)
    
    # Créer un graphique à barres
    plt.figure(figsize=(12, 6))
    
    # Barres des poids des catégories
    bars = plt.bar(categories, weights, color='skyblue')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Poids des catégories de signaux')
    plt.xlabel('Catégorie')
    plt.ylabel('Poids')
    plt.ylim(0, max(weights) * 1.2)  # Espace pour les étiquettes
    
    plt.tight_layout()
    plt.savefig('signal_categories_weights.png')
    plt.close()
    
    # Créer un graphique des scores par catégorie
    components = signal_data['components_by_category']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    category_colors = {
        'TECHNICAL': 'royalblue',
        'SENTIMENT': 'lightcoral',
        'VOLATILITY': 'mediumseagreen',
        'ORDER_BOOK': 'gold',
        'ON_CHAIN': 'purple'
    }
    
    for i, category in enumerate(components.keys()):
        signals = components[category]
        signal_names = [s['name'] for s in signals]
        signal_scores = [s['score'] for s in signals]
        
        # Position des barres
        x = np.arange(len(signal_names))
        width = 0.8
        
        # Couleur de la catégorie
        color = category_colors.get(category, 'gray')
        
        # Créer les barres
        bars = ax.bar(x + i*width*1.2, signal_scores, width, label=category, color=color, alpha=0.7)
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars, signal_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        # Ajouter les noms des signaux
        for j, name in enumerate(signal_names):
            ax.text(x[j] + i*width*1.2, -5, name, ha='center', va='top', rotation=90, fontsize=8)
    
    # Ligne du score neutre (50)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Score neutre')
    
    # Ligne du score agrégé
    ax.axhline(y=signal_data['score'], color='green', linestyle='-', alpha=0.7, 
              label=f'Score agrégé: {signal_data["score"]:.1f}')
    
    ax.set_title('Scores des signaux par catégorie')
    ax.set_ylabel('Score (0-100)')
    ax.set_ylim(0, 105)  # Pour laisser de la place aux étiquettes
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('signal_components_scores.png')
    plt.close()
    
    logger.info("Graphiques générés: signal_categories_weights.png et signal_components_scores.png")

def main():
    """
    Fonction principale de démonstration de l'agrégateur de signaux.
    """
    logger.info("Initialisation de l'agrégateur de signaux...")
    
    # Initialiser l'agrégateur avec des poids par défaut
    aggregator = SignalAggregator(
        default_weights={
            SignalCategory.TECHNICAL: 0.45,
            SignalCategory.SENTIMENT: 0.15,
            SignalCategory.VOLATILITY: 0.15,
            SignalCategory.ORDER_BOOK: 0.15,
            SignalCategory.ON_CHAIN: 0.10
        },
        signal_expiry=3600,  # 1 heure
        fallbacks_enabled=True
    )
    
    # Générer des données de marché
    logger.info("Génération des données de marché...")
    market_data = generate_sample_market_data(days=30, volatility=0.02)
    
    # Analyser le contexte du marché
    logger.info("Analyse du contexte du marché...")
    context = aggregator.analyze_market_context(market_data.tail(14))
    
    logger.info(f"Contexte détecté: Volatilité={context['volatility']:.4f}, "
               f"Changement de volume={context['volume_change']:.2f}, "
               f"Marché latéral={context['sideways']}")
    
    # Créer et ajouter plusieurs signaux fictifs provenant de différentes stratégies
    logger.info("Ajout des signaux provenant des différentes stratégies...")
    
    # Signaux techniques
    aggregator.add_signal(create_signal_from_strategy(
        "SMA_Crossover", SignalCategory.TECHNICAL, 65.0, 0.75))
    aggregator.add_signal(create_signal_from_strategy(
        "EMA_Crossover", SignalCategory.TECHNICAL, 70.0, 0.8))
    aggregator.add_signal(create_signal_from_strategy(
        "MACD_Divergence", SignalCategory.TECHNICAL, 62.0, 0.7))
    aggregator.add_signal(create_signal_from_strategy(
        "RSI", SignalCategory.TECHNICAL, 58.0, 0.65))
    
    # Signaux de sentiment
    aggregator.add_signal(create_signal_from_strategy(
        "Social_Media_Sentiment", SignalCategory.SENTIMENT, 72.0, 0.6))
    aggregator.add_signal(create_signal_from_strategy(
        "News_Sentiment", SignalCategory.SENTIMENT, 68.0, 0.55))
    
    # Signaux de volatilité
    aggregator.add_signal(create_signal_from_strategy(
        "Volatility_Breakout", SignalCategory.VOLATILITY, 55.0, 0.7))
    
    # Signaux du carnet d'ordres
    aggregator.add_signal(create_signal_from_strategy(
        "Orderbook_Imbalance", SignalCategory.ORDER_BOOK, 44.0, 0.8))
    
    # Signaux on-chain (uniquement pour les crypto-monnaies)
    aggregator.add_signal(create_signal_from_strategy(
        "Whale_Transactions", SignalCategory.ON_CHAIN, 60.0, 0.6))
    
    # Agréger les signaux
    logger.info("Agrégation des signaux...")
    aggregated_signal = aggregator.aggregate_signals()
    
    # Obtenir les informations sur le signal actuel
    signal_data = aggregator.get_current_signal()
    
    # Afficher les résultats
    logger.info(f"Signal agrégé: Score={signal_data['score']:.2f}, "
               f"Recommandation={signal_data['recommendation']}, "
               f"Confiance={signal_data['confidence']:.2f}")
    
    logger.info(f"Nombre total de signaux: {signal_data['signals_count']}")
    
    # Afficher le poids de chaque catégorie
    logger.info("Poids des catégories après ajustement dynamique:")
    for category, weight in signal_data['category_weights'].items():
        logger.info(f"  {category}: {weight:.2f}")
    
    # Visualiser les résultats
    logger.info("Génération des visualisations...")
    plot_signal_categories(signal_data)
    
    logger.info("Exemple terminé. Vérifiez les fichiers graphiques générés.")

if __name__ == "__main__":
    main()
