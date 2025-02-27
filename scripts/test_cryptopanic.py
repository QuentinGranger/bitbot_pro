#!/usr/bin/env python
"""
Script de test pour le client CryptoPanic.

Ce script récupère les actualités récentes pour le Bitcoin et effectue
une analyse de sentiment basée sur ces données.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from loguru import logger

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.cryptopanic_client import CryptoPanicClient
from bitbot.models.market_data import NewsCollection

def main():
    """Fonction principale du script de test."""
    try:
        # Initialiser le client CryptoPanic
        print("Initialisation du client CryptoPanic...")
        client = CryptoPanicClient()
        
        # Récupérer les actualités pour Bitcoin
        print("Récupération des actualités pour Bitcoin...")
        news_items = client.get_news(currencies='BTC', page=1)
        
        if not news_items:
            print("Aucune actualité trouvée. Vérifiez votre connexion internet ou l'API CryptoPanic.")
            return
        
        print(f"\nAperçu des actualités récupérées ({len(news_items)} articles):")
        for i, item in enumerate(news_items[:5]):
            sentiment = f" [{item.sentiment}]" if item.sentiment else ""
            print(f"{i+1}. {item.published_at.strftime('%Y-%m-%d %H:%M')}{sentiment}: {item.title[:80]}...")
        
        # Analyse de sentiment
        sentiment_analysis = client.get_sentiment_analysis(currency='BTC', days=7)
        
        print("\nAnalyse de sentiment pour Bitcoin (7 derniers jours):")
        print(f"  - Nombre d'articles: {sentiment_analysis['count']}")
        print(f"  - Score moyen: {sentiment_analysis['average_score']:.4f} (-1.0 à 1.0)")
        print(f"  - Articles positifs: {sentiment_analysis['positive_ratio']*100:.1f}%")
        print(f"  - Articles négatifs: {sentiment_analysis['negative_ratio']*100:.1f}%")
        print(f"  - Articles neutres: {sentiment_analysis['neutral_ratio']*100:.1f}%")
        
        # Signal de trading
        sentiment_signal = client.get_sentiment_signal(currency='BTC', days=7)
        print(f"\nSignal de sentiment: {sentiment_signal:.4f} (-1.0 à 1.0)")
        
        signal_interpretation = "neutre"
        if sentiment_signal > 0.5:
            signal_interpretation = "fortement haussier"
        elif sentiment_signal > 0.2:
            signal_interpretation = "modérément haussier"
        elif sentiment_signal < -0.5:
            signal_interpretation = "fortement baissier"
        elif sentiment_signal < -0.2:
            signal_interpretation = "modérément baissier"
            
        print(f"Interprétation: Le sentiment est {signal_interpretation}.")
        
        # Récupérer les sujets tendance
        print("\nSujets tendance dans les actualités crypto (7 derniers jours):")
        trending_topics = client.get_trending_topics(days=7)
        
        # Afficher les 10 premiers sujets
        for i, (currency, count) in enumerate(list(trending_topics.items())[:10]):
            print(f"{i+1}. {currency}: {count} mentions")
        
        # Visualiser le sentiment au fil du temps
        visualize_sentiment_over_time(client.news_collection)
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")
    
    print("\nTest terminé.")

def visualize_sentiment_over_time(news_collection: NewsCollection):
    """
    Visualise l'évolution du sentiment au fil du temps.
    
    Args:
        news_collection: Collection d'articles à analyser
    """
    # Trier les articles par date
    sorted_items = sorted(news_collection.news_items, key=lambda x: x.published_at)
    
    if len(sorted_items) < 5:
        print("Pas assez d'articles pour visualiser le sentiment au fil du temps.")
        return
    
    # Calculer le score de sentiment pour chaque article
    dates = [item.published_at for item in sorted_items]
    scores = [item.get_sentiment_score() for item in sorted_items]
    
    # Créer le graphique
    plt.figure(figsize=(15, 6))
    
    # Graphique linéaire
    plt.subplot(2, 1, 1)
    plt.plot(dates, scores, 'b-', alpha=0.5)
    plt.plot(dates, scores, 'bo', markersize=4)
    
    # Ajouter une courbe de tendance (moyenne mobile)
    window_size = min(15, len(scores)//3)
    if window_size > 1:
        rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        rolling_mean_dates = dates[window_size-1:]
        plt.plot(rolling_mean_dates, rolling_mean, 'r-', linewidth=2, label=f'Tendance (moyenne mobile sur {window_size} articles)')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Sentiment des actualités Bitcoin au fil du temps', fontsize=14)
    plt.ylabel('Score de sentiment (-1 à 1)', fontsize=12)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    
    # Configurer l'axe des x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Histogramme du sentiment
    plt.subplot(2, 1, 2)
    plt.hist(scores, bins=20, range=(-1, 1), alpha=0.7, color='blue')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=np.mean(scores), color='red', linestyle='-', alpha=0.7, label=f'Moyenne: {np.mean(scores):.3f}')
    plt.title('Distribution des scores de sentiment', fontsize=14)
    plt.xlabel('Score de sentiment', fontsize=12)
    plt.ylabel('Nombre d\'articles', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_dir = os.path.join('data', 'cryptopanic')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bitcoin_sentiment_analysis.png')
    plt.savefig(output_path)
    print(f"\nGraphique sauvegardé: {output_path}")
    
    # Afficher le graphique
    plt.close()

if __name__ == "__main__":
    main()
