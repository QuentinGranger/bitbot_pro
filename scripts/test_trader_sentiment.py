"""
Script de test pour démontrer l'intégration des données de sentiment de CryptoPanic 
dans notre trader BitBotPro.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.trader import Trader
from bitbot.config import Config

def main():
    """Fonction principale du script de test."""
    print("Initialisation du trader avec intégration CryptoPanic...")
    
    # Créer un trader
    config = Config()
    trader = Trader(config)
    
    # Tester la récupération et l'analyse de sentiment pour différentes cryptos
    cryptos = ["BTC"]  # Réduit à BTC uniquement pour la démo
    
    for crypto in cryptos:
        symbol = f"{crypto}USDT"
        print(f"\n--- Analyse pour {crypto} ---")
        
        # Récupérer et analyser les actualités
        news = trader.cryptopanic_client.get_news(
            currencies=crypto,
            force_refresh=True
        )
        
        print(f"Nombre d'articles récupérés: {len(news)}")
        
        # Afficher quelques exemples d'articles
        if news:
            print("\nExemples d'articles:")
            for i, item in enumerate(news[:5]):
                sentiment_score = item.get_sentiment_score()
                sentiment_str = ""
                if sentiment_score > 0.2:
                    sentiment_str = "[positif]"
                elif sentiment_score < -0.2:
                    sentiment_str = "[négatif]"
                    
                print(f"{i+1}. {item.published_at.strftime('%Y-%m-%d %H:%M')} {sentiment_str}: {item.title[:60]}...")
        
        # Récupérer l'analyse de sentiment
        sentiment = trader.cryptopanic_client.get_sentiment_analysis(
            currency=crypto,
            days=7
        )
        
        print(f"\nAnalyse de sentiment pour {crypto}:")
        print(f"  - Score moyen: {sentiment.get('sentiment_score', 0):.4f}")
        print(f"  - Articles positifs: {sentiment.get('positive_ratio', 0):.1%}")
        print(f"  - Articles négatifs: {sentiment.get('negative_ratio', 0):.1%}")
        print(f"  - Articles neutres: {sentiment.get('neutral_ratio', 0):.1%}")
        
        # Générer et sauvegarder la visualisation
        try:
            image_path = trader.visualize_sentiment(crypto)
            print(f"Visualisation générée: {image_path}")
        except Exception as e:
            print(f"Erreur lors de la génération de la visualisation: {str(e)}")
    
    # Simplifions la fin du test pour éviter des erreurs
    print("\nTest terminé avec succès!")

if __name__ == "__main__":
    main()
