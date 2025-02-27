#!/usr/bin/env python3
"""
Script de test pour l'intégration de Google Trends.
Ce script récupère les données de tendances Google pour le Bitcoin et
analyse la corrélation avec les mouvements de prix.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.data.market_data import MarketDataProvider, MarketDataConfig
from bitbot.utils.logger import logger


async def test_google_trends_client():
    """Test direct du client GoogleTrendsClient."""
    print("Test du client GoogleTrendsClient...")
    client = GoogleTrendsClient({})
    
    # Récupérer les données de tendances pour "bitcoin"
    trends_data = await client.get_interest_over_time("bitcoin", "today 3-m")
    if trends_data:
        print(f"Données de tendances récupérées pour 'bitcoin'")
        df = trends_data.get_normalized_interest()
        print(f"Nombre d'entrées: {len(df)}")
        print(df.head())
        
        # Récupérer les requêtes associées en hausse
        rising_queries = trends_data.get_rising_queries()
        if rising_queries:
            print("\nRequêtes associées en hausse:")
            for i, query in enumerate(rising_queries[:5]):
                print(f"  {i+1}. {query.get('query', 'N/A')} (+{query.get('value', 'N/A')}%)")
    else:
        print("Erreur: Impossible de récupérer les données de tendances")
    
    await client.close()


async def test_market_data_provider():
    """Test via le MarketDataProvider."""
    print("\nTest via le MarketDataProvider...")
    config = MarketDataConfig(
        data_dir="data",
        enable_google_trends=True,
        verify_ssl=False
    )
    provider = MarketDataProvider(config)
    
    # Récupérer les données de tendances
    trends_data = await provider.get_google_trends("bitcoin", "today 3-m")
    if trends_data:
        print(f"Données de tendances récupérées via MarketDataProvider")
        
        # Calculer les signaux de momentum
        momentum_df = trends_data.get_momentum_signal()
        print(f"Signal de momentum calculé:")
        print(momentum_df[['normalized_interest', 'momentum_signal']].tail())
    else:
        print("Erreur: Impossible de récupérer les données de tendances via MarketDataProvider")
    
    # Récupérer les corrélations avec les prix
    print("\nCalcul des corrélations entre les prix du Bitcoin et les recherches Google...")
    corr_df = await provider.get_bitcoin_price_trends_correlation(days=90)
    
    if not corr_df.empty:
        print(f"Corrélations calculées sur {len(corr_df)} points de données")
        
        # Identifier les périodes de forte corrélation
        strong_corr = corr_df[abs(corr_df['correlation_30d']) > 0.7].copy()
        if not strong_corr.empty:
            print("\nPériodes de forte corrélation (>0.7 ou <-0.7) sur 30 jours:")
            for _, row in strong_corr.iterrows():
                corr_value = row['correlation_30d']
                date = row['date_x'] if 'date_x' in row else row['date']
                price = row['close']
                print(f"  {date.strftime('%Y-%m-%d')}: Corrélation = {corr_value:.2f}, Prix = {price:.2f} USD")
        
        # Visualiser la corrélation
        try:
            plt.figure(figsize=(12, 10))
            
            # Premier sous-graphique: Prix du Bitcoin
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(corr_df['date_x'] if 'date_x' in corr_df.columns else corr_df['date'], 
                    corr_df['close'], 'b-', linewidth=2)
            ax1.set_title('Prix du Bitcoin (USD)')
            ax1.grid(True, alpha=0.3)
            
            # Deuxième sous-graphique: Intérêt Google Trends
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(corr_df['date_x'] if 'date_x' in corr_df.columns else corr_df['date'], 
                    corr_df['normalized_interest'], 'r-', linewidth=2)
            ax2.set_title("Intérêt de recherche Google pour 'bitcoin'")
            ax2.grid(True, alpha=0.3)
            
            # Troisième sous-graphique: Corrélation glissante
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(corr_df['date_x'] if 'date_x' in corr_df.columns else corr_df['date'], 
                   corr_df['correlation_30d'], 'g-', linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
            ax3.axhline(y=-0.7, color='red', linestyle='--', alpha=0.5)
            ax3.set_title('Corrélation glissante sur 30 jours')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-1, 1)
            
            plt.tight_layout()
            
            # Sauvegarder le graphique
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "bitcoin_google_trends_correlation.png"
            plt.savefig(output_file)
            print(f"\nGraphique sauvegardé dans {output_file}")
            
            # Sauvegarder les données
            csv_file = output_dir / "bitcoin_google_trends_correlation.csv"
            corr_df.to_csv(csv_file, index=False)
            print(f"Données sauvegardées dans {csv_file}")
            
        except Exception as e:
            print(f"Erreur lors de la création du graphique: {str(e)}")
    else:
        print("Erreur: Impossible de calculer les corrélations")
    
    await provider.close()


async def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test de l'intégration Google Trends")
    parser.add_argument("--client", action="store_true", help="Tester directement le client GoogleTrendsClient")
    parser.add_argument("--provider", action="store_true", help="Tester via le MarketDataProvider")
    args = parser.parse_args()
    
    if args.client or not (args.client or args.provider):
        await test_google_trends_client()
    
    if args.provider or not (args.client or args.provider):
        await test_market_data_provider()


if __name__ == "__main__":
    asyncio.run(main())
