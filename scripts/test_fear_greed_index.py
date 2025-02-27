#!/usr/bin/env python3
"""
Script de test pour l'indice de peur et d'avidité.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitbot.data.alternative_me_client import AlternativeMeClient
from bitbot.data.market_data import MarketDataProvider, MarketDataConfig


async def test_alternative_me_client():
    """Test direct du client AlternativeMeClient."""
    print("Test du client AlternativeMeClient...")
    client = AlternativeMeClient({})
    
    # Récupérer l'indice actuel
    latest = await client.get_latest_fear_greed_index()
    if latest:
        print(f"Indice actuel: {latest.value} ({latest.classification})")
        print(f"Date: {latest.timestamp}")
        print(f"Score de sentiment: {latest.get_sentiment_score():.2f}")
        print(f"Temps jusqu'à la prochaine mise à jour: {latest.time_until_update} secondes")
    else:
        print("Erreur: Impossible de récupérer l'indice actuel")
    
    # Récupérer l'historique sur 30 jours
    indices = await client.get_fear_greed_index(limit=30)
    if indices:
        print(f"\nHistorique sur 30 jours ({len(indices)} entrées):")
        for idx, index in enumerate(indices[:5]):
            print(f"  {idx+1}. {index.timestamp.date()}: {index.value} ({index.classification})")
        print("  ...")
    else:
        print("Erreur: Impossible de récupérer l'historique")
    
    # Récupérer les données sous forme de DataFrame
    df = await client.get_fear_greed_dataframe(days=30)
    if not df.empty:
        print(f"\nDataFrame: {len(df)} lignes")
        print(df.head())
    else:
        print("Erreur: Impossible de récupérer le DataFrame")
    
    await client.close()


async def test_market_data_provider():
    """Test via le MarketDataProvider."""
    print("\nTest via le MarketDataProvider...")
    config = MarketDataConfig(
        data_dir="data",
        enable_fear_greed_index=True
    )
    provider = MarketDataProvider(config)
    
    # Récupérer l'indice actuel
    latest = await provider.get_latest_fear_greed_index()
    if latest:
        print(f"Indice actuel: {latest.value} ({latest.classification})")
    else:
        print("Erreur: Impossible de récupérer l'indice actuel")
    
    # Récupérer l'analyse du sentiment
    sentiment = await provider.get_market_sentiment()
    if sentiment:
        print("\nAnalyse du sentiment:")
        print(f"  Valeur: {sentiment['value']}")
        print(f"  Classification: {sentiment['classification']}")
        print(f"  Score: {sentiment['sentiment_score']:.2f}")
        print(f"  Phase du marché: {sentiment['market_phase']}")
        print(f"  Action recommandée: {sentiment['recommended_action']}")
    else:
        print("Erreur: Impossible de récupérer l'analyse du sentiment")
    
    # Récupérer l'historique
    df = await provider.get_fear_greed_history(days=90)
    if not df.empty:
        print(f"\nHistorique sur 90 jours: {len(df)} entrées")
        
        # Dessiner un graphique
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['value'], 'b-', linewidth=2)
        plt.fill_between(df['date'], df['value'], alpha=0.2)
        plt.axhline(y=25, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=50, color='y', linestyle='--', alpha=0.5)
        plt.axhline(y=75, color='g', linestyle='--', alpha=0.5)
        plt.title('Crypto Fear & Greed Index (90 jours)')
        plt.ylabel('Indice (0-100)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "fear_greed_index.png"
        plt.savefig(output_file)
        print(f"Graphique sauvegardé dans {output_file}")
    else:
        print("Erreur: Impossible de récupérer l'historique")
    
    await provider.close()


async def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test de l'indice de peur et d'avidité")
    parser.add_argument("--client", action="store_true", help="Tester directement le client AlternativeMeClient")
    parser.add_argument("--provider", action="store_true", help="Tester via le MarketDataProvider")
    args = parser.parse_args()
    
    if args.client or not (args.client or args.provider):
        await test_alternative_me_client()
    
    if args.provider or not (args.client or args.provider):
        await test_market_data_provider()


if __name__ == "__main__":
    asyncio.run(main())
