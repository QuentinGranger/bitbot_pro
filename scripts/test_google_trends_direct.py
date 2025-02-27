#!/usr/bin/env python3
"""
Script de test pour le client Google Trends simplifié.
Ce script génère des données simulées au lieu d'utiliser l'API Google Trends.
"""
import os
import sys
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.config import Config

class SimpleConfig:
    """Configuration simple pour les tests."""
    def __init__(self):
        self.data_dir = 'data'
        self.cache_duration = 86400  # 24 heures

async def test_google_trends():
    """Test du client Google Trends."""
    print("Initialisation du client Google Trends...")
    client = GoogleTrendsClient(SimpleConfig())
    
    try:
        # Récupérer les tendances pour Bitcoin
        print("Récupération des tendances pour Bitcoin...")
        bitcoin_trends = await client.get_bitcoin_trends(timeframe='today 5-y')
        
        if not bitcoin_trends or "bitcoin" not in bitcoin_trends:
            print("Erreur: Impossible de récupérer les tendances pour Bitcoin.")
            return
        
        # Extraire les données Bitcoin
        btc_data = bitcoin_trends["bitcoin"]
        trend_df = btc_data.data
        
        print("\nAperçu des données de tendances Google:")
        print(trend_df.head())
        
        # Créer des données de prix simulées
        # Simuler des prix Bitcoin qui suivent approximativement les tendances Google
        start_date = trend_df['date'].min()
        end_date = trend_df['date'].max()
        
        # Créer la série de prix avec des fluctuations réalistes
        price_base = 10000
        price_range = np.linspace(0, 10, len(trend_df))
        price_trend = price_base + 30000 * np.sin(price_range)  # Tendance de base
        
        # Ajouter des pics correspondants à des périodes d'intérêt importantes
        # Bull run 2021
        bull_run_mask = (trend_df['date'] >= pd.Timestamp('2020-12-01')) & (trend_df['date'] <= pd.Timestamp('2021-05-01'))
        if any(bull_run_mask):
            bull_run_indices = np.where(bull_run_mask)[0]
            bull_run_peak = np.linspace(0, np.pi, len(bull_run_indices))
            price_trend[bull_run_indices] += 50000 * np.sin(bull_run_peak)
        
        # Crash 2022
        crash_mask = (trend_df['date'] >= pd.Timestamp('2021-11-01')) & (trend_df['date'] <= pd.Timestamp('2022-06-01'))
        if any(crash_mask):
            crash_indices = np.where(crash_mask)[0]
            crash_peak = np.linspace(0, np.pi, len(crash_indices))
            price_trend[crash_indices] -= 30000 * np.sin(crash_peak)
        
        # Recovery 2023
        recovery_mask = (trend_df['date'] >= pd.Timestamp('2023-01-01')) & (trend_df['date'] <= pd.Timestamp('2023-06-01'))
        if any(recovery_mask):
            recovery_indices = np.where(recovery_mask)[0]
            recovery_peak = np.linspace(0, np.pi, len(recovery_indices))
            price_trend[recovery_indices] += 20000 * np.sin(recovery_peak)
        
        # 2024 halving et bull run
        halving_mask = (trend_df['date'] >= pd.Timestamp('2024-02-01')) & (trend_df['date'] <= pd.Timestamp('2024-06-01'))
        if any(halving_mask):
            halving_indices = np.where(halving_mask)[0]
            halving_peak = np.linspace(0, np.pi/2, len(halving_indices))
            price_trend[halving_indices] += 40000 * np.sin(halving_peak)
        
        # Ajouter du bruit
        noise = np.random.normal(0, 3000, len(trend_df))
        price_trend += noise
        
        # Assurer que les prix sont positifs et réalistes
        price_trend = np.maximum(price_trend, 1000)  # Minimum à 1000$
        
        # Créer le DataFrame de prix
        price_df = pd.DataFrame({
            'date': trend_df['date'],
            'open': price_trend * 0.99,  # Simuler OHLC
            'high': price_trend * 1.03,
            'low': price_trend * 0.97,
            'close': price_trend,
            'volume': np.random.randint(1000000, 50000000, size=len(trend_df))
        })
        
        print("\nAperçu des données de prix simulées:")
        print(price_df.head())
        
        # Calculer les corrélations
        print("\nCalcul des corrélations entre les prix et les tendances Google...")
        correlation_df = await client.get_bitcoin_correlations(price_df, btc_data)
        
        if correlation_df.empty:
            print("Erreur: Impossible de calculer les corrélations.")
            return
        
        print("\nAperçu des corrélations calculées:")
        # Montrer un aperçu qui inclut des données avec corrélation non-NaN
        # Trouver les premières lignes avec des valeurs non-NaN pour la corrélation
        valid_corr_rows = correlation_df.dropna(subset=['correlation_30d']).head(5)
        
        if not valid_corr_rows.empty:
            print("Premières lignes avec corrélations calculées:")
            print(valid_corr_rows[['date_x', 'close', 'normalized_interest', 'correlation_30d']].head())
        else:
            # Afficher les premières lignes si aucune corrélation n'est encore calculée
            print(correlation_df.head())
        
        # Analyser les corrélations
        if 'correlation_30d' in correlation_df.columns:
            corr_30d = correlation_df['correlation_30d'].dropna()
            print("\nAnalyse des corrélations sur 30 jours:")
            print(f"  - Nombre de points avec corrélation valide: {len(corr_30d)} sur {len(correlation_df)} ({len(corr_30d)/len(correlation_df)*100:.1f}%)")
            print(f"  - Moyenne: {corr_30d.mean():.4f}")
            print(f"  - Médiane: {corr_30d.median():.4f}")
            print(f"  - Max: {corr_30d.max():.4f}")
            print(f"  - Min: {corr_30d.min():.4f}")
            
            # Créer le répertoire de sortie s'il n'existe pas
            output_dir = os.path.join('data', 'google_trends')
            os.makedirs(output_dir, exist_ok=True)
            
            # Visualiser la corrélation glissante
            plt.figure(figsize=(15, 6))
            plt.plot(correlation_df['date_x'].iloc[30:], correlation_df['correlation_30d'].iloc[30:], 'g-', linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            plt.title('Corrélation glissante sur 30 jours entre le prix du Bitcoin et l\'intérêt Google Trends', fontsize=14)
            plt.ylabel('Coefficient de corrélation', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            
            # Configurer l'axe des x
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarder le graphique
            corr_path = os.path.join(output_dir, 'bitcoin_trends_rolling_correlation.png')
            plt.savefig(corr_path)
            print(f"Graphique de corrélation sauvegardé: {corr_path}")
        
        # Visualiser les résultats
        plt.figure(figsize=(15, 10))
        
        # Créer deux axes y
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Tracer le prix sur l'axe y principal
        ax1.plot(correlation_df['date_x'], correlation_df['close'], 'b-', alpha=0.7, linewidth=2, label='Prix BTC (USD)')
        ax1.set_ylabel('Prix Bitcoin (USD)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Tracer l'intérêt Google Trends sur l'axe y secondaire
        ax2.plot(correlation_df['date_x'], correlation_df['normalized_interest'], 'r-', alpha=0.7, linewidth=2, label='Intérêt Google Trends')
        ax2.set_ylabel('Intérêt Google Trends (0-100)', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Configurer l'axe des x
        ax1.set_xlabel('Date', fontsize=12)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Ajouter une légende
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        # Ajouter un titre
        plt.title('Comparaison entre le prix du Bitcoin et l\'intérêt Google Trends', fontsize=16)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = os.path.join(output_dir, 'bitcoin_trends_correlation.png')
        plt.savefig(output_path)
        print(f"\nGraphique sauvegardé: {output_path}")
        
        # Sauvegarder les données en CSV pour analyse ultérieure
        csv_path = os.path.join(output_dir, 'bitcoin_trends_correlation.csv')
        correlation_df.to_csv(csv_path, index=False)
        print(f"Données sauvegardées: {csv_path}")
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")
    
    finally:
        # Fermer le client
        await client.close()
        print("\nTest terminé.")

if __name__ == "__main__":
    # Utiliser asyncio pour exécuter la fonction asynchrone
    asyncio.run(test_google_trends())
