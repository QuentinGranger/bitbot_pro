#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'uniformisation des timestamps pour des données multi-sources.

Ce script démontre comment synchroniser des données provenant de différentes
sources avec des timestamps potentiellement décalés ou dans des fuseaux horaires différents.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
import logging
import sys
import os

# Ajouter le répertoire parent au chemin pour permettre les imports relatifs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from bitbot_pro.data_processing.data_cleaner import DataCleaner
from bitbot_pro.utils.performance import timeit

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@timeit
def main():
    """
    Exemple d'uniformisation des timestamps pour des données multi-sources.
    """
    # Initialiser le nettoyeur de données
    cleaner = DataCleaner(
        volatility_window=20,
        outlier_threshold=3.0,
        batch_size=50  # Augmenter la taille du lot pour de meilleures performances
    )
    
    # 1. Générer des données simulées de différentes sources avec des timestamps non alignés
    data_sources = generate_sample_data()
    
    # 2. Analyser les données avant uniformisation
    analyze_timestamps(data_sources, "avant uniformisation")
    
    # 3. Uniformiser les timestamps
    target_timeframe = "5m"  # Intervalle cible pour l'uniformisation
    logger.info(f"Uniformisation des timestamps en timeframe {target_timeframe} et fuseau horaire UTC...")
    
    standardized_data = cleaner.standardize_timestamps(
        data_frames=data_sources,
        target_timeframe=target_timeframe,
        align_to_intervals=True,
        ensure_utc=True
    )
    
    # 4. Analyser les données après uniformisation
    analyze_timestamps(standardized_data, "après uniformisation")
    
    # 5. Visualiser les résultats
    plot_timestamp_comparison(data_sources, standardized_data)


def generate_sample_data():
    """
    Génère des données simulées pour différentes sources avec des timestamps non alignés.
    
    Simule:
    - Source 1: Données en UTC avec intervalles irréguliers
    - Source 2: Données en fuseau EST (UTC-5) avec intervalles réguliers
    - Source 3: Données sans fuseau horaire avec timestamp légèrement décalés
    
    Returns:
        Dictionnaire de DataFrames indexés par source
    """
    # Point de départ: maintenant, arrondi à l'heure précédente
    now = pd.Timestamp.now().floor('H')
    
    # SOURCE 1: Données UTC avec intervalles irréguliers
    # Créer un index temporel légèrement irrégulier
    timestamps_utc = []
    current_time = now - pd.Timedelta(hours=24)  # Commencer 24h avant
    
    for _ in range(300):
        timestamps_utc.append(current_time)
        # Ajouter un intervalle de 5 minutes +/- une petite variation aléatoire
        offset = pd.Timedelta(minutes=5) + pd.Timedelta(seconds=np.random.randint(-30, 30))
        current_time = current_time + offset
    
    # Créer le DataFrame avec un index UTC
    df_source1 = pd.DataFrame({
        'price': np.cumsum(np.random.normal(0, 1, len(timestamps_utc))) + 100,
        'volume': np.abs(np.random.normal(1000, 300, len(timestamps_utc)))
    }, index=pd.DatetimeIndex(timestamps_utc).tz_localize('UTC'))
    
    # SOURCE 2: Données en fuseau EST (UTC-5) avec intervalles réguliers
    est_tz = pytz.timezone('America/New_York')
    start_time_est = now.tz_localize('UTC').tz_convert(est_tz) - pd.Timedelta(hours=24)
    
    # Créer un index régulier en EST
    timestamps_est = pd.date_range(
        start=start_time_est,
        periods=288,  # 24h x 12 (5-min intervals)
        freq='5min'
    )
    
    # Créer le DataFrame avec fuseau EST
    df_source2 = pd.DataFrame({
        'price': np.cumsum(np.random.normal(0, 1, len(timestamps_est))) + 110,
        'volume': np.abs(np.random.normal(1200, 350, len(timestamps_est)))
    }, index=timestamps_est)
    
    # SOURCE 3: Données sans fuseau horaire mais avec timestamps légèrement décalés
    start_time_naive = now - pd.Timedelta(hours=24)
    
    # Créer un index légèrement décalé (3 minutes)
    timestamps_naive = pd.date_range(
        start=start_time_naive + pd.Timedelta(minutes=3),
        periods=288,  # 24h x 12 (5-min intervals)
        freq='5min'
    )
    
    # Créer le DataFrame sans spécifier de fuseau horaire
    df_source3 = pd.DataFrame({
        'price': np.cumsum(np.random.normal(0, 1, len(timestamps_naive))) + 105,
        'volume': np.abs(np.random.normal(900, 250, len(timestamps_naive)))
    }, index=timestamps_naive)
    
    return {
        'Source_UTC': df_source1,
        'Source_EST': df_source2,
        'Source_NoTZ': df_source3
    }


def analyze_timestamps(data_sources, description):
    """
    Analyse les propriétés temporelles des données.
    
    Args:
        data_sources: Dictionnaire de DataFrames
        description: Description pour les logs (avant/après standardisation)
    """
    logger.info(f"Analyse des timestamps {description}:")
    
    for source, df in data_sources.items():
        if df.empty:
            logger.info(f"  - {source}: DataFrame vide")
            continue
        
        # Vérifier le type d'index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"  - {source}: Index non temporel")
            continue
        
        # Informations sur le fuseau horaire de manière sécurisée
        if df.index.tz is None:
            tz_info = "Pas de fuseau horaire"
        else:
            # Différentes façons d'obtenir le nom du fuseau horaire selon le type
            try:
                if hasattr(df.index.tz, 'zone'):
                    tz_info = df.index.tz.zone
                elif hasattr(df.index.tz, 'tzname'):
                    tz_info = df.index.tz.tzname(None)
                else:
                    tz_info = str(df.index.tz)
            except Exception:
                tz_info = str(df.index.tz)
        
        # Calculer les intervalles entre les points
        intervals = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
        avg_interval = intervals.mean().total_seconds() / 60  # en minutes
        std_interval = intervals.std().total_seconds() / 60  # en minutes
        
        # Analyse de régularité
        regularity = "régulier" if std_interval < 0.5 else "irrégulier"
        
        logger.info(f"  - {source}: {len(df)} points, fuseau: {tz_info}, "
                  f"intervalle moyen: {avg_interval:.2f} min ({regularity})")
        
        # Montrer la plage temporelle
        time_range = f"{df.index.min()} à {df.index.max()}"
        logger.info(f"    Plage temporelle: {time_range}")


def plot_timestamp_comparison(original_data, standardized_data):
    """
    Visualise la comparaison des timestamps avant et après uniformisation.
    
    Args:
        original_data: Dictionnaire de DataFrames avant uniformisation
        standardized_data: Dictionnaire de DataFrames après uniformisation
    """
    plt.figure(figsize=(14, 10))
    
    # Sous-figure 1: Prix avant uniformisation
    plt.subplot(2, 1, 1)
    plt.title('Prix par source avant uniformisation')
    
    for source, df in original_data.items():
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            # Convertir tous les timestamps en UTC pour l'affichage si nécessaire
            if df.index.tz is None:
                plot_idx = df.index.tz_localize('UTC')
            else:
                # Vérifier si déjà en UTC
                is_utc = False
                try:
                    if hasattr(df.index.tz, 'zone'):
                        is_utc = df.index.tz.zone == 'UTC'
                    elif hasattr(df.index.tz, 'tzname'):
                        tz_name = df.index.tz.tzname(None)
                        is_utc = tz_name == 'UTC' or tz_name == 'GMT'
                    else:
                        tz_str = str(df.index.tz)
                        is_utc = 'UTC' in tz_str or 'GMT' in tz_str
                except Exception:
                    is_utc = False
                    
                if is_utc:
                    plot_idx = df.index
                else:
                    plot_idx = df.index.tz_convert('UTC')
                
            plt.plot(plot_idx, df['price'], '.-', alpha=0.7, label=f"{source}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel('Prix')
    
    # Formatter l'axe X pour montrer date et heure
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M', tz=pytz.UTC))
    plt.gcf().autofmt_xdate()
    
    # Sous-figure 2: Prix après uniformisation
    plt.subplot(2, 1, 2)
    plt.title('Prix par source après uniformisation (UTC)')
    
    for source, df in standardized_data.items():
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            plt.plot(df.index, df['price'], '.-', alpha=0.7, label=f"{source}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel('Prix')
    
    # Formatter l'axe X pour montrer date et heure
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M', tz=pytz.UTC))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('timestamp_sync_results.png')
    logger.info("Graphique des résultats enregistré dans 'timestamp_sync_results.png'")


if __name__ == '__main__':
    main()
