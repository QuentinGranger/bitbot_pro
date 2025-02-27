#!/usr/bin/env python3
"""
Script de test pour le module MVRV (Market Value to Realized Value).
Ce script récupère les données MVRV via des API gratuites et affiche les résultats.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.strategie.base.MVRVRatio import MVRVIndicator, MVRVSignal
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module MVRV."""
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description='Test du module MVRV (Market Value to Realized Value)')
    parser.add_argument('--days', type=int, default=365, help='Nombre de jours de données à récupérer')
    parser.add_argument('--asset', type=str, default='BTC', help='Actif à analyser')
    parser.add_argument('--ema-period', type=int, default=50, help='Période pour le calcul de l\'EMA')
    parser.add_argument('--save-path', type=str, help='Chemin pour enregistrer le graphique')
    
    args = parser.parse_args()
    
    logger.info("Test du module MVRV (Market Value to Realized Value)")
    
    # Initialiser l'indicateur MVRV
    mvrv_indicator = MVRVIndicator(
        ema_period=args.ema_period
    )
    
    # Récupérer les données MVRV
    logger.info(f"Récupération des données MVRV pour {args.asset} (derniers {args.days} jours)")
    mvrv_data = mvrv_indicator.get_mvrv_data(asset=args.asset, days=args.days)
    
    if mvrv_data.empty:
        logger.error("Aucune donnée MVRV récupérée.")
        return
    
    # Afficher les dernières lignes des données MVRV
    logger.info("\nDernières lignes des données MVRV:")
    print(mvrv_data.tail().to_string())
    
    # Calculer le Z-score
    logger.info("\nCalcul du Z-score MVRV")
    mvrv_data_with_z = mvrv_indicator.calculate_mvrv_z_score(mvrv_data)
    
    # Afficher les dernières lignes avec le Z-score
    logger.info("\nDernières lignes avec le Z-score:")
    print(mvrv_data_with_z[['mvrv_ratio', 'mvrv_ema', 'mvrv_z_score']].tail().to_string())
    
    # Obtenir le signal
    signal = mvrv_indicator.get_signal(mvrv_data)
    logger.info(f"\nSignal MVRV: {signal.value}")
    
    # Vérifier si le marché est sous-évalué ou surévalué
    is_undervalued = mvrv_indicator.is_undervalued(mvrv_data)
    is_overvalued = mvrv_indicator.is_overvalued(mvrv_data)
    logger.info(f"\nMarché sous-évalué: {is_undervalued}")
    logger.info(f"Marché surévalué: {is_overvalued}")
    
    # Analyse complète
    logger.info("\nAnalyse complète du MVRV")
    analysis = mvrv_indicator.analyze(asset=args.asset, days=args.days)
    
    # Afficher les résultats de l'analyse
    logger.info("\nRésultats de l'analyse:")
    for key, value in analysis.items():
        if key != 'data':
            if isinstance(value, MVRVSignal):
                logger.info(f"{key}: {value.value}")
            elif isinstance(value, float) or isinstance(value, int):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    
    # Créer un graphique
    logger.info("\nCréation du graphique MVRV")
    fig = mvrv_indicator.plot_mvrv(mvrv_data_with_z, title=f"Ratio MVRV pour {args.asset}")
    
    # Enregistrer le graphique si un chemin est spécifié
    if args.save_path:
        save_path = args.save_path
    else:
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mvrv')
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin du fichier de sortie
        save_path = os.path.join(output_dir, f'{args.asset}_mvrv_analysis.png')
    
    # Enregistrer le graphique
    fig.savefig(save_path)
    logger.info(f"Graphique enregistré: {save_path}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
