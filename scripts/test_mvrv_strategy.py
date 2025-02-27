#!/usr/bin/env python3
"""
Script de test pour la stratégie basée sur l'indicateur MVRV.
Ce script analyse le ratio MVRV et génère des recommandations d'investissement,
en utilisant des API gratuites comme CoinGecko pour les données.
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

from bitbot.strategie.indicators.mvrv_strategy import MVRVStrategy
from bitbot.models.trade_signal import SignalType
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester la stratégie basée sur le MVRV."""
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description='Test de la stratégie basée sur le MVRV')
    parser.add_argument('--days', type=int, default=365, help='Nombre de jours de données à récupérer')
    parser.add_argument('--asset', type=str, default='BTC', help='Actif à analyser')
    parser.add_argument('--save-path', type=str, help='Chemin pour enregistrer le graphique')
    
    args = parser.parse_args()
    
    logger.info("Test de la stratégie basée sur l'indicateur MVRV")
    
    # Initialiser la stratégie
    strategy = MVRVStrategy(
        ema_period=50,
        undervalued_threshold=1.0,
        strong_undervalued_threshold=0.75,
        overvalued_threshold=2.5,
        strong_overvalued_threshold=3.5,
        use_z_score=True,
        z_score_threshold=2.0
    )
    
    # Générer des signaux
    logger.info(f"Génération des signaux pour {args.asset}")
    signals = strategy.generate_signals(asset=args.asset, days=args.days)
    
    # Afficher les signaux
    if signals:
        logger.info(f"\nSignaux générés ({len(signals)}):")
        for signal in signals:
            logger.info(f"Type: {signal.signal_type.name}, "
                       f"Timestamp: {signal.timestamp}, "
                       f"Confiance: {signal.confidence}, "
                       f"MVRV: {signal.metadata.get('mvrv_ratio', 'N/A')}, "
                       f"Z-Score: {signal.metadata.get('mvrv_z_score', 'N/A')}")
    else:
        logger.info("Aucun signal généré")
    
    # Obtenir la position dans le cycle de marché
    logger.info("\nAnalyse de la position dans le cycle de marché")
    cycle_position = strategy.get_market_cycle_position(asset=args.asset, days=args.days)
    
    # Afficher la position dans le cycle
    logger.info(f"\nPosition dans le cycle de marché:")
    logger.info(f"Position: {cycle_position['cycle_position']}")
    logger.info(f"Confiance: {cycle_position['confidence']:.2f}")
    logger.info(f"Détails: {cycle_position['details']}")
    logger.info(f"MVRV: {cycle_position['mvrv_ratio']:.4f}")
    if 'mvrv_z_score' in cycle_position and cycle_position['mvrv_z_score'] is not None:
        logger.info(f"Z-Score: {cycle_position['mvrv_z_score']:.4f}")
    
    # Obtenir les recommandations d'investissement
    logger.info("\nGénération des recommandations d'investissement")
    recommendation = strategy.get_investment_recommendation(asset=args.asset, days=args.days)
    
    # Afficher les recommandations
    logger.info(f"\nRecommandation d'investissement:")
    logger.info(f"Recommandation: {recommendation['recommendation']}")
    logger.info(f"Confiance: {recommendation['confidence']:.2f}")
    logger.info(f"Allocation suggérée: {recommendation['allocation'] * 100:.1f}%")
    logger.info(f"Détails: {recommendation['details']}")
    
    # Créer un graphique
    logger.info("\nCréation du graphique MVRV")
    
    # Récupérer les données MVRV
    mvrv_data = strategy.mvrv_indicator.get_mvrv_data(asset=args.asset, days=args.days)
    
    if mvrv_data.empty:
        logger.error("Aucune donnée MVRV disponible pour créer un graphique")
        return
    
    # Calculer le Z-score
    mvrv_data_with_z = strategy.mvrv_indicator.calculate_mvrv_z_score(mvrv_data)
    
    # Créer le graphique
    fig = strategy.mvrv_indicator.plot_mvrv(
        mvrv_data_with_z, 
        title=f"Analyse MVRV pour {args.asset} - {recommendation['recommendation']}"
    )
    
    # Ajouter des annotations pour la recommandation
    plt.figtext(0.5, 0.01, 
               f"Recommandation: {recommendation['recommendation']} | "
               f"Position dans le cycle: {cycle_position['cycle_position']} | "
               f"MVRV: {recommendation['mvrv_ratio']:.4f}",
               ha='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Enregistrer le graphique si un chemin est spécifié
    if args.save_path:
        save_path = args.save_path
    else:
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mvrv_strategy')
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin du fichier de sortie
        save_path = os.path.join(output_dir, f'{args.asset}_mvrv_strategy.png')
    
    # Enregistrer le graphique
    fig.savefig(save_path)
    logger.info(f"Graphique enregistré: {save_path}")
    
    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    main()
