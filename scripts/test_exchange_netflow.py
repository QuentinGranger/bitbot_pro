#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du module ExchangeNetflow pour l'analyse des entrées/sorties de crypto-monnaies des exchanges.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le chemin parent au path pour importer bitbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.strategie.base.ExchangeNetflow import ExchangeNetflow, NetflowSignal
from bitbot.utils.logger import logger

def main():
    """Fonction principale pour tester le module ExchangeNetflow."""
    
    parser = argparse.ArgumentParser(description="Test du module Exchange Netflow")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours de données à récupérer")
    parser.add_argument("--asset", type=str, default="BTC", help="Actif à analyser (BTC, ETH, etc.)")
    parser.add_argument("--ema", type=int, default=14, help="Période pour l'EMA du netflow")
    parser.add_argument("--outflow", type=float, default=-1000, help="Seuil pour flux sortant")
    parser.add_argument("--strong-outflow", type=float, default=-5000, help="Seuil pour fort flux sortant")
    parser.add_argument("--inflow", type=float, default=1000, help="Seuil pour flux entrant")
    parser.add_argument("--strong-inflow", type=float, default=5000, help="Seuil pour fort flux entrant")
    
    args = parser.parse_args()
    
    logger.info("Test du module Exchange Netflow (Analyse des flux d'échanges)")
    
    # Initialiser l'indicateur avec les paramètres de ligne de commande
    netflow_indicator = ExchangeNetflow(
        ema_period=args.ema,
        outflow_threshold=args.outflow,
        strong_outflow_threshold=args.strong_outflow,
        inflow_threshold=args.inflow,
        strong_inflow_threshold=args.strong_inflow
    )
    
    # Récupérer les données de flux d'échange pour l'actif spécifié
    logger.info(f"Récupération des données de flux d'échange pour {args.asset} (derniers {args.days} jours)")
    netflow_data = netflow_indicator.get_netflow_data(asset=args.asset, days=args.days)
    
    if netflow_data.empty:
        logger.error("Aucune donnée de flux d'échange récupérée.")
        return
    
    # Afficher les dernières lignes des données de flux d'échange
    logger.info("\nDernières lignes des données de flux d'échange:")
    print(netflow_data.tail())
    
    # Analyser les données de flux d'échange
    logger.info("\nAnalyse des flux d'échange")
    analysis = netflow_indicator.analyze_netflow(netflow_data)
    
    if not analysis:
        logger.error("Analyse des flux d'échange impossible.")
        return
    
    # Afficher le signal actuel
    signal = analysis.get("signal", "Inconnu")
    logger.info(f"\nSignal de flux d'échange: {signal}")
    
    # Afficher si le marché est en flux sortant ou entrant
    is_outflow = analysis.get("is_outflow", False)
    is_strong_outflow = analysis.get("is_strong_outflow", False)
    is_inflow = analysis.get("is_inflow", False)
    is_strong_inflow = analysis.get("is_strong_inflow", False)
    
    logger.info(f"\nFlux sortant: {is_outflow}")
    logger.info(f"Fort flux sortant: {is_strong_outflow}")
    logger.info(f"Flux entrant: {is_inflow}")
    logger.info(f"Fort flux entrant: {is_strong_inflow}")
    
    # Afficher l'analyse complète des flux d'échange
    logger.info("\nAnalyse complète des flux d'échange")
    for key, value in analysis.items():
        if key not in ["signal", "is_outflow", "is_strong_outflow", "is_inflow", "is_strong_inflow"]:
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Générer et sauvegarder le graphique
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "exchange_netflow")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.asset}_exchange_netflow_analysis.png")
    
    logger.info(f"\nCréation du graphique de flux d'échange")
    netflow_indicator.plot_netflow(netflow_data, asset=args.asset, output_path=output_path)
    
    # Afficher une interprétation du signal
    logger.info("\nInterprétation du signal de flux d'échange:")
    if is_strong_outflow:
        logger.info("Le fort flux sortant des exchanges suggère une accumulation significative de Bitcoin.")
        logger.info("Cela indique généralement un sentiment très haussier, car les investisseurs déplacent leurs BTC")
        logger.info("des exchanges vers un stockage à long terme (cold storage).")
    elif is_outflow:
        logger.info("Le flux sortant des exchanges suggère une légère accumulation de Bitcoin.")
        logger.info("Cela indique généralement un sentiment haussier, les investisseurs préférant conserver leurs BTC.")
    elif is_strong_inflow:
        logger.info("Le fort flux entrant vers les exchanges suggère une pression de vente significative.")
        logger.info("Cela indique généralement un sentiment très baissier, car les investisseurs déplacent leurs BTC")
        logger.info("vers les exchanges probablement pour les vendre.")
    elif is_inflow:
        logger.info("Le flux entrant vers les exchanges suggère une légère pression de vente.")
        logger.info("Cela indique généralement un sentiment baissier à court terme.")
    else:
        logger.info("Les flux d'échange sont relativement équilibrés.")
        logger.info("Cela suggère un marché neutre sans pression significative d'achat ou de vente.")

if __name__ == "__main__":
    main()
