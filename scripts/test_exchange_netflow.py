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

def test_netflow_price_correlation(netflow_data, days_shift=5):
    """
    Teste la corrélation entre les pics de netflow et les mouvements de prix.
    
    Args:
        netflow_data: DataFrame contenant les données de netflow.
        days_shift: Nombre de jours de décalage pour analyser l'impact du netflow sur le prix.
    
    Returns:
        Dict avec les statistiques de corrélation.
    """
    if 'netflow' not in netflow_data.columns or 'price' not in netflow_data.columns:
        logger.warning("Données insuffisantes pour l'analyse de corrélation")
        return {}
    
    # Calculer le décalage des prix pour analyser l'impact futur du netflow
    shifted_prices = netflow_data['price'].shift(-days_shift)
    
    # Identifier les pics significatifs de netflow (outflow important)
    significant_outflow = netflow_data['netflow'] < -1000  # Seuil arbitraire pour les flux sortants significatifs
    
    # Identifier les pics significatifs de netflow (inflow important)
    significant_inflow = netflow_data['netflow'] > 1000  # Seuil arbitraire pour les flux entrants significatifs
    
    # Calculer le pourcentage de fois où un outflow significatif est suivi d'une hausse de prix
    if sum(significant_outflow) > 0:
        outflow_followed_by_price_increase = sum((netflow_data['netflow'] < -1000) & 
                                                (shifted_prices > netflow_data['price']))
        outflow_price_increase_pct = outflow_followed_by_price_increase / sum(significant_outflow) * 100
    else:
        outflow_price_increase_pct = 0
    
    # Calculer le pourcentage de fois où un inflow significatif est suivi d'une baisse de prix
    if sum(significant_inflow) > 0:
        inflow_followed_by_price_decrease = sum((netflow_data['netflow'] > 1000) & 
                                               (shifted_prices < netflow_data['price']))
        inflow_price_decrease_pct = inflow_followed_by_price_decrease / sum(significant_inflow) * 100
    else:
        inflow_price_decrease_pct = 0
    
    # Calculer la corrélation entre netflow et variation de prix future
    price_change_pct = (shifted_prices - netflow_data['price']) / netflow_data['price'] * 100
    correlation = netflow_data['netflow'].corr(price_change_pct)
    
    return {
        "outflow_followed_by_price_increase_pct": outflow_price_increase_pct,
        "inflow_followed_by_price_decrease_pct": inflow_price_decrease_pct,
        "correlation": correlation,
        "significant_outflow_count": sum(significant_outflow),
        "significant_inflow_count": sum(significant_inflow),
        "forecast_days": days_shift
    }

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
    parser.add_argument("--volatility-threshold", type=float, default=0.02, help="Seuil de volatilité (en décimal)")
    parser.add_argument("--check-orderbooks", action="store_true", help="Analyser les carnets d'ordres")
    
    args = parser.parse_args()
    
    logger.info("Test du module Exchange Netflow (Analyse des flux d'échanges)")
    
    # Initialiser l'indicateur avec les paramètres de ligne de commande
    netflow_indicator = ExchangeNetflow(
        ema_period=args.ema,
        outflow_threshold=args.outflow,
        strong_outflow_threshold=args.strong_outflow,
        inflow_threshold=args.inflow,
        strong_inflow_threshold=args.strong_inflow,
        volatility_threshold=args.volatility_threshold,
        consider_orderbook=args.check_orderbooks
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
    
    # Analyser les données avec la nouvelle méthode analyze qui inclut les carnets d'ordres
    logger.info("\nAnalyse des flux d'échange avec les nouvelles fonctionnalités")
    analysis = netflow_indicator.analyze(asset=args.asset, days=args.days, check_orderbooks=args.check_orderbooks)
    
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
    
    # Afficher la pondération de volatilité
    volatility_weight = analysis.get("volatility_weight", 1.0)
    logger.info(f"\nPondération de volatilité: {volatility_weight:.2f}")
    
    # Afficher les informations du carnet d'ordres si disponibles
    if "orderbook_analyzed" in analysis and analysis["orderbook_analyzed"]:
        logger.info("\nRésultats de l'analyse du carnet d'ordres:")
        logger.info(f"Murs d'achat détectés: {analysis.get('buy_walls_detected', False)}")
        logger.info(f"Murs de vente détectés: {analysis.get('sell_walls_detected', False)}")
        logger.info(f"Note: {analysis.get('orderbook_note', '')}")
        
        if "signal_reinforced" in analysis and analysis["signal_reinforced"]:
            logger.info("→ Le signal est renforcé par l'analyse du carnet d'ordres")
        elif "signal_attenuated" in analysis and analysis["signal_attenuated"]:
            logger.info("→ Le signal est atténué par l'analyse du carnet d'ordres")
    
    # Afficher l'analyse complète des flux d'échange (autres métriques)
    logger.info("\nAutres métriques d'analyse des flux d'échange")
    for key, value in analysis.items():
        if key not in ["signal", "is_outflow", "is_strong_outflow", "is_inflow", "is_strong_inflow", 
                     "volatility_weight", "orderbook_analyzed", "buy_walls_detected", "sell_walls_detected", 
                     "orderbook_note", "signal_reinforced", "signal_attenuated", "signal_note", "data"]:
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Générer et sauvegarder le graphique
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "exchange_netflow")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.asset}_exchange_netflow_analysis.png")
    
    logger.info(f"\nCréation du graphique de flux d'échange")
    netflow_indicator.plot_netflow(netflow_data, asset=args.asset, output_path=output_path)
    
    # Tester la corrélation entre les pics de netflow et les mouvements de prix
    logger.info("\nTest de la corrélation entre les pics de Netflow et les mouvements de prix")
    correlation_results = test_netflow_price_correlation(netflow_data)
    
    if correlation_results:
        logger.info("\nRésultats du test de corrélation:")
        logger.info(f"Pourcentage de fois où un outflow important est suivi d'une hausse de prix: {correlation_results.get('outflow_followed_by_price_increase_pct', 0):.2f}%")
        logger.info(f"Pourcentage de fois où un inflow important est suivi d'une baisse de prix: {correlation_results.get('inflow_followed_by_price_decrease_pct', 0):.2f}%")
        logger.info(f"Corrélation entre netflow et variation de prix future: {correlation_results.get('correlation', 0):.4f}")
        logger.info(f"Nombre d'événements de flux sortant significatifs: {correlation_results.get('significant_outflow_count', 0)}")
        logger.info(f"Nombre d'événements de flux entrant significatifs: {correlation_results.get('significant_inflow_count', 0)}")
        
        # Interprétation de la corrélation
        correlation = correlation_results.get('correlation', 0)
        if correlation < -0.4:
            logger.info("La corrélation négative significative suggère une forte relation inverse entre les flux d'échange et les prix futurs")
            logger.info("(un outflow important tend à être suivi d'une hausse de prix, un inflow important tend à être suivi d'une baisse)")
        elif correlation < -0.2:
            logger.info("La corrélation négative modérée suggère une relation inverse entre les flux d'échange et les prix futurs")
        elif correlation < 0:
            logger.info("La corrélation négative faible suggère une légère tendance inverse entre les flux d'échange et les prix futurs")
        elif correlation > 0.4:
            logger.info("La corrélation positive significative est inattendue et suggère que le modèle de simulation")
            logger.info("ne reflète peut-être pas correctement la dynamique réelle des flux d'échange")
        else:
            logger.info("La corrélation proche de zéro suggère que les flux d'échange simulés ne sont pas")
            logger.info("fortement prédictifs des mouvements de prix dans cette période de temps.")
    
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
    
    # Afficher l'impact de la volatilité
    if volatility_weight < 0.8:
        logger.info("\nNote sur la volatilité:")
        logger.info(f"La volatilité actuelle du marché est faible (pondération: {volatility_weight:.2f}).")
        logger.info("En période de faible volatilité, les signaux des indicateurs on-chain comme")
        logger.info("les flux d'échange sont moins réactifs à court terme et doivent être interprétés avec prudence.")

if __name__ == "__main__":
    main()
