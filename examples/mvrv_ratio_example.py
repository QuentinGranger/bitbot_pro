"""
Example d'utilisation de l'indicateur MVRV (Market Value to Realized Value).

Ce script démontre comment utiliser l'indicateur MVRV pour évaluer si le Bitcoin
est surévalué ou sous-évalué, avec des fonctionnalités de fallback vers NUPL
et ajustement de la pondération en fonction de la volatilité.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.strategie.base.MVRVRatio import MVRVIndicator, MVRVSignal
from bitbot.utils.logger import logger
from bitbot.data.onchain_client import OnChainClient

def main():
    """
    Exemple d'utilisation de l'indicateur MVRV avec plusieurs configurations
    pour comparer leur efficacité.
    """
    
    # Configuration de base
    days = 365  # Données sur un an
    asset = "BTC"
    
    print(f"Analyse du ratio MVRV pour {asset} sur {days} jours")
    print("-" * 50)
    
    # 1. Configuration standard sans fallback
    print("\n1. Configuration standard sans fallback")
    mvrv_standard = MVRVIndicator(
        ema_period=50,
        undervalued_threshold=1.0,
        strong_undervalued_threshold=0.75,
        overvalued_threshold=2.5,
        strong_overvalued_threshold=3.5,
        use_fallback=False,
        consider_orderbook=False
    )
    
    # Analyser le MVRV
    result_standard = mvrv_standard.analyze(asset=asset, days=days)
    
    # Afficher les résultats de l'analyse standard
    if 'mvrv_ratio' in result_standard and result_standard['mvrv_ratio'] is not None:
        print(f"Ratio MVRV: {result_standard['mvrv_ratio']:.4f}")
        print(f"Signal: {result_standard['signal'].value}")
        print(f"Le marché est {'sous-évalué' if result_standard['is_undervalued'] else 'non sous-évalué'}")
        print(f"Le marché est {'surévalué' if result_standard['is_overvalued'] else 'non surévalué'}")
    else:
        print("Aucune donnée disponible pour l'analyse standard")
    
    # 2. Configuration avec fallback vers NUPL
    print("\n2. Configuration avec fallback vers NUPL")
    mvrv_with_fallback = MVRVIndicator(
        ema_period=50,
        undervalued_threshold=1.0,
        strong_undervalued_threshold=0.75,
        overvalued_threshold=2.5,
        strong_overvalued_threshold=3.5,
        use_fallback=True,
        consider_orderbook=False
    )
    
    # Simuler une analyse avec fallback en forçant l'utilisation de NUPL
    # Pour cela, on crée une instance de OnChainClient où get_approximate_mvrv_ratio retournerait un DataFrame vide
    onchain_client = OnChainClient()
    original_mvrv_method = onchain_client.get_approximate_mvrv_ratio
    
    try:
        # Remplacer temporairement la méthode pour simuler un échec
        onchain_client.get_approximate_mvrv_ratio = lambda **kwargs: pd.DataFrame()
        
        # Remplacer le client dans l'indicateur
        mvrv_with_fallback.client = onchain_client
        
        # Analyse avec fallback vers NUPL
        result_fallback = mvrv_with_fallback.analyze(asset=asset, days=days)
        
        # Afficher les résultats de l'analyse avec fallback
        if 'nupl' in result_fallback:
            print(f"Utilisant NUPL comme fallback: Oui")
            print(f"NUPL: {result_fallback['nupl']:.4f}")
            print(f"Catégorie NUPL: {result_fallback['nupl_category']}")
            print(f"Ratio MVRV (calculé à partir de NUPL): {result_fallback['mvrv_ratio']:.4f}")
            print(f"Signal: {result_fallback['signal'].value}")
        else:
            print("Aucune donnée disponible pour l'analyse avec fallback")
    finally:
        # Restaurer la méthode originale
        onchain_client.get_approximate_mvrv_ratio = original_mvrv_method
    
    # 3. Configuration avec ajustement de volatilité
    print("\n3. Configuration avec ajustement de volatilité")
    mvrv_with_volatility = MVRVIndicator(
        ema_period=50,
        undervalued_threshold=1.0,
        strong_undervalued_threshold=0.75,
        overvalued_threshold=2.5,
        strong_overvalued_threshold=3.5,
        use_fallback=True,
        volatility_threshold=0.02,  # 2% de volatilité comme seuil
        consider_orderbook=False
    )
    
    # Analyser le MVRV avec ajustement de volatilité
    result_volatility = mvrv_with_volatility.analyze(asset=asset, days=days)
    
    # Afficher les résultats de l'analyse avec ajustement de volatilité
    if 'mvrv_ratio' in result_volatility and result_volatility['mvrv_ratio'] is not None:
        print(f"Ratio MVRV: {result_volatility['mvrv_ratio']:.4f}")
        print(f"Signal: {result_volatility['signal'].value}")
        print(f"Volatilité actuelle: {result_volatility['volatility_weight']:.2f}")
        print(f"Pondération des indicateurs on-chain: {result_volatility['volatility_weight']:.2f}")
    else:
        print("Aucune donnée disponible pour l'analyse avec ajustement de volatilité")
    
    # 4. Configuration avec analyse des carnets d'ordres
    print("\n4. Configuration avec analyse des carnets d'ordres")
    mvrv_with_orderbook = MVRVIndicator(
        ema_period=50,
        undervalued_threshold=1.0,
        strong_undervalued_threshold=0.75,
        overvalued_threshold=2.5,
        strong_overvalued_threshold=3.5,
        use_fallback=True,
        volatility_threshold=0.02,
        consider_orderbook=True
    )
    
    # Analyser le MVRV avec analyse des carnets d'ordres
    result_orderbook = mvrv_with_orderbook.analyze(asset=asset, days=days, check_orderbooks=True)
    
    # Afficher les résultats de l'analyse avec carnets d'ordres
    if 'mvrv_ratio' in result_orderbook and result_orderbook['mvrv_ratio'] is not None:
        print(f"Ratio MVRV: {result_orderbook['mvrv_ratio']:.4f}")
        print(f"Signal: {result_orderbook['signal'].value}")
        
        if 'orderbook_analyzed' in result_orderbook and result_orderbook['orderbook_analyzed']:
            print(f"Analyse des carnets d'ordres effectuée: Oui")
            print(f"Murs d'achat détectés: {result_orderbook.get('buy_walls_detected', False)}")
            print(f"Murs de vente détectés: {result_orderbook.get('sell_walls_detected', False)}")
            print(f"Note: {result_orderbook.get('orderbook_note', '')}")
    else:
        print("Aucune donnée disponible pour l'analyse avec carnets d'ordres")
    
    # Visualisation
    print("\n5. Visualisation du ratio MVRV")
    
    # Récupérer les données pour la visualisation
    data = result_standard.get('data', pd.DataFrame())
    
    if not data.empty:
        # Créer un graphique avec les données standard
        fig = mvrv_standard.plot_mvrv(
            data,
            title=f"Ratio MVRV pour {asset} sur {days} jours",
            show_thresholds=True,
            show_z_score=True
        )
        
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune donnée disponible pour la visualisation")

if __name__ == "__main__":
    main()
