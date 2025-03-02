#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour tester l'agrégateur de signaux sur un marché latéral.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Chemin absolu vers le répertoire parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import direct avec le chemin absolu
BacktestRunner = None
try:
    # Essayer plusieurs approches d'import
    try:
        from tests.test_signal_aggregator_backtest import BacktestRunner
    except ImportError:
        # Tenter une importation avec importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_signal_aggregator_backtest", 
            os.path.join(parent_dir, "tests", "test_signal_aggregator_backtest.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BacktestRunner = module.BacktestRunner
except Exception as e:
    print(f"Erreur d'importation: {e}")
    sys.exit(1)

def main():
    """Fonction principale pour exécuter le backtest"""
    print("Test de l'agrégateur de signaux sur un marché latéral...")
    
    # Vérifier que la classe a bien été importée
    if BacktestRunner is None:
        print("Erreur: Impossible d'importer la classe BacktestRunner")
        return

    # Créer une instance du BacktestRunner
    runner = BacktestRunner()
    
    # Exécuter le backtest sur un marché latéral avec affichage du graphique
    market_data, results, performance = runner.run_sideways_backtest(days=60, plot=True)
    
    # Afficher les résultats de performance
    print("\nRésultats de performance:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # Afficher le nombre de signaux générés
    if results:
        print(f"\nNombre total de points d'agrégation: {len(results)}")
        
        # Vérifier combien de signaux sont au-dessus ou en-dessous des seuils
        buy_signals = sum(1 for r in results if r['aggregated_score'] > 70)
        sell_signals = sum(1 for r in results if r['aggregated_score'] < 30)
        print(f"Signaux d'achat (score > 70): {buy_signals}")
        print(f"Signaux de vente (score < 30): {sell_signals}")
        print(f"Signaux neutres: {len(results) - buy_signals - sell_signals}")

if __name__ == "__main__":
    main()
