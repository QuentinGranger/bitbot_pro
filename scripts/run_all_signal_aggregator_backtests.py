#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour exécuter tous les backtests de l'agrégateur de signaux
et comparer les performances entre les différents scénarios de marché.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Ajouter le répertoire parent au path pour pouvoir importer les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import des modules nécessaires après avoir défini le chemin
from tests.test_signal_aggregator_backtest import BacktestRunner

# Importation de la fonction run_comprehensive_backtests ou définition si elle n'existe pas
try:
    from tests.test_signal_aggregator_backtest import run_comprehensive_backtests
except ImportError:
    # Définir la fonction si elle n'est pas disponible dans le module
    def run_comprehensive_backtests(plot=False):
        """
        Exécute des backtests complets et retourne les résultats.
        """
        runner = BacktestRunner()
        results = {}
        
        # Uptrend
        print("Exécution du backtest en tendance haussière...")
        market_data, agg_results, performance = runner.run_uptrend_backtest(plot=plot)
        results['uptrend'] = {
            'market_data': market_data,
            'aggregation_results': agg_results,
            'performance': performance
        }
        
        # Downtrend
        print("Exécution du backtest en tendance baissière...")
        market_data, agg_results, performance = runner.run_downtrend_backtest(plot=plot)
        results['downtrend'] = {
            'market_data': market_data,
            'aggregation_results': agg_results,
            'performance': performance
        }
        
        # Sideways
        print("Exécution du backtest en marché latéral...")
        market_data, agg_results, performance = runner.run_sideways_backtest(plot=plot)
        results['sideways'] = {
            'market_data': market_data,
            'aggregation_results': agg_results,
            'performance': performance
        }
        
        # Flash crash
        print("Exécution du backtest en scénario de flash crash...")
        market_data, agg_results, performance = runner.run_flash_crash_backtest(plot=plot)
        results['flash_crash'] = {
            'market_data': market_data,
            'aggregation_results': agg_results,
            'performance': performance
        }
        
        return results


def compare_performance_metrics(results_dict, plot=True):
    """
    Comparer les métriques de performance entre les différents scénarios.
    
    Args:
        results_dict: Dictionnaire contenant les résultats pour chaque scénario
        plot: Si True, génère des graphiques de comparaison
    """
    # Extraire les performances de chaque scénario
    performances = {
        scenario: data['performance'] for scenario, data in results_dict.items()
    }
    
    # Créer un DataFrame pour la comparaison
    metrics = ['total_trades', 'win_rate', 'avg_profit', 'max_profit', 'max_loss', 
              'profit_factor', 'sharpe_ratio', 'max_drawdown']
    
    comparison_data = []
    for scenario, perf in performances.items():
        row = [scenario]
        for metric in metrics:
            value = perf.get(metric, 0)
            if metric in ['win_rate', 'avg_profit', 'max_profit', 'max_loss', 'max_drawdown']:
                row.append(f"{value:.2f}%")
            elif metric in ['profit_factor', 'sharpe_ratio']:
                row.append(f"{value:.2f}")
            else:
                row.append(f"{value}")
        comparison_data.append(row)
    
    # Afficher le tableau de comparaison
    headers = ['Scénario'] + metrics
    print("\nComparaison des performances entre les scénarios:")
    print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
    
    # Plots de comparaison si demandé
    if plot:
        try:
            # 1. Graphique de comparaison des scores agrégés moyens
            plt.figure(figsize=(12, 6))
            
            for scenario, data in results_dict.items():
                aggregation_results = data['aggregation_results']
                if aggregation_results:
                    scores = [r['aggregated_score'] for r in aggregation_results]
                    plt.plot(range(len(scores)), scores, label=f"{scenario.capitalize()}")
            
            plt.axhline(y=65, color='green', linestyle='--', alpha=0.5, label='Seuil d\'achat')
            plt.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='Seuil de vente')
            plt.title('Comparaison des scores agrégés pour différents scénarios de marché')
            plt.xlabel('Période')
            plt.ylabel('Score agrégé')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('backtest_aggregated_scores_comparison.png')
            
            # 2. Graphique de comparaison des Win Rates
            plt.figure(figsize=(10, 6))
            win_rates = [float(row[2].strip('%')) for row in comparison_data]
            scenarios = [row[0] for row in comparison_data]
            
            bars = plt.bar(scenarios, win_rates, color=['#4CAF50', '#F44336', '#2196F3', '#FFC107'])
            plt.title('Comparaison des taux de réussite (Win Rate)')
            plt.xlabel('Scénario de marché')
            plt.ylabel('Win Rate (%)')
            plt.ylim(0, 100)
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('backtest_win_rate_comparison.png')
            
            # 3. Graphique de comparaison des profits moyens
            plt.figure(figsize=(10, 6))
            avg_profits = [float(row[3].strip('%')) for row in comparison_data]
            
            bars = plt.bar(scenarios, avg_profits, color=['#4CAF50', '#F44336', '#2196F3', '#FFC107'])
            plt.title('Comparaison des profits moyens par trade')
            plt.xlabel('Scénario de marché')
            plt.ylabel('Profit moyen (%)')
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.1 if height >= 0 else height - 0.5,
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('backtest_avg_profit_comparison.png')
            
            plt.show()
            
            print("\nGraphiques de comparaison sauvegardés sous:")
            print("- backtest_aggregated_scores_comparison.png")
            print("- backtest_win_rate_comparison.png")
            print("- backtest_avg_profit_comparison.png")
            
        except Exception as e:
            print(f"Erreur lors de la génération des graphiques de comparaison: {e}")


def analyze_robustness(results_dict):
    """
    Analyser la robustesse de l'agrégateur dans les différents scénarios.
    
    Args:
        results_dict: Dictionnaire contenant les résultats pour chaque scénario
    """
    print("\nAnalyse de la robustesse de l'agrégateur:")
    
    # 1. Distribution des scores agrégés par scénario
    score_distributions = {}
    for scenario, data in results_dict.items():
        if data['aggregation_results']:
            scores = [r['aggregated_score'] for r in data['aggregation_results']]
            score_distributions[scenario] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores),
                'buy_signal_ratio': sum(1 for s in scores if s > 65) / len(scores),
                'sell_signal_ratio': sum(1 for s in scores if s < 35) / len(scores),
                'neutral_ratio': sum(1 for s in scores if 35 <= s <= 65) / len(scores)
            }
    
    # Afficher les distributions
    for scenario, stats in score_distributions.items():
        print(f"\n{scenario.capitalize()}:")
        print(f"  Score moyen: {stats['mean']:.2f}")
        print(f"  Score médian: {stats['median']:.2f}")
        print(f"  Écart-type: {stats['std']:.2f}")
        print(f"  Min/Max: {stats['min']:.2f}/{stats['max']:.2f}")
        print(f"  % Signaux d'achat (>65): {stats['buy_signal_ratio']*100:.1f}%")
        print(f"  % Signaux de vente (<35): {stats['sell_signal_ratio']*100:.1f}%")
        print(f"  % Signaux neutres: {stats['neutral_ratio']*100:.1f}%")
    
    # 2. Analyser la variabilité des performances
    performances = {
        scenario: data['performance'] for scenario, data in results_dict.items()
    }
    
    # Calculer la variabilité des métriques clés
    metrics_to_analyze = ['win_rate', 'avg_profit', 'profit_factor', 'sharpe_ratio']
    metric_values = {}
    
    for metric in metrics_to_analyze:
        values = [perf.get(metric, 0) for perf in performances.values()]
        metric_values[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }
    
    print("\nVariabilité des métriques de performance:")
    for metric, stats in metric_values.items():
        print(f"\n{metric}:")
        print(f"  Moyenne: {stats['mean']:.2f}")
        print(f"  Écart-type: {stats['std']:.2f}")
        print(f"  Min/Max: {stats['min']:.2f}/{stats['max']:.2f}")
        print(f"  Plage: {stats['range']:.2f}")
        print(f"  Coefficient de variation: {stats['coefficient_of_variation']:.2f}")
    
    # 3. Conclusion sur la robustesse
    print("\nConclusion sur la robustesse:")
    
    # Calculer un score de robustesse basé sur la variabilité des performances
    cv_sum = sum(stats['coefficient_of_variation'] for stats in metric_values.values())
    robustness_score = max(0, 100 - (cv_sum * 50))  # Plus le CV est bas, plus le score est élevé
    
    if robustness_score > 80:
        conclusion = "EXCELLENT"
    elif robustness_score > 60:
        conclusion = "BON"
    elif robustness_score > 40:
        conclusion = "MOYEN"
    else:
        conclusion = "FAIBLE"
    
    print(f"Score de robustesse: {robustness_score:.1f}/100 - {conclusion}")
    print("Ce score est basé sur la cohérence des performances à travers différents scénarios de marché.")
    
    # Suggestions d'amélioration
    worst_scenario = min(performances.items(), key=lambda x: x[1].get('profit_factor', 0))
    print(f"\nLe scénario le plus difficile pour l'agrégateur: {worst_scenario[0]}")
    print("Suggestions d'amélioration:")
    
    if worst_scenario[0] == 'sideways':
        print("- Ajuster les poids pour donner plus d'importance aux indicateurs performants en marchés latéraux")
        print("- Augmenter les seuils d'entrée/sortie pour éviter les faux signaux dans les marchés peu volatils")
    elif worst_scenario[0] == 'flash_crash':
        print("- Intégrer des mécanismes de détection de mouvements brusques")
        print("- Ajouter des filtres de volatilité plus sophistiqués")
    elif worst_scenario[0] == 'downtrend':
        print("- Améliorer la détection des inversions de tendance baissière")
        print("- Ajuster les poids pour mieux détecter les rebonds dans un marché baissier")


def main():
    """Fonction principale pour exécuter les tests et analyser les résultats"""
    print("Exécution des backtests complets sur différents scénarios de marché...")
    
    # Exécuter tous les backtests
    results = run_comprehensive_backtests(plot=False)
    
    # Comparer les performances entre les scénarios
    compare_performance_metrics(results, plot=True)
    
    # Analyser la robustesse de l'agrégateur
    analyze_robustness(results)


if __name__ == "__main__":
    main()
