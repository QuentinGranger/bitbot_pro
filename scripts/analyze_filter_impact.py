#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse de l'impact des filtres de lissage sur les performances de trading.

Ce script compare les performances des stratégies de trading avec et sans filtres 
de lissage sur différentes paires de trading et différents timeframes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile
import json

# Ajouter le répertoire parent au path pour pouvoir importer bitbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.data.market_data import MarketData
from bitbot.data.binance_client import BinanceClient
from bitbot.strategies.mean_reversion import MeanReversionStrategy
from bitbot.strategies.trend_following import TrendFollowingStrategy
from bitbot.strategies.breakout import BreakoutStrategy
from bitbot.utils.data_cleaner import clean_market_data, auto_select_filter
from bitbot.backtest.backtester import Backtester

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "filter_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Définir les stratégies à tester
STRATEGIES = {
    "MeanReversion": {
        "class": MeanReversionStrategy,
        "params": {"window": 20, "deviation_threshold": 1.5}
    },
    "TrendFollowing": {
        "class": TrendFollowingStrategy,
        "params": {"short_window": 10, "long_window": 30}
    },
    "Breakout": {
        "class": BreakoutStrategy,
        "params": {"window": 20, "volatility_factor": 2.0}
    }
}

# Classe pour l'analyse de l'impact des filtres
class FilterImpactAnalyzer:
    def __init__(self, client=None):
        self.client = client or BinanceClient()
        self.results = {
            "raw": {},
            "filtered": {}
        }
    
    def run_analysis(self, symbols=SYMBOLS, timeframes=TIMEFRAMES, strategies=STRATEGIES):
        """Exécute l'analyse pour les symboles, timeframes et stratégies spécifiés."""
        for symbol in symbols:
            for timeframe in timeframes:
                print(f"\nAnalyse de {symbol} ({timeframe}):")
                
                # Récupérer les données historiques
                market_data = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=datetime.now() - timedelta(days=30),
                    end_time=datetime.now()
                )
                
                # Sauvegarder les données brutes
                raw_data = market_data.copy()
                
                for strategy_name, strategy_config in strategies.items():
                    print(f"  - Stratégie: {strategy_name}")
                    
                    # 1. Backtest avec données brutes (nettoyées des outliers uniquement)
                    raw_backtest = self._run_backtest(
                        raw_data, 
                        strategy_name, 
                        strategy_config,
                        filter_type=None
                    )
                    
                    # 2. Backtest avec données filtrées (filtre auto-sélectionné)
                    use_case = "general"
                    if "trend" in strategy_name.lower():
                        use_case = "trend_following"
                    elif "mean" in strategy_name.lower() or "reversion" in strategy_name.lower():
                        use_case = "mean_reversion"
                    elif "breakout" in strategy_name.lower():
                        use_case = "breakout"
                        
                    filtered_data = clean_market_data(raw_data.copy(), use_case=use_case)
                    filtered_backtest = self._run_backtest(
                        filtered_data, 
                        strategy_name, 
                        strategy_config,
                        filter_type=filtered_data.metadata.get('filter_type', 'auto')
                    )
                    
                    # Stocker les résultats
                    key = f"{symbol}_{timeframe}_{strategy_name}"
                    self.results["raw"][key] = raw_backtest
                    self.results["filtered"][key] = filtered_backtest
                    
                    # Afficher les résultats comparatifs
                    self._print_comparison(raw_backtest, filtered_backtest)
                    
                    # Générer un graphique comparatif
                    self._generate_comparison_chart(
                        raw_backtest,
                        filtered_backtest,
                        symbol,
                        timeframe,
                        strategy_name
                    )
    
    def _run_backtest(self, market_data, strategy_name, strategy_config, filter_type=None):
        """Exécute un backtest avec les données et la stratégie spécifiées."""
        strategy_class = strategy_config["class"]
        strategy_params = strategy_config["params"]
        
        # Créer la stratégie
        strategy = strategy_class(**strategy_params)
        
        # Configurer et exécuter le backtest
        backtester = Backtester(
            market_data=market_data,
            strategy=strategy,
            initial_balance=10000,
            commission_rate=0.001
        )
        
        # Exécuter le backtest
        results = backtester.run()
        
        # Ajouter des métadonnées utiles
        results["filter_type"] = filter_type
        results["strategy_name"] = strategy_name
        results["strategy_params"] = strategy_params
        results["symbol"] = market_data.symbol
        results["timeframe"] = market_data.timeframe
        
        return results
    
    def _print_comparison(self, raw_results, filtered_results):
        """Affiche une comparaison des résultats de backtest."""
        print(f"    Sans filtre:  Rendement: {raw_results['total_return']:.2f}%, "
              f"Max drawdown: {raw_results['max_drawdown']:.2f}%, "
              f"Ratio Sharpe: {raw_results['sharpe_ratio']:.2f}")
        
        print(f"    Avec filtre:  Rendement: {filtered_results['total_return']:.2f}%, "
              f"Max drawdown: {filtered_results['max_drawdown']:.2f}%, "
              f"Ratio Sharpe: {filtered_results['sharpe_ratio']:.2f}")
        
        # Calculer les différences
        return_diff = filtered_results['total_return'] - raw_results['total_return']
        drawdown_diff = filtered_results['max_drawdown'] - raw_results['max_drawdown']
        sharpe_diff = filtered_results['sharpe_ratio'] - raw_results['sharpe_ratio']
        
        # Afficher les différences
        print(f"    Différence:   Rendement: {return_diff:+.2f}%, "
              f"Max drawdown: {drawdown_diff:+.2f}%, "
              f"Ratio Sharpe: {sharpe_diff:+.2f}")
    
    def _generate_comparison_chart(self, raw_results, filtered_results, symbol, timeframe, strategy_name):
        """Génère un graphique comparant les performances avec et sans filtre."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Tracer les courbes de rendement cumulatif
        ax1.plot(raw_results['equity_curve'], label='Sans filtre', color='blue', alpha=0.7)
        ax1.plot(filtered_results['equity_curve'], label=f'Avec filtre ({filtered_results["filter_type"]})', 
                 color='green', alpha=0.7)
        ax1.set_title(f'Comparaison des rendements - {symbol} ({timeframe}) - {strategy_name}')
        ax1.set_ylabel('Équité ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tracer les drawdowns
        ax2.plot(raw_results['drawdown_curve'], label='Sans filtre', color='red', alpha=0.7)
        ax2.plot(filtered_results['drawdown_curve'], label=f'Avec filtre ({filtered_results["filter_type"]})', 
                 color='orange', alpha=0.7)
        ax2.set_title('Comparaison des drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajouter un texte avec les métriques clés
        metrics_text = (
            f"Sans filtre - Rendement: {raw_results['total_return']:.2f}%, "
            f"Max DD: {raw_results['max_drawdown']:.2f}%, Sharpe: {raw_results['sharpe_ratio']:.2f}\n"
            f"Avec filtre - Rendement: {filtered_results['total_return']:.2f}%, "
            f"Max DD: {filtered_results['max_drawdown']:.2f}%, Sharpe: {filtered_results['sharpe_ratio']:.2f}"
        )
        fig.text(0.5, 0.01, metrics_text, ha='center', va='center', fontsize=10)
        
        # Ajuster la mise en page et sauvegarder
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        filename = f"{symbol}_{timeframe}_{strategy_name}_filter_comparison.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        
        print(f"    Graphique sauvegardé: {filename}")
    
    def save_results(self):
        """Sauvegarde les résultats de l'analyse dans un fichier CSV et JSON."""
        # Préparer un DataFrame pour les résultats
        results_data = []
        
        for key in self.results["raw"].keys():
            symbol, timeframe, strategy_name = key.split("_")
            
            raw = self.results["raw"][key]
            filtered = self.results["filtered"][key]
            
            results_data.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy_name,
                "filter_type": filtered["filter_type"],
                "raw_return": raw["total_return"],
                "filtered_return": filtered["total_return"],
                "return_diff": filtered["total_return"] - raw["total_return"],
                "raw_max_drawdown": raw["max_drawdown"],
                "filtered_max_drawdown": filtered["max_drawdown"],
                "drawdown_diff": filtered["max_drawdown"] - raw["max_drawdown"],
                "raw_sharpe": raw["sharpe_ratio"],
                "filtered_sharpe": filtered["sharpe_ratio"],
                "sharpe_diff": filtered["sharpe_ratio"] - raw["sharpe_ratio"],
                "raw_trade_count": raw["trade_count"],
                "filtered_trade_count": filtered["trade_count"],
                "raw_win_rate": raw["win_rate"],
                "filtered_win_rate": filtered["win_rate"]
            })
        
        # Créer et sauvegarder le DataFrame
        df = pd.DataFrame(results_data)
        csv_file = os.path.join(OUTPUT_DIR, "filter_impact_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nRésultats sauvegardés dans: {csv_file}")
        
        # Sauvegarder également un résumé au format JSON
        summary = {
            "average_return_improvement": df["return_diff"].mean(),
            "average_drawdown_reduction": -df["drawdown_diff"].mean(),
            "average_sharpe_improvement": df["sharpe_diff"].mean(),
            "best_filter_by_return": df.loc[df["return_diff"].idxmax()].to_dict(),
            "best_filter_by_drawdown": df.loc[df["drawdown_diff"].idxmin()].to_dict(),
            "best_filter_by_sharpe": df.loc[df["sharpe_diff"].idxmax()].to_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        json_file = os.path.join(OUTPUT_DIR, "filter_impact_summary.json")
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Résumé sauvegardé dans: {json_file}")
        
        # Générer un graphique de synthèse
        self._generate_summary_chart(df)
        
        return df, summary
    
    def _generate_summary_chart(self, df):
        """Génère un graphique de synthèse des améliorations par filtre."""
        plt.figure(figsize=(12, 8))
        
        # Créer un graphique en barres groupées par stratégie
        strategies = df["strategy"].unique()
        x = np.arange(len(strategies))
        width = 0.25
        
        # Calculer les moyennes par stratégie
        strategy_means = df.groupby("strategy").mean()
        
        plt.bar(x - width, strategy_means["return_diff"], width, label='Amélioration du rendement (%)', color='green')
        plt.bar(x, -strategy_means["drawdown_diff"], width, label='Réduction du drawdown (%)', color='red')
        plt.bar(x + width, strategy_means["sharpe_diff"], width, label='Amélioration du ratio Sharpe', color='blue')
        
        plt.xlabel('Stratégie')
        plt.ylabel('Amélioration')
        plt.title('Impact moyen des filtres par stratégie')
        plt.xticks(x, strategies)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Ajouter une ligne horizontale à zéro
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Sauvegarder le graphique
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, "filter_impact_summary.png")
        plt.savefig(filename)
        plt.close()
        print(f"Graphique de synthèse sauvegardé: {filename}")


if __name__ == "__main__":
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Analyse de l'impact des filtres sur les performances de trading")
    print(f"================================================================")
    print(f"Paires analysées: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Stratégies: {', '.join(STRATEGIES.keys())}")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print(f"================================================================\n")
    
    # Lancer l'analyse
    analyzer = FilterImpactAnalyzer()
    analyzer.run_analysis()
    
    # Sauvegarder et afficher les résultats
    df, summary = analyzer.save_results()
    
    print("\nRésumé des résultats:")
    print(f"Amélioration moyenne du rendement: {summary['average_return_improvement']:.2f}%")
    print(f"Réduction moyenne du drawdown: {summary['average_drawdown_reduction']:.2f}%")
    print(f"Amélioration moyenne du ratio Sharpe: {summary['average_sharpe_improvement']:.2f}")
    print(f"\nMeilleur résultat par rendement: {summary['best_filter_by_return']['symbol']} "
          f"({summary['best_filter_by_return']['timeframe']}) avec {summary['best_filter_by_return']['strategy']}")
    print(f"Meilleur résultat par drawdown: {summary['best_filter_by_drawdown']['symbol']} "
          f"({summary['best_filter_by_drawdown']['timeframe']}) avec {summary['best_filter_by_drawdown']['strategy']}")
