#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de la stratégie OBV (On-Balance Volume).

Ce script démontre l'utilisation des stratégies OBV pour analyser la pression
acheteuse ou vendeuse basée sur le volume. Il montre comment:
1. Charger des données historiques
2. Appliquer différentes variantes de la stratégie OBV
3. Visualiser les résultats et les divergences
4. Comparer l'efficacité des signaux générés
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Importer les modules BitBotPro
from bitbot_pro.strategies.base_strategies.volume.obv import (
    OBVStrategy,
    OBVDivergenceStrategy,
    RateOfChangeOBVStrategy
)
from bitbot_pro.data.historic_data_provider import HistoricDataProvider
from bitbot_pro.utils.performance import timeit


@timeit
def load_sample_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Charge des données historiques pour l'exemple.
    
    Args:
        symbol: Paire de trading (défaut: BTC/USDT)
        timeframe: Intervalle de temps (défaut: 1d)
        lookback_days: Nombre de jours d'historique (défaut: 365)
        
    Returns:
        DataFrame contenant les données historiques
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"Chargement des données pour {symbol}, période: {start_date} à {end_date}, timeframe: {timeframe}")
    
    # Initialiser le provider de données
    data_provider = HistoricDataProvider()
    
    # Charger les données
    data = data_provider.get_historic_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Données chargées: {len(data)} points")
    return data


@timeit
def plot_obv_results(
    data: pd.DataFrame,
    title: str = "Analyse OBV",
    save_path: str = None
) -> None:
    """
    Visualise les résultats de l'analyse OBV.
    
    Args:
        data: DataFrame contenant les données de prix et les indicateurs OBV
        title: Titre du graphique
        save_path: Chemin pour sauvegarder l'image (si None, affiche le graphique)
    """
    # Créer une figure avec 2 subplots partagant l'axe des x
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Convertir l'index en datetime si nécessaire
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Format de date pour l'axe des x
    date_format = mdates.DateFormatter('%Y-%m-%d')
    
    # Configurer le premier subplot pour le prix
    ax1.set_title(title, fontsize=16)
    ax1.plot(data.index, data['price'], label='Prix de clôture', color='black', alpha=0.7)
    
    # Ajouter les signaux d'achat et de vente sur le graphique
    if 'signal_buy' in data.columns and 'signal_sell' in data.columns:
        buy_signals = data[data['signal_buy']]
        sell_signals = data[data['signal_sell']]
        
        ax1.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Signal Achat')
        ax1.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Signal Vente')
    
    # Ajouter les signaux forts si disponibles
    if 'signal_strong_buy' in data.columns and 'signal_strong_sell' in data.columns:
        strong_buy = data[data['signal_strong_buy']]
        strong_sell = data[data['signal_strong_sell']]
        
        ax1.scatter(strong_buy.index, strong_buy['price'], color='lime', marker='^', s=150, 
                   edgecolors='green', linewidth=1.5, label='Signal Achat Fort')
        ax1.scatter(strong_sell.index, strong_sell['price'], color='pink', marker='v', s=150, 
                   edgecolors='red', linewidth=1.5, label='Signal Vente Fort')
    
    # Ajouter les divergences si disponibles
    for divergence_type, color, marker, label in [
        ('bullish_divergence', 'green', '^', 'Divergence Haussière'),
        ('bearish_divergence', 'red', 'v', 'Divergence Baissière'),
        ('hidden_bullish_divergence', 'blue', '^', 'Divergence Haussière Cachée'),
        ('hidden_bearish_divergence', 'purple', 'v', 'Divergence Baissière Cachée')
    ]:
        if divergence_type in data.columns:
            divergence_points = data[data[divergence_type]]
            if not divergence_points.empty:
                ax1.scatter(divergence_points.index, divergence_points['price'], color=color, marker=marker, 
                           s=150, alpha=0.6, label=label)
    
    ax1.set_ylabel('Prix', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Configurer le deuxième subplot pour l'OBV
    ax2.set_title('On-Balance Volume (OBV)', fontsize=14)
    
    # Tracer l'OBV et ses moyennes mobiles
    ax2.plot(data.index, data['obv'], label='OBV', color='blue')
    
    if 'obv_ma' in data.columns:
        ax2.plot(data.index, data['obv_ma'], label='OBV MA', color='orange', linestyle='--')
    
    if 'obv_signal' in data.columns:
        ax2.plot(data.index, data['obv_signal'], label='OBV Signal', color='red', linestyle='-.')
    
    # Ajouter le ROC si disponible
    if 'obv_roc' in data.columns:
        ax3 = ax2.twinx()  # Créer un axe secondaire pour le ROC
        ax3.plot(data.index, data['obv_roc'], label='OBV ROC', color='purple', alpha=0.6)
        ax3.set_ylabel('Rate of Change (%)', color='purple', fontsize=12)
        ax3.tick_params(axis='y', colors='purple')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('OBV', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Configurer l'axe des x
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Ajuster l'espacement et afficher ou sauvegarder
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique OBV sauvegardé dans '{save_path}'")
    else:
        plt.show()
    
    plt.close()


@timeit
def plot_comparison(
    data_list: List[pd.DataFrame],
    labels: List[str],
    title: str = "Comparaison des stratégies OBV",
    save_path: str = None
) -> None:
    """
    Compare différentes variantes de la stratégie OBV.
    
    Args:
        data_list: Liste de DataFrames contenant les résultats de différentes stratégies
        labels: Liste des étiquettes pour chaque stratégie
        title: Titre du graphique
        save_path: Chemin pour sauvegarder l'image (si None, affiche le graphique)
    """
    # Vérifier que les listes ont la même longueur
    if len(data_list) != len(labels):
        raise ValueError("Les listes data_list et labels doivent avoir la même longueur")
    
    # Créer une figure avec 2 subplots partagant l'axe des x
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Convertir l'index en datetime si nécessaire
    if not isinstance(data_list[0].index, pd.DatetimeIndex):
        for i in range(len(data_list)):
            data_list[i].index = pd.to_datetime(data_list[i].index)
    
    # Format de date pour l'axe des x
    date_format = mdates.DateFormatter('%Y-%m-%d')
    
    # Configurer le premier subplot pour le prix
    ax1.set_title(title, fontsize=16)
    ax1.plot(data_list[0].index, data_list[0]['price'], label='Prix de clôture', color='black', alpha=0.7)
    
    # Couleurs pour les différentes stratégies
    colors = ['green', 'red', 'blue', 'purple', 'orange', 'brown', 'pink']
    
    # Tracer les signaux d'achat/vente pour chaque stratégie
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color = colors[i % len(colors)]
        
        # Signaux d'achat
        if 'signal_buy' in data.columns:
            buy_signals = data[data['signal_buy']]
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals['price'], 
                           color=color, marker='^', s=100, alpha=0.7,
                           label=f'{label} - Achat')
        
        # Signaux de vente
        if 'signal_sell' in data.columns:
            sell_signals = data[data['signal_sell']]
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals['price'], 
                           color=color, marker='v', s=100, alpha=0.7,
                           label=f'{label} - Vente')
    
    ax1.set_ylabel('Prix', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Configurer le deuxième subplot pour l'OBV
    ax2.set_title('Comparaison des indicateurs OBV', fontsize=14)
    
    # Tracer l'OBV pour chaque stratégie
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color = colors[i % len(colors)]
        
        # OBV brut ou normalisé
        if 'obv' in data.columns:
            # Normaliser l'OBV pour une meilleure comparaison
            normalized_obv = (data['obv'] - data['obv'].min()) / (data['obv'].max() - data['obv'].min())
            ax2.plot(data.index, normalized_obv, label=f'OBV {label}', color=color)
    
    ax2.set_ylabel('OBV Normalisé', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Configurer l'axe des x
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Ajuster l'espacement et afficher ou sauvegarder
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique de comparaison OBV sauvegardé dans '{save_path}'")
    else:
        plt.show()
    
    plt.close()


@timeit
def evaluate_signals(data: pd.DataFrame, signal_column: str, periods: List[int] = [5, 10, 20]) -> Dict:
    """
    Évalue les performances des signaux de trading.
    
    Args:
        data: DataFrame contenant les signaux et les prix
        signal_column: Nom de la colonne contenant les signaux (True/False)
        periods: Liste des périodes futures à évaluer
        
    Returns:
        Dictionnaire contenant les statistiques de performance
    """
    if data.empty or signal_column not in data.columns or 'price' not in data.columns:
        logger.warning(f"Données insuffisantes pour évaluer les signaux {signal_column}")
        return {}
    
    results = {}
    
    # Nombre total de signaux
    signal_indices = data.index[data[signal_column]]
    results['count'] = len(signal_indices)
    
    if results['count'] == 0:
        logger.warning(f"Aucun signal {signal_column} trouvé dans les données")
        return results
    
    # Évaluer chaque période
    for period in periods:
        changes = []
        positive_count = 0
        
        for signal_idx in signal_indices:
            # Trouver l'index de la période future
            signal_idx_pos = data.index.get_loc(signal_idx)
            
            # Vérifier si nous avons suffisamment de données pour la période future
            if signal_idx_pos + period < len(data):
                # Obtenir le prix au moment du signal
                price_at_signal = data.loc[signal_idx, 'price']
                
                # Obtenir le prix après la période
                future_idx = data.index[signal_idx_pos + period]
                future_price = data.loc[future_idx, 'price']
                
                # Calculer le changement de prix en pourcentage
                if 'signal_sell' in signal_column:
                    # Pour les signaux de vente, on inverse le calcul
                    pct_change = ((price_at_signal - future_price) / price_at_signal) * 100
                else:
                    # Pour les signaux d'achat
                    pct_change = ((future_price - price_at_signal) / price_at_signal) * 100
                
                changes.append(pct_change)
                if pct_change > 0:
                    positive_count += 1
        
        if changes:
            # Taux de réussite (% de signaux donnant un résultat positif)
            results[f'positive_rate_{period}'] = (positive_count / len(changes)) * 100
            
            # Changement moyen après la période
            results[f'avg_change_{period}'] = sum(changes) / len(changes)
            
            # Min et max
            results[f'min_change_{period}'] = min(changes)
            results[f'max_change_{period}'] = max(changes)
    
    return results


@timeit
def main():
    """
    Fonction principale pour l'exemple OBV.
    """
    # Charger les données
    data = load_sample_data(
        symbol="BTC/USDT",
        timeframe="1d",
        lookback_days=365
    )
    
    # Initialiser les stratégies
    logger.info("\nAnalyse avec l'OBV standard...")
    standard_strategy = OBVStrategy(window=20, signal_window=9)
    obv_indicators, standard_results = standard_strategy.apply(data)
    
    # Visualiser les résultats
    plot_obv_results(standard_results, 
                    title="Analyse OBV Standard",
                    save_path="obv_standard.png")
    
    # Stratégie spécialisée dans les divergences
    logger.info("\nAnalyse des divergences OBV...")
    divergence_strategy = OBVDivergenceStrategy(
        window=20, 
        signal_window=9,
        divergence_window=20,
        divergence_threshold=0.1
    )
    _, divergence_results = divergence_strategy.apply(data)
    
    plot_obv_results(divergence_results, 
                    title="Analyse des Divergences OBV",
                    save_path="obv_divergence.png")
    
    # Stratégie basée sur le taux de variation
    logger.info("\nAnalyse du taux de variation de l'OBV...")
    roc_strategy = RateOfChangeOBVStrategy(
        window=20,
        signal_window=9,
        roc_period=14,
        roc_threshold=0.05
    )
    _, roc_results = roc_strategy.apply(data)
    
    plot_obv_results(roc_results, 
                     title="Analyse du taux de variation de l'OBV",
                     save_path="obv_roc.png")
    
    # Comparer les stratégies
    logger.info("\nComparaison des stratégies OBV...")
    plot_comparison(
        [standard_results, divergence_results, roc_results],
        ["Standard", "Divergence", "ROC"],
        title="Comparaison des Stratégies OBV",
        save_path="obv_comparison.png"
    )
    
    # Analyser les signaux de croisement OBV
    logger.info("\nAnalyse des croisements OBV...")
    crossover_up_count = standard_results['signal_buy'].sum()
    crossover_down_count = standard_results['signal_sell'].sum()
    
    logger.info(f"Nombre de croisements OBV à la hausse: {crossover_up_count}")
    logger.info(f"Nombre de croisements OBV à la baisse: {crossover_down_count}")
    
    # Évaluation des performances des signaux
    logger.info("\nÉvaluation des performances des signaux...")
    
    # Signaux d'achat standard
    buy_stats = evaluate_signals(standard_results, 'signal_buy')
    if buy_stats:
        logger.info("\nPerformance des signaux d'achat standard:")
        logger.info(f"Nombre de signaux: {buy_stats['count']}")
        if buy_stats['count'] > 0:
            logger.info(f"Taux de réussite après 5 périodes: {buy_stats.get('positive_rate_5', 0):.2f}%")
            logger.info(f"Changement moyen après 5 périodes: {buy_stats.get('avg_change_5', 0):.2f}%")
            logger.info(f"Changement moyen après 10 périodes: {buy_stats.get('avg_change_10', 0):.2f}%")
        else:
            logger.info("Aucune statistique disponible (pas assez de signaux)")
    
    # Signaux de vente standard
    sell_stats = evaluate_signals(standard_results, 'signal_sell')
    if sell_stats:
        logger.info("\nPerformance des signaux de vente standard:")
        logger.info(f"Nombre de signaux: {sell_stats['count']}")
        if sell_stats['count'] > 0:
            logger.info(f"Taux de réussite après 5 périodes: {sell_stats.get('positive_rate_5', 0):.2f}%")
            logger.info(f"Changement moyen après 5 périodes: {sell_stats.get('avg_change_5', 0):.2f}%")
            logger.info(f"Changement moyen après 10 périodes: {sell_stats.get('avg_change_10', 0):.2f}%")
        else:
            logger.info("Aucune statistique disponible (pas assez de signaux)")
    
    # Performance des signaux de divergence
    divergence_buy_stats = evaluate_signals(divergence_results, 'signal_buy')
    if divergence_buy_stats:
        logger.info("\nPerformance des signaux de divergence haussière:")
        logger.info(f"Nombre de signaux: {divergence_buy_stats['count']}")
        if divergence_buy_stats['count'] > 0:
            logger.info(f"Taux de réussite après 5 périodes: {divergence_buy_stats.get('positive_rate_5', 0):.2f}%")
            logger.info(f"Changement moyen après 5 périodes: {divergence_buy_stats.get('avg_change_5', 0):.2f}%")
            logger.info(f"Changement moyen après 10 périodes: {divergence_buy_stats.get('avg_change_10', 0):.2f}%")
        else:
            logger.info("Aucune statistique disponible (pas assez de signaux)")
    
    # Performance des signaux ROC
    roc_buy_stats = evaluate_signals(roc_results, 'signal_buy')
    if roc_buy_stats:
        logger.info("\nPerformance des signaux ROC à l'achat:")
        logger.info(f"Nombre de signaux: {roc_buy_stats['count']}")
        if roc_buy_stats['count'] > 0:
            logger.info(f"Taux de réussite après 5 périodes: {roc_buy_stats.get('positive_rate_5', 0):.2f}%")
            logger.info(f"Changement moyen après 5 périodes: {roc_buy_stats.get('avg_change_5', 0):.2f}%")
            logger.info(f"Changement moyen après 10 périodes: {roc_buy_stats.get('avg_change_10', 0):.2f}%")
        else:
            logger.info("Aucune statistique disponible (pas assez de signaux)")
    
    # Conclusion
    logger.info("\nAnalyse OBV complétée avec succès!")


if __name__ == "__main__":
    main()
