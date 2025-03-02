"""
Exemple d'utilisation de l'indicateur VWAP et de la stratégie VWAP.

Ce script montre comment :
1. Initialiser l'indicateur VWAP et calculer les valeurs.
2. Traiter les cas de données manquantes avec différentes stratégies.
3. Utiliser la stratégie VWAP pour générer des signaux.
4. Visualiser le VWAP, ses bandes, et les signaux générés.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# Ajouter le chemin du projet pour l'importation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bitbot.strategie.indicators.vwap import (
    VWAPIndicator, VWAPStrategy, VWAPTimeFrame, MissingDataStrategy
)
from bitbot.models.market_data import MarketData

# Créer des données de test
def create_test_data(n_periods=100, missing_data=False):
    """Crée des données OHLCV de test."""
    np.random.seed(42)
    
    # Paramètres de base
    base_price = 100
    volatility = 2.0
    volume_base = 1000
    
    # Générer les prix
    prices = np.cumsum(np.random.normal(0, volatility, n_periods))
    prices = base_price + prices
    
    # Générer un trend sinusoïdal
    trend = 10 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    prices = prices + trend
    
    # Créer le DataFrame
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_periods), periods=n_periods, freq='D')
    
    df = pd.DataFrame(index=dates)
    df['open'] = prices
    df['high'] = prices + np.random.uniform(0.5, 2.0, n_periods)
    df['low'] = prices - np.random.uniform(0.5, 2.0, n_periods)
    df['close'] = prices + np.random.normal(0, 0.5, n_periods)
    
    # Générer les volumes avec des pics pour simuler des zones d'intérêt
    volumes = np.random.normal(volume_base, volume_base * 0.2, n_periods)
    
    # Ajouter des pics de volume à certains moments
    volume_spikes = np.random.randint(0, n_periods, 10)
    for spike in volume_spikes:
        volumes[spike] = volumes[spike] * np.random.uniform(2, 5)
    
    df['volume'] = volumes
    
    # Simuler des données manquantes si demandé
    if missing_data:
        # Supprimer des volumes aléatoirement
        volume_missing = np.random.choice(
            np.arange(n_periods), 
            size=int(n_periods * 0.1),  # 10% des données
            replace=False
        )
        df.loc[df.index[volume_missing], 'volume'] = np.nan
        
        # Supprimer des prix hauts/bas aléatoirement
        price_missing = np.random.choice(
            np.arange(n_periods), 
            size=int(n_periods * 0.05),  # 5% des données
            replace=False
        )
        df.loc[df.index[price_missing], ['high', 'low']] = np.nan
    
    return df


def demonstrate_vwap_indicator():
    """Démontre l'utilisation de l'indicateur VWAP avec différentes configurations."""
    print("\n" + "="*50)
    print("Démonstration de l'indicateur VWAP")
    print("="*50)
    
    # Créer des données de test
    df = create_test_data(n_periods=60)
    
    # 1. Exemple basique - VWAP journalier
    vwap_daily = VWAPIndicator(time_frame=VWAPTimeFrame.DAILY)
    df_vwap = vwap_daily.calculate_vwap(df)
    
    print(f"\n1. VWAP journalier calculé sur {len(df_vwap)} périodes.")
    print(f"   Dernière valeur VWAP: {df_vwap['vwap'].iloc[-1]:.2f}")
    print(f"   Bande supérieure: {df_vwap['upper_band'].iloc[-1]:.2f}")
    print(f"   Bande inférieure: {df_vwap['lower_band'].iloc[-1]:.2f}")
    
    # 2. VWAP avec différentes périodes
    vwap_weekly = VWAPIndicator(time_frame=VWAPTimeFrame.WEEKLY)
    df_weekly = vwap_weekly.calculate_vwap(df)
    
    print("\n2. Comparaison des périodes VWAP:")
    print(f"   VWAP journalier: {df_vwap['vwap'].iloc[-1]:.2f}")
    print(f"   VWAP hebdomadaire: {df_weekly['vwap'].iloc[-1]:.2f}")
    
    # 3. Identifier les niveaux de support et résistance
    levels = vwap_daily.identify_support_resistance(df)
    
    print("\n3. Niveaux de support et résistance identifiés:")
    print(f"   Supports: {[f'{level:.2f}' for level in levels['supports']]}")
    print(f"   Résistances: {[f'{level:.2f}' for level in levels['resistances']]}")
    
    # 4. Tester avec des données manquantes
    df_missing = create_test_data(n_periods=60, missing_data=True)
    
    # Tester les différentes stratégies de données manquantes
    strategies = [
        MissingDataStrategy.FAIL,
        MissingDataStrategy.INTERPOLATE,
        MissingDataStrategy.PREVIOUS,
        MissingDataStrategy.FALLBACK
    ]
    
    print("\n4. Gestion des données manquantes:")
    
    for strategy in strategies:
        try:
            vwap_handler = VWAPIndicator(missing_data_strategy=strategy)
            df_handled = vwap_handler.calculate_vwap(df_missing)
            
            if df_handled is not None:
                print(f"   Stratégie {strategy.value}: VWAP calculé avec succès.")
                if 'vwap_fallback' in df_handled.columns and not df_handled['vwap_fallback'].isna().all():
                    print(f"     ⚠️ Indicateur de remplacement utilisé!")
            else:
                print(f"   Stratégie {strategy.value}: Échec du calcul.")
        except Exception as e:
            print(f"   Stratégie {strategy.value}: Erreur - {str(e)}")
    
    # 5. Visualiser
    plt.figure(figsize=(12, 6))
    plt.plot(df_vwap.index, df_vwap['close'], label='Prix de clôture', color='black')
    plt.plot(df_vwap.index, df_vwap['vwap'], label='VWAP', color='blue', linewidth=2)
    plt.plot(df_vwap.index, df_vwap['upper_band'], label='Bande supérieure', color='red', linestyle='--')
    plt.plot(df_vwap.index, df_vwap['lower_band'], label='Bande inférieure', color='green', linestyle='--')
    
    plt.fill_between(df_vwap.index, df_vwap['lower_band'], df_vwap['upper_band'], color='blue', alpha=0.1)
    
    # Ajouter les supports et résistances
    for support in levels['supports']:
        plt.axhline(y=support, color='green', linestyle='-', alpha=0.4)
    
    for resistance in levels['resistances']:
        plt.axhline(y=resistance, color='red', linestyle='-', alpha=0.4)
    
    plt.title('VWAP avec bandes et niveaux de support/résistance')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vwap_indicator_example.png')
    print("\n   Graphique sauvegardé sous 'vwap_indicator_example.png'")


def demonstrate_vwap_strategy():
    """Démontre l'utilisation de la stratégie VWAP pour générer des signaux."""
    print("\n" + "="*50)
    print("Démonstration de la stratégie VWAP")
    print("="*50)
    
    # Créer des données de test
    df = create_test_data(n_periods=60)
    
    # 1. Mode retour à la moyenne (mode par défaut)
    vwap_reversion = VWAPStrategy(reversion_mode=True)
    df_reversion = vwap_reversion.calculate_signals(df)
    
    # 2. Mode suivi de tendance
    vwap_trend = VWAPStrategy(reversion_mode=False)
    df_trend = vwap_trend.calculate_signals(df)
    
    # Compter les signaux
    reversion_buys = (df_reversion['vwap_signal'] == 1).sum()
    reversion_sells = (df_reversion['vwap_signal'] == -1).sum()
    
    trend_buys = (df_trend['vwap_signal'] == 1).sum()
    trend_sells = (df_trend['vwap_signal'] == -1).sum()
    
    print("\n1. Comparaison des modes de stratégie:")
    print(f"   Mode retour à la moyenne: {reversion_buys} achats, {reversion_sells} ventes")
    print(f"   Mode suivi de tendance: {trend_buys} achats, {trend_sells} ventes")
    
    # 3. Générer des signaux de trading complets
    signals = vwap_reversion.generate_trading_signals(df)
    
    print("\n2. Informations du dernier signal:")
    print(f"   Signal: {signals['signal']} ({vwap_reversion.get_signal_description(signals)})")
    print(f"   VWAP: {signals['vwap']:.2f}")
    print(f"   Bandes: {signals['lower_band']:.2f} - {signals['upper_band']:.2f}")
    
    # 4. Visualiser les différentes stratégies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Tracer la stratégie de retour à la moyenne
    vwap_reversion.plot_vwap(df, ax=ax1)
    ax1.set_title("Stratégie VWAP - Mode Retour à la Moyenne")
    
    # Tracer la stratégie de suivi de tendance
    vwap_trend.plot_vwap(df, ax=ax2)
    ax2.set_title("Stratégie VWAP - Mode Suivi de Tendance")
    
    plt.tight_layout()
    plt.savefig('vwap_strategy_comparison.png')
    print("\n   Graphique comparatif sauvegardé sous 'vwap_strategy_comparison.png'")


def demonstrate_vwap_fallback():
    """Démontre le mécanisme de fallback lorsque les données sont manquantes."""
    print("\n" + "="*50)
    print("Démonstration du mécanisme de fallback")
    print("="*50)
    
    # Créer des données manquantes
    df_complete = create_test_data(n_periods=60)
    df_incomplete = df_complete.copy()
    
    # Retirer des colonnes essentielles
    df_no_volume = df_complete.copy()
    df_no_volume.drop(columns=['volume'], inplace=True)
    
    df_no_high_low = df_complete.copy()
    df_no_high_low.drop(columns=['high', 'low'], inplace=True)
    
    # Initialiser l'indicateur avec stratégie de fallback
    vwap_fallback = VWAPIndicator(missing_data_strategy=MissingDataStrategy.FALLBACK)
    
    # Traiter les différents cas
    results = {
        'Données complètes': vwap_fallback.calculate_vwap(df_complete),
        'Sans volume': vwap_fallback.calculate_vwap(df_no_volume),
        'Sans high/low': vwap_fallback.calculate_vwap(df_no_high_low)
    }
    
    print("\nRésultats avec différentes données manquantes:")
    
    for name, df_result in results.items():
        if df_result is not None:
            vwap_col = 'vwap'
            fallback_used = False
            
            if 'vwap_fallback' in df_result.columns and not df_result['vwap_fallback'].isna().all():
                vwap_col = 'vwap_fallback'
                fallback_used = True
            
            print(f"\n{name}:")
            print(f"  Fallback utilisé: {'Oui' if fallback_used else 'Non'}")
            print(f"  Valeur VWAP: {df_result[vwap_col].iloc[-1]:.2f}")
            print(f"  Bande supérieure: {df_result['upper_band'].iloc[-1]:.2f}")
            print(f"  Bande inférieure: {df_result['lower_band'].iloc[-1]:.2f}")
        else:
            print(f"\n{name}: Échec du calcul")
    
    # Visualiser la comparaison
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), sharex=True)
    
    for i, (name, df_result) in enumerate(results.items()):
        ax = axes[i] if len(results) > 1 else axes
        
        if df_result is not None:
            ax.plot(df_result.index, df_result['close'], label='Prix', color='black')
            
            vwap_col = 'vwap'
            if 'vwap_fallback' in df_result.columns and not df_result['vwap_fallback'].isna().all():
                vwap_col = 'vwap_fallback'
                label = 'Fallback (MA pondérée)'
            else:
                label = 'VWAP'
                
            ax.plot(df_result.index, df_result[vwap_col], label=label, color='blue')
            ax.plot(df_result.index, df_result['upper_band'], label='Bande sup.', color='red', linestyle='--')
            ax.plot(df_result.index, df_result['lower_band'], label='Bande inf.', color='green', linestyle='--')
            
            ax.fill_between(df_result.index, df_result['lower_band'], df_result['upper_band'], color='blue', alpha=0.1)
            
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vwap_fallback_comparison.png')
    print("\n   Graphique de comparaison des fallbacks sauvegardé sous 'vwap_fallback_comparison.png'")


if __name__ == "__main__":
    demonstrate_vwap_indicator()
    demonstrate_vwap_strategy()
    demonstrate_vwap_fallback()
    
    print("\nExemple terminé !")
