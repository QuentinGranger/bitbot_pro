"""
Module pour analyser les flux de crypto-monnaies entrant et sortant des exchanges.

Ce module fournit des fonctionnalités pour surveiller les mouvements de
Bitcoin entre les adresses d'exchanges et les adresses personnelles,
ce qui peut être un indicateur avancé des mouvements de prix.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from bitbot.data.onchain_client import OnChainClient
from bitbot.utils.logger import logger

class NetflowSignal(Enum):
    """Signaux basés sur les flux d'échange."""
    STRONG_OUTFLOW = "Fort flux sortant"  # Bullish (les BTC quittent les exchanges pour le stockage long terme)
    OUTFLOW = "Flux sortant"  # Légèrement bullish
    NEUTRAL = "Neutre"  # Pas de signal clair
    INFLOW = "Flux entrant"  # Légèrement bearish
    STRONG_INFLOW = "Fort flux entrant"  # Très bearish (les BTC arrivent en masse sur les exchanges pour être vendus)

class ExchangeNetflow:
    """
    Indicateur pour analyser les flux de crypto-monnaies entrant et sortant des exchanges.
    
    Le netflow des exchanges (différence entre les entrées et les sorties) est un indicateur 
    avancé important car:
    - Un netflow négatif signifie que les bitcoins quittent les exchanges, indiquant une 
      accumulation et généralement un sentiment haussier.
    - Un netflow positif signifie que les bitcoins entrent dans les exchanges, souvent pour 
      être vendus, indiquant un sentiment baissier.
    """
    
    def __init__(self,
                ema_period: int = 14,
                outflow_threshold: float = -1000,  # BTC
                strong_outflow_threshold: float = -5000,  # BTC
                inflow_threshold: float = 1000,  # BTC
                strong_inflow_threshold: float = 5000):  # BTC
        """
        Initialise l'indicateur de flux d'échange.
        
        Args:
            ema_period: Période pour le calcul de l'EMA du netflow.
            outflow_threshold: Seuil pour considérer un flux sortant significatif.
            strong_outflow_threshold: Seuil pour considérer un flux sortant fort.
            inflow_threshold: Seuil pour considérer un flux entrant significatif.
            strong_inflow_threshold: Seuil pour considérer un flux entrant fort.
        """
        self.client = OnChainClient()
        self.ema_period = ema_period
        self.outflow_threshold = outflow_threshold
        self.strong_outflow_threshold = strong_outflow_threshold
        self.inflow_threshold = inflow_threshold
        self.strong_inflow_threshold = strong_inflow_threshold
        
        logger.info(f"Indicateur Exchange Netflow initialisé avec les paramètres: "
                   f"ema_period={ema_period}, "
                   f"outflow_threshold={outflow_threshold}, "
                   f"strong_outflow_threshold={strong_outflow_threshold}, "
                   f"inflow_threshold={inflow_threshold}, "
                   f"strong_inflow_threshold={strong_inflow_threshold}")
    
    def get_netflow_data(self, asset: str = "BTC", days: int = 30) -> pd.DataFrame:
        """
        Récupère les données de flux d'échange pour un actif.
        
        Args:
            asset: Symbole de l'actif (BTC, ETH, etc.).
            days: Nombre de jours de données à récupérer.
            
        Returns:
            DataFrame contenant les données de flux d'échange.
        """
        try:
            logger.info(f"Récupération des données de flux d'échange pour {asset} (derniers {days} jours)")
            
            # Récupérer les données de flux d'échange via l'API blockchain.info
            # Pour cette démonstration, nous allons simuler les données de flux d'échange
            # En pratique, nous utiliserions une API comme blockchain.info, blockchair, etc.
            netflow_data = self._get_approximate_netflow(asset, days)
            
            if netflow_data.empty:
                logger.warning(f"Aucune donnée de flux d'échange récupérée pour {asset}")
                return pd.DataFrame()
            
            # Calculer l'EMA du netflow
            netflow_data['netflow_ema'] = netflow_data['netflow'].ewm(span=self.ema_period, adjust=False).mean()
            
            return netflow_data
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de flux d'échange: {e}")
            return pd.DataFrame()
    
    def _get_approximate_netflow(self, asset: str, days: int) -> pd.DataFrame:
        """
        Obtient des données approximatives de flux d'échange en utilisant des API gratuites.
        
        Dans un environnement de production, cette méthode devrait être remplacée par une
        intégration avec une API qui fournit des données réelles de flux d'échange, comme 
        blockchain.info, blockchair ou Glassnode.
        
        Args:
            asset: Symbole de l'actif (BTC, ETH, etc.).
            days: Nombre de jours de données à récupérer.
            
        Returns:
            DataFrame contenant les données approximatives de flux d'échange.
        """
        try:
            # Récupérer les données de marché pour l'actif
            market_data = self.client.get_market_data(asset=asset, days=days)
            
            if market_data.empty:
                logger.warning(f"Aucune donnée de marché pour {asset}")
                return pd.DataFrame()
            
            # Pour cette démonstration, nous allons simuler les données de flux d'échange
            # en utilisant une corrélation inverse avec les variations de prix
            # (quand les prix baissent, les entrées augmentent et inversement)
            df_netflow = pd.DataFrame()
            df_netflow['timestamp'] = market_data['timestamp']
            df_netflow['price'] = market_data['price']
            
            # Calculer les variations de prix
            df_netflow['price_change'] = df_netflow['price'].pct_change()
            
            # Simuler le netflow basé sur les variations de prix (corrélation inverse)
            # Quand le prix baisse, le netflow tend à être positif (entrées dans les exchanges)
            # Quand le prix monte, le netflow tend à être négatif (sorties des exchanges)
            # Cette simulation est simplifiée et ne reflète pas la complexité des données réelles
            price_volatility = df_netflow['price'].std() / df_netflow['price'].mean()
            base_flow = 1000.0 * price_volatility * np.sqrt(market_data['market_cap'].iloc[-1] / 1e12)
            
            # Générer des valeurs de netflow qui varient en fonction des changements de prix
            # et incluent également un composant aléatoire pour plus de réalisme
            df_netflow['netflow'] = -base_flow * df_netflow['price_change'] * 100
            
            # Ajouter une composante aléatoire pour simuler la volatilité naturelle des flux
            np.random.seed(42)  # Pour la reproductibilité
            random_factor = np.random.normal(0, base_flow / 2, size=len(df_netflow))
            df_netflow['netflow'] = df_netflow['netflow'] + random_factor
            
            # Remplacer NaN par 0
            df_netflow['netflow'] = df_netflow['netflow'].fillna(0)
            
            # Pour montrer des tendances, ajouter une composante de tendance
            trend = np.linspace(-base_flow/10, base_flow/10, len(df_netflow))
            df_netflow['netflow'] = df_netflow['netflow'] + trend
            
            return df_netflow
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données approximatives de flux d'échange: {e}")
            return pd.DataFrame()
    
    def analyze_netflow(self, netflow_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse les données de flux d'échange et génère des signaux.
        
        Args:
            netflow_data: DataFrame contenant les données de flux d'échange.
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse.
        """
        if netflow_data.empty:
            logger.warning("Aucune donnée de flux d'échange pour l'analyse")
            return {}
        
        # Dernière valeur de netflow
        current_netflow = netflow_data['netflow'].iloc[-1]
        current_ema = netflow_data['netflow_ema'].iloc[-1]
        
        # Statistiques descriptives
        netflow_min = netflow_data['netflow'].min()
        netflow_max = netflow_data['netflow'].max()
        netflow_mean = netflow_data['netflow'].mean()
        netflow_median = netflow_data['netflow'].median()
        
        # Déterminer le signal
        signal = self._determine_signal(current_netflow)
        
        # Calculer les distances aux seuils (en pourcentage)
        distance_to_outflow = 100 * (current_netflow - self.outflow_threshold) / abs(self.outflow_threshold) if self.outflow_threshold != 0 else float('inf')
        distance_to_inflow = 100 * (current_netflow - self.inflow_threshold) / abs(self.inflow_threshold) if self.inflow_threshold != 0 else float('inf')
        
        # Tendance du netflow (basée sur la comparaison avec l'EMA)
        netflow_trend = "Augmentation" if current_netflow > current_ema else "Diminution"
        
        # Accumulation des derniers jours
        recent_days = min(7, len(netflow_data))
        recent_netflow_sum = netflow_data['netflow'].iloc[-recent_days:].sum()
        
        return {
            "signal": signal.value,
            "current_netflow": current_netflow,
            "netflow_ema": current_ema,
            "netflow_min": netflow_min,
            "netflow_max": netflow_max,
            "netflow_mean": netflow_mean,
            "netflow_median": netflow_median,
            "distance_to_outflow": distance_to_outflow,
            "distance_to_inflow": distance_to_inflow,
            "netflow_trend": netflow_trend,
            "recent_accumulated_netflow": recent_netflow_sum,
            "is_outflow": current_netflow < self.outflow_threshold,
            "is_strong_outflow": current_netflow < self.strong_outflow_threshold,
            "is_inflow": current_netflow > self.inflow_threshold,
            "is_strong_inflow": current_netflow > self.strong_inflow_threshold
        }
    
    def _determine_signal(self, netflow: float) -> NetflowSignal:
        """
        Détermine le signal de flux d'échange basé sur la valeur de netflow.
        
        Args:
            netflow: Valeur du netflow.
            
        Returns:
            Signal de flux d'échange.
        """
        if netflow < self.strong_outflow_threshold:
            return NetflowSignal.STRONG_OUTFLOW
        elif netflow < self.outflow_threshold:
            return NetflowSignal.OUTFLOW
        elif netflow > self.strong_inflow_threshold:
            return NetflowSignal.STRONG_INFLOW
        elif netflow > self.inflow_threshold:
            return NetflowSignal.INFLOW
        else:
            return NetflowSignal.NEUTRAL
    
    def plot_netflow(self, netflow_data: pd.DataFrame, asset: str = "BTC", output_path: Optional[str] = None) -> None:
        """
        Génère un graphique des données de flux d'échange.
        
        Args:
            netflow_data: DataFrame contenant les données de flux d'échange.
            asset: Symbole de l'actif.
            output_path: Chemin pour enregistrer le graphique. Si None, le graphique sera affiché.
        """
        if netflow_data.empty:
            logger.warning("Aucune donnée de flux d'échange pour la génération du graphique")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Sous-graphique 1: Netflow et EMA
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(netflow_data['timestamp'], netflow_data['netflow'], label='Netflow', color='blue')
        plt.plot(netflow_data['timestamp'], netflow_data['netflow_ema'], label=f'EMA ({self.ema_period})', color='orange')
        
        # Ajouter des lignes pour les seuils
        plt.axhline(y=self.outflow_threshold, color='green', linestyle='--', alpha=0.7, label='Seuil de sortie')
        plt.axhline(y=self.strong_outflow_threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Seuil de forte sortie')
        plt.axhline(y=self.inflow_threshold, color='red', linestyle='--', alpha=0.7, label='Seuil d\'entrée')
        plt.axhline(y=self.strong_inflow_threshold, color='darkred', linestyle='--', alpha=0.7, label='Seuil de forte entrée')
        
        # Remplir les zones entre les seuils
        plt.fill_between(netflow_data['timestamp'], self.strong_outflow_threshold, self.outflow_threshold, 
                         color='green', alpha=0.1)
        plt.fill_between(netflow_data['timestamp'], self.outflow_threshold, self.inflow_threshold, 
                         color='yellow', alpha=0.1)
        plt.fill_between(netflow_data['timestamp'], self.inflow_threshold, self.strong_inflow_threshold, 
                         color='red', alpha=0.1)
        
        plt.title(f'Flux d\'échange pour {asset}')
        plt.ylabel('Netflow (BTC)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Sous-graphique 2: Prix
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(netflow_data['timestamp'], netflow_data['price'], label='Prix', color='purple')
        plt.title(f'Prix de {asset}')
        plt.xlabel('Date')
        plt.ylabel('Prix (USD)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Graphique enregistré: {output_path}")
        else:
            plt.show()
        
        plt.close()
