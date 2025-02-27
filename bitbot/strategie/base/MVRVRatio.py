"""
Module pour analyser le ratio MVRV (Market Value to Realized Value).

Ce module fournit des fonctionnalités pour évaluer si le Bitcoin est
surévalué ou sous-évalué par rapport à sa valeur réalisée.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from bitbot.data.onchain_client import OnChainClient
from bitbot.utils.logger import logger

class MVRVSignal(Enum):
    """Signaux basés sur le ratio MVRV."""
    STRONG_UNDERVALUED = "Fortement sous-évalué"
    UNDERVALUED = "Sous-évalué"
    NEUTRAL = "Neutre"
    OVERVALUED = "Surévalué"
    STRONG_OVERVALUED = "Fortement surévalué"

class MVRVIndicator:
    """
    Indicateur pour analyser le ratio MVRV (Market Value to Realized Value).
    
    Le ratio MVRV est un indicateur qui compare la capitalisation boursière (Market Cap)
    à la capitalisation réalisée (Realized Cap). Il permet d'évaluer si le Bitcoin
    est surévalué ou sous-évalué par rapport à sa valeur réalisée.
    """
    
    def __init__(self, 
                ema_period: int = 50,
                undervalued_threshold: float = 1.0,
                strong_undervalued_threshold: float = 0.75,
                overvalued_threshold: float = 2.5,
                strong_overvalued_threshold: float = 3.5):
        """
        Initialise l'indicateur MVRV.
        
        Args:
            ema_period: Période pour le calcul de l'EMA du ratio MVRV.
            undervalued_threshold: Seuil pour considérer le marché comme sous-évalué.
            strong_undervalued_threshold: Seuil pour considérer le marché comme fortement sous-évalué.
            overvalued_threshold: Seuil pour considérer le marché comme surévalué.
            strong_overvalued_threshold: Seuil pour considérer le marché comme fortement surévalué.
        """
        self.client = OnChainClient()
        self.ema_period = ema_period
        self.undervalued_threshold = undervalued_threshold
        self.strong_undervalued_threshold = strong_undervalued_threshold
        self.overvalued_threshold = overvalued_threshold
        self.strong_overvalued_threshold = strong_overvalued_threshold
        
        logger.info(f"Indicateur MVRV initialisé avec les paramètres: "
                   f"ema_period={ema_period}, "
                   f"undervalued_threshold={undervalued_threshold}, "
                   f"strong_undervalued_threshold={strong_undervalued_threshold}, "
                   f"overvalued_threshold={overvalued_threshold}, "
                   f"strong_overvalued_threshold={strong_overvalued_threshold}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Définit les paramètres de l'indicateur.
        
        Args:
            **kwargs: Paramètres à définir.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Paramètres de l'indicateur MVRV mis à jour: {kwargs}")
    
    def get_mvrv_data(self, 
                     asset: str = "BTC", 
                     days: int = 365,
                     until: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Récupère les données MVRV approximatives pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer les données MVRV (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            DataFrame contenant les données MVRV.
        """
        # Notes sur l'utilisation d'API gratuites
        logger.info("Utilisation d'API gratuites pour calculer une approximation du MVRV")
        
        # Récupérer les données MVRV approximatives
        df_mvrv = self.client.get_approximate_mvrv_ratio(asset=asset, days=days)
        
        if df_mvrv.empty:
            logger.warning(f"Aucune donnée MVRV récupérée pour {asset}")
            return pd.DataFrame()
        
        # Calculer l'EMA
        df_mvrv['mvrv_ema'] = df_mvrv['mvrv_ratio'].ewm(span=self.ema_period, adjust=False).mean()
        
        return df_mvrv
    
    def calculate_mvrv_from_caps(self, 
                               asset: str = "BTC", 
                               days: int = 365,
                               until: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Calcule le ratio MVRV à partir des capitalisations boursière et réalisée.
        
        Args:
            asset: Actif pour lequel calculer le ratio MVRV (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            DataFrame contenant les données MVRV calculées.
        """
        # Cette méthode est maintenant équivalente à get_mvrv_data car nous utilisons
        # l'approximation basée sur les données de marché de CoinGecko
        return self.get_mvrv_data(asset=asset, days=days, until=until)
    
    def get_signal(self, mvrv_data: pd.DataFrame) -> MVRVSignal:
        """
        Génère un signal basé sur le ratio MVRV.
        
        Args:
            mvrv_data: DataFrame contenant les données MVRV.
            
        Returns:
            Signal MVRV.
        """
        if mvrv_data.empty:
            logger.warning("Aucune donnée MVRV disponible pour générer un signal")
            return MVRVSignal.NEUTRAL
        
        # Récupérer la dernière valeur du ratio MVRV
        last_mvrv = mvrv_data['mvrv_ratio'].iloc[-1]
        
        # Déterminer le signal
        if last_mvrv <= self.strong_undervalued_threshold:
            return MVRVSignal.STRONG_UNDERVALUED
        elif last_mvrv <= self.undervalued_threshold:
            return MVRVSignal.UNDERVALUED
        elif last_mvrv >= self.strong_overvalued_threshold:
            return MVRVSignal.STRONG_OVERVALUED
        elif last_mvrv >= self.overvalued_threshold:
            return MVRVSignal.OVERVALUED
        else:
            return MVRVSignal.NEUTRAL
    
    def is_undervalued(self, mvrv_data: pd.DataFrame) -> bool:
        """
        Vérifie si le marché est sous-évalué selon le ratio MVRV.
        
        Args:
            mvrv_data: DataFrame contenant les données MVRV.
            
        Returns:
            True si le marché est sous-évalué, False sinon.
        """
        if mvrv_data.empty:
            return False
        
        # Récupérer la dernière valeur du ratio MVRV
        last_mvrv = mvrv_data['mvrv_ratio'].iloc[-1]
        
        return last_mvrv <= self.undervalued_threshold
    
    def is_overvalued(self, mvrv_data: pd.DataFrame) -> bool:
        """
        Vérifie si le marché est surévalué selon le ratio MVRV.
        
        Args:
            mvrv_data: DataFrame contenant les données MVRV.
            
        Returns:
            True si le marché est surévalué, False sinon.
        """
        if mvrv_data.empty:
            return False
        
        # Récupérer la dernière valeur du ratio MVRV
        last_mvrv = mvrv_data['mvrv_ratio'].iloc[-1]
        
        return last_mvrv >= self.overvalued_threshold
    
    def calculate_mvrv_z_score(self, mvrv_data: pd.DataFrame, window: int = 365) -> pd.DataFrame:
        """
        Calcule le Z-score du ratio MVRV.
        
        Le Z-score indique combien d'écarts-types une valeur est éloignée de la moyenne.
        Un Z-score élevé indique une surévaluation, un Z-score bas indique une sous-évaluation.
        
        Args:
            mvrv_data: DataFrame contenant les données MVRV.
            window: Fenêtre pour le calcul de la moyenne et de l'écart-type.
            
        Returns:
            DataFrame avec le Z-score ajouté.
        """
        if mvrv_data.empty:
            return pd.DataFrame()
        
        # Copier les données
        df = mvrv_data.copy()
        
        # Calculer la moyenne mobile et l'écart-type
        df['mvrv_mean'] = df['mvrv_ratio'].rolling(window=window).mean()
        df['mvrv_std'] = df['mvrv_ratio'].rolling(window=window).std()
        
        # Calculer le Z-score
        df['mvrv_z_score'] = (df['mvrv_ratio'] - df['mvrv_mean']) / df['mvrv_std']
        
        return df
    
    def analyze(self, 
               asset: str = "BTC", 
               days: int = 365,
               until: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Analyse complète du ratio MVRV.
        
        Args:
            asset: Actif à analyser (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse.
        """
        # Récupérer les données MVRV
        mvrv_data = self.get_mvrv_data(asset=asset, days=days, until=until)
        
        if mvrv_data.empty:
            logger.warning(f"Aucune donnée MVRV disponible pour {asset}")
            return {
                "signal": MVRVSignal.NEUTRAL,
                "is_undervalued": False,
                "is_overvalued": False,
                "mvrv_ratio": None,
                "mvrv_ema": None,
                "data": pd.DataFrame()
            }
        
        # Calculer le Z-score
        mvrv_data_with_z = self.calculate_mvrv_z_score(mvrv_data)
        
        # Générer le signal
        signal = self.get_signal(mvrv_data)
        
        # Vérifier si le marché est sous-évalué ou surévalué
        is_undervalued = self.is_undervalued(mvrv_data)
        is_overvalued = self.is_overvalued(mvrv_data)
        
        # Récupérer les dernières valeurs
        last_mvrv = mvrv_data['mvrv_ratio'].iloc[-1]
        last_mvrv_ema = mvrv_data['mvrv_ema'].iloc[-1]
        last_z_score = mvrv_data_with_z['mvrv_z_score'].iloc[-1] if 'mvrv_z_score' in mvrv_data_with_z.columns else None
        
        # Calculer les statistiques
        mvrv_min = mvrv_data['mvrv_ratio'].min()
        mvrv_max = mvrv_data['mvrv_ratio'].max()
        mvrv_mean = mvrv_data['mvrv_ratio'].mean()
        mvrv_median = mvrv_data['mvrv_ratio'].median()
        
        # Calculer la distance aux seuils
        distance_to_undervalued = (last_mvrv - self.undervalued_threshold) / self.undervalued_threshold * 100
        distance_to_overvalued = (last_mvrv - self.overvalued_threshold) / self.overvalued_threshold * 100
        
        return {
            "signal": signal,
            "is_undervalued": is_undervalued,
            "is_overvalued": is_overvalued,
            "mvrv_ratio": last_mvrv,
            "mvrv_ema": last_mvrv_ema,
            "mvrv_z_score": last_z_score,
            "mvrv_min": mvrv_min,
            "mvrv_max": mvrv_max,
            "mvrv_mean": mvrv_mean,
            "mvrv_median": mvrv_median,
            "distance_to_undervalued": distance_to_undervalued,
            "distance_to_overvalued": distance_to_overvalued,
            "data": mvrv_data_with_z
        }
    
    def plot_mvrv(self, 
                 mvrv_data: pd.DataFrame, 
                 title: str = "Ratio MVRV (Market Value to Realized Value)",
                 show_thresholds: bool = True,
                 show_z_score: bool = True) -> plt.Figure:
        """
        Crée un graphique du ratio MVRV.
        
        Args:
            mvrv_data: DataFrame contenant les données MVRV.
            title: Titre du graphique.
            show_thresholds: Si True, affiche les seuils de surévaluation/sous-évaluation.
            show_z_score: Si True, affiche le Z-score du MVRV.
            
        Returns:
            Figure matplotlib.
        """
        if mvrv_data.empty:
            logger.warning("Aucune donnée MVRV disponible pour créer un graphique")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "Aucune donnée disponible", ha='center', va='center')
            return fig
        
        # Créer la figure
        if show_z_score and 'mvrv_z_score' in mvrv_data.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Tracer le ratio MVRV
        ax1.plot(mvrv_data.index, mvrv_data['mvrv_ratio'], label='MVRV Ratio', color='blue')
        
        # Tracer l'EMA du ratio MVRV
        if 'mvrv_ema' in mvrv_data.columns:
            ax1.plot(mvrv_data.index, mvrv_data['mvrv_ema'], label=f'MVRV EMA ({self.ema_period})', color='red')
        
        # Ajouter les seuils
        if show_thresholds:
            ax1.axhline(y=self.strong_undervalued_threshold, color='green', linestyle='--', alpha=0.5, 
                       label=f'Fortement sous-évalué ({self.strong_undervalued_threshold})')
            ax1.axhline(y=self.undervalued_threshold, color='lightgreen', linestyle='--', alpha=0.5, 
                       label=f'Sous-évalué ({self.undervalued_threshold})')
            ax1.axhline(y=self.overvalued_threshold, color='orange', linestyle='--', alpha=0.5, 
                       label=f'Surévalué ({self.overvalued_threshold})')
            ax1.axhline(y=self.strong_overvalued_threshold, color='red', linestyle='--', alpha=0.5, 
                       label=f'Fortement surévalué ({self.strong_overvalued_threshold})')
            
            # Colorer les zones
            ax1.axhspan(0, self.strong_undervalued_threshold, alpha=0.1, color='green')
            ax1.axhspan(self.strong_undervalued_threshold, self.undervalued_threshold, alpha=0.05, color='green')
            ax1.axhspan(self.overvalued_threshold, self.strong_overvalued_threshold, alpha=0.05, color='red')
            ax1.axhspan(self.strong_overvalued_threshold, mvrv_data['mvrv_ratio'].max() * 1.1, alpha=0.1, color='red')
        
        # Configurer l'axe principal
        ax1.set_title(title)
        ax1.set_ylabel('MVRV Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Tracer le Z-score
        if show_z_score and 'mvrv_z_score' in mvrv_data.columns:
            ax2.plot(mvrv_data.index, mvrv_data['mvrv_z_score'], label='MVRV Z-Score', color='purple')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Z-Score = 2')
            ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.5, label='Z-Score = -2')
            
            # Colorer les zones
            ax2.axhspan(2, mvrv_data['mvrv_z_score'].max() * 1.1, alpha=0.1, color='red')
            ax2.axhspan(-2, mvrv_data['mvrv_z_score'].min() * 1.1, alpha=0.1, color='green')
            
            # Configurer l'axe du Z-score
            ax2.set_ylabel('Z-Score')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
            # Aligner les axes x
            ax2.set_xlabel('Date')
            ax1.sharex(ax2)
        else:
            ax1.set_xlabel('Date')
        
        plt.tight_layout()
        return fig
