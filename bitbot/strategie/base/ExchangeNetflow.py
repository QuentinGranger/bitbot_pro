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
                strong_inflow_threshold: float = 5000,  # BTC
                volatility_threshold: float = 0.02,  # 2% comme seuil de volatilité
                consider_orderbook: bool = False):  # Nouveau paramètre pour l'analyse des carnets d'ordres
        """
        Initialise l'indicateur de flux d'échange.
        
        Args:
            ema_period: Période pour le calcul de l'EMA du netflow.
            outflow_threshold: Seuil pour considérer un flux sortant significatif.
            strong_outflow_threshold: Seuil pour considérer un flux sortant fort.
            inflow_threshold: Seuil pour considérer un flux entrant significatif.
            strong_inflow_threshold: Seuil pour considérer un flux entrant fort.
            volatility_threshold: Seuil de volatilité pour ajuster la pondération des signaux.
            consider_orderbook: Si True, analyse les carnets d'ordres pour renforcer les signaux.
        """
        self.client = OnChainClient()
        self.ema_period = ema_period
        self.outflow_threshold = outflow_threshold
        self.strong_outflow_threshold = strong_outflow_threshold
        self.inflow_threshold = inflow_threshold
        self.strong_inflow_threshold = strong_inflow_threshold
        self.volatility_threshold = volatility_threshold
        self.volatility_weight = 1.0  # Poids initial (pleine confiance)
        self.consider_orderbook = consider_orderbook
        
        logger.info(f"Indicateur Exchange Netflow initialisé avec les paramètres: "
                   f"ema_period={ema_period}, "
                   f"outflow_threshold={outflow_threshold}, "
                   f"strong_outflow_threshold={strong_outflow_threshold}, "
                   f"inflow_threshold={inflow_threshold}, "
                   f"strong_inflow_threshold={strong_inflow_threshold}, "
                   f"volatility_threshold={volatility_threshold}, "
                   f"consider_orderbook={consider_orderbook}")
    
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
    
    def analyze(self, asset: str = "BTC", days: int = 30, check_orderbooks: bool = False) -> Dict[str, Any]:
        """
        Récupère et analyse les données de flux d'échange pour un actif.
        
        Args:
            asset: Symbole de l'actif (BTC, ETH, etc.).
            days: Nombre de jours de données à récupérer.
            check_orderbooks: Si True, analyse les carnets d'ordres pour renforcer les signaux.
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse.
        """
        # Récupérer les données de flux d'échange
        netflow_data = self.get_netflow_data(asset=asset, days=days)
        
        if netflow_data.empty:
            logger.warning(f"Aucune donnée de flux d'échange pour {asset}")
            return {}
        
        # Analyser les données de flux d'échange
        result = self.analyze_netflow(netflow_data)
        
        # Ajouter les données au résultat
        result['data'] = netflow_data
        
        # Analyser les carnets d'ordres si demandé
        if check_orderbooks or self.consider_orderbook:
            orderbook_analysis = self._analyze_orderbooks(asset)
            result.update(orderbook_analysis)
            
            # Renforcer ou atténuer le signal en fonction de l'analyse du carnet d'ordres
            if self._should_reinforce_signal(result['signal'], orderbook_analysis):
                result['signal_reinforced'] = True
                result['signal_note'] = "Signal renforcé par l'analyse du carnet d'ordres"
            elif self._should_attenuate_signal(result['signal'], orderbook_analysis):
                result['signal_attenuated'] = True
                result['signal_note'] = "Signal atténué par l'analyse du carnet d'ordres"
        
        return result
    
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
        
        # Ajuster la pondération en fonction de la volatilité
        self._adjust_volatility_weight(netflow_data)
        
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
            "is_strong_inflow": current_netflow > self.strong_inflow_threshold,
            "volatility_weight": self.volatility_weight
        }
    
    def _determine_signal(self, netflow: float) -> NetflowSignal:
        """
        Détermine le signal de flux d'échange basé sur la valeur de netflow.
        
        Args:
            netflow: Valeur du netflow.
            
        Returns:
            Signal de flux d'échange.
        """
        # Appliquer la pondération de volatilité aux seuils
        adjusted_outflow = self.outflow_threshold * self.volatility_weight
        adjusted_strong_outflow = self.strong_outflow_threshold * self.volatility_weight
        adjusted_inflow = self.inflow_threshold * self.volatility_weight
        adjusted_strong_inflow = self.strong_inflow_threshold * self.volatility_weight
        
        if netflow < adjusted_strong_outflow:
            return NetflowSignal.STRONG_OUTFLOW
        elif netflow < adjusted_outflow:
            return NetflowSignal.OUTFLOW
        elif netflow > adjusted_strong_inflow:
            return NetflowSignal.STRONG_INFLOW
        elif netflow > adjusted_inflow:
            return NetflowSignal.INFLOW
        else:
            return NetflowSignal.NEUTRAL
    
    def _adjust_volatility_weight(self, data: pd.DataFrame) -> None:
        """
        Ajuste la pondération des signaux en fonction de la volatilité du marché.
        
        En période de faible volatilité, les indicateurs on-chain sont moins réactifs à court terme,
        donc leur impact sur la décision de trading est réduit.
        
        Args:
            data: DataFrame contenant l'historique des prix.
        """
        if 'price' not in data.columns or len(data) < 14:
            # Pas assez de données pour calculer la volatilité
            self.volatility_weight = 1.0
            return
        
        # Calculer la volatilité sur les 14 derniers jours (ou moins si pas assez de données)
        window = min(14, len(data) - 1)
        recent_data = data.iloc[-window-1:].copy()
        
        # Calculer les rendements journaliers
        recent_data['returns'] = recent_data['price'].pct_change()
        
        # Écart-type des rendements (mesure de volatilité)
        volatility = recent_data['returns'].std()
        
        # Si la volatilité est inférieure au seuil, réduire le poids
        if volatility < self.volatility_threshold:
            # Formule: volatility_weight varie de 0.1 (volatilité minimale) à 1.0 (seuil atteint)
            self.volatility_weight = max(0.1, min(1.0, volatility / self.volatility_threshold))
        else:
            # Si la volatilité est supérieure au seuil, poids normal
            self.volatility_weight = 1.0
        
        logger.info(f"Volatilité: {volatility:.4f}, Poids des indicateurs on-chain: {self.volatility_weight:.2f}")
    
    def _analyze_orderbooks(self, asset: str) -> Dict[str, Any]:
        """
        Analyse les carnets d'ordres pour identifier des murs d'achat/vente.
        
        Args:
            asset: Symbole de l'actif.
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse des carnets d'ordres.
        """
        logger.info(f"Analyse des carnets d'ordres pour {asset}")
        
        # Dans une implémentation réelle, on utiliserait une API d'exchange comme Binance ou Coinbase
        # pour récupérer les données de carnet d'ordres en temps réel
        
        # Cette implémentation simule l'analyse des carnets d'ordres
        # En réalité, il faudrait intégrer des API d'exchanges et implémenter une logique d'analyse
        # plus sophistiquée pour identifier des murs d'achat/vente
        
        # Simuler des résultats d'analyse de carnet d'ordres pour démonstration
        # Dans une implémentation réelle, cette méthode analyserait les données réelles
        # des carnets d'ordres des principales exchanges où l'actif est tradé
        
        # Résultats simulés
        buy_walls_detected = False
        sell_walls_detected = False
        
        # Détection aléatoire pour simulation
        import random
        random.seed(datetime.now().timestamp())
        if random.random() < 0.3:  # 30% de chance de détecter un mur d'achat
            buy_walls_detected = True
        if random.random() < 0.3:  # 30% de chance de détecter un mur de vente
            sell_walls_detected = True
        
        # Générer une note d'analyse basée sur les résultats
        orderbook_note = ""
        if buy_walls_detected and sell_walls_detected:
            orderbook_note = "Murs d'achat et de vente détectés, indiquant une zone de consolidation potentielle."
        elif buy_walls_detected:
            orderbook_note = "Murs d'achat significatifs détectés, suggérant un support solide aux niveaux actuels."
        elif sell_walls_detected:
            orderbook_note = "Murs de vente significatifs détectés, suggérant une résistance aux niveaux actuels."
        else:
            orderbook_note = "Aucun mur d'achat ou de vente significatif détecté dans les carnets d'ordres."
        
        # Construction du résultat
        orderbook_result = {
            "orderbook_analyzed": True,
            "buy_walls_detected": buy_walls_detected,
            "sell_walls_detected": sell_walls_detected,
            "orderbook_note": orderbook_note
        }
        
        # Dans une implémentation réelle, on ajouterait plus de détails:
        # - niveaux de prix des murs
        # - volume cumulé aux niveaux de support/résistance
        # - distribution du volume dans le carnet d'ordres
        # - métrique d'imbalance entre acheteurs et vendeurs
        
        return orderbook_result
    
    def _should_reinforce_signal(self, signal: str, orderbook_analysis: Dict[str, Any]) -> bool:
        """
        Détermine si le signal doit être renforcé basé sur l'analyse du carnet d'ordres.
        
        Args:
            signal: Signal de flux d'échange.
            orderbook_analysis: Résultats de l'analyse du carnet d'ordres.
            
        Returns:
            True si le signal doit être renforcé, False sinon.
        """
        if not orderbook_analysis.get("orderbook_analyzed", False):
            return False
        
        # Si le signal est d'achat (sortie des exchanges) et qu'il y a des murs d'achat
        # C'est un renforcement du signal haussier
        if (signal == NetflowSignal.STRONG_OUTFLOW.value or signal == NetflowSignal.OUTFLOW.value) and \
           orderbook_analysis.get("buy_walls_detected", False):
            return True
        
        # Si le signal est de vente (entrée dans les exchanges) et qu'il y a des murs de vente
        # C'est un renforcement du signal baissier
        if (signal == NetflowSignal.STRONG_INFLOW.value or signal == NetflowSignal.INFLOW.value) and \
           orderbook_analysis.get("sell_walls_detected", False):
            return True
        
        return False
    
    def _should_attenuate_signal(self, signal: str, orderbook_analysis: Dict[str, Any]) -> bool:
        """
        Détermine si le signal doit être atténué basé sur l'analyse du carnet d'ordres.
        
        Args:
            signal: Signal de flux d'échange.
            orderbook_analysis: Résultats de l'analyse du carnet d'ordres.
            
        Returns:
            True si le signal doit être atténué, False sinon.
        """
        if not orderbook_analysis.get("orderbook_analyzed", False):
            return False
        
        # Si le signal est d'achat (sortie des exchanges) mais qu'il y a des murs de vente
        # Cela contredit le signal haussier
        if (signal == NetflowSignal.STRONG_OUTFLOW.value or signal == NetflowSignal.OUTFLOW.value) and \
           orderbook_analysis.get("sell_walls_detected", False):
            return True
        
        # Si le signal est de vente (entrée dans les exchanges) mais qu'il y a des murs d'achat
        # Cela contredit le signal baissier
        if (signal == NetflowSignal.STRONG_INFLOW.value or signal == NetflowSignal.INFLOW.value) and \
           orderbook_analysis.get("buy_walls_detected", False):
            return True
        
        return False
    
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
