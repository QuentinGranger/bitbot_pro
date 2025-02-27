"""
Module pour intégrer l'indicateur MVRV (Market Value to Realized Value) dans les stratégies de trading.

Ce module fournit des classes et fonctions pour utiliser le MVRV
dans les stratégies de trading, en générant des signaux d'achat et de vente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from enum import Enum
from datetime import datetime, timedelta

from bitbot.models.market_data import MarketData
from bitbot.strategie.base.MVRVRatio import MVRVIndicator, MVRVSignal
from bitbot.strategie.base.strategy_base import StrategyBase
from bitbot.models.trade_signal import TradeSignal, SignalType
from bitbot.utils.logger import logger

class MVRVStrategy(StrategyBase):
    """
    Stratégie de trading basée sur l'indicateur MVRV (Market Value to Realized Value).
    
    Cette stratégie utilise le ratio MVRV pour identifier les périodes où le Bitcoin
    est surévalué ou sous-évalué, et génère des signaux de trading en conséquence.
    """
    
    def __init__(self, 
                ema_period: int = 50,
                undervalued_threshold: float = 1.0,
                strong_undervalued_threshold: float = 0.75,
                overvalued_threshold: float = 2.5,
                strong_overvalued_threshold: float = 3.5,
                use_z_score: bool = True,
                z_score_threshold: float = 2.0):
        """
        Initialise la stratégie basée sur le MVRV.
        
        Args:
            ema_period: Période pour le calcul de l'EMA du ratio MVRV.
            undervalued_threshold: Seuil pour considérer le marché comme sous-évalué.
            strong_undervalued_threshold: Seuil pour considérer le marché comme fortement sous-évalué.
            overvalued_threshold: Seuil pour considérer le marché comme surévalué.
            strong_overvalued_threshold: Seuil pour considérer le marché comme fortement surévalué.
            use_z_score: Si True, utilise également le Z-score pour générer des signaux.
            z_score_threshold: Seuil pour le Z-score (valeur absolue).
        """
        super().__init__()
        
        self.name = "MVRVStrategy"
        self.description = "Stratégie basée sur l'indicateur MVRV (Market Value to Realized Value)"
        
        self.ema_period = ema_period
        self.undervalued_threshold = undervalued_threshold
        self.strong_undervalued_threshold = strong_undervalued_threshold
        self.overvalued_threshold = overvalued_threshold
        self.strong_overvalued_threshold = strong_overvalued_threshold
        self.use_z_score = use_z_score
        self.z_score_threshold = z_score_threshold
        
        # Initialiser l'indicateur MVRV
        self.mvrv_indicator = MVRVIndicator(
            ema_period=ema_period,
            undervalued_threshold=undervalued_threshold,
            strong_undervalued_threshold=strong_undervalued_threshold,
            overvalued_threshold=overvalued_threshold,
            strong_overvalued_threshold=strong_overvalued_threshold
        )
        
        logger.info(f"Stratégie {self.name} initialisée avec les paramètres: "
                   f"ema_period={ema_period}, "
                   f"undervalued_threshold={undervalued_threshold}, "
                   f"strong_undervalued_threshold={strong_undervalued_threshold}, "
                   f"overvalued_threshold={overvalued_threshold}, "
                   f"strong_overvalued_threshold={strong_overvalued_threshold}, "
                   f"use_z_score={use_z_score}, "
                   f"z_score_threshold={z_score_threshold}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Définit les paramètres de la stratégie.
        
        Args:
            **kwargs: Paramètres à définir.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Mettre à jour l'indicateur
        self.mvrv_indicator.set_parameters(
            ema_period=self.ema_period,
            undervalued_threshold=self.undervalued_threshold,
            strong_undervalued_threshold=self.strong_undervalued_threshold,
            overvalued_threshold=self.overvalued_threshold,
            strong_overvalued_threshold=self.strong_overvalued_threshold
        )
        
        logger.info(f"Paramètres de la stratégie {self.name} mis à jour: {kwargs}")
    
    def generate_signals(self, 
                        asset: str = "BTC", 
                        days: int = 365,
                        until: Optional[Union[str, datetime]] = None) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur l'indicateur MVRV.
        
        Args:
            asset: Actif à analyser (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            Liste de signaux de trading.
        """
        signals = []
        
        # Analyser les données avec le MVRV
        analysis = self.mvrv_indicator.analyze(asset=asset, days=days, until=until)
        
        if analysis['mvrv_ratio'] is None:
            logger.warning(f"Aucune donnée MVRV disponible pour {asset}")
            return signals
        
        # Extraire les données
        signal = analysis['signal']
        is_undervalued = analysis['is_undervalued']
        is_overvalued = analysis['is_overvalued']
        mvrv_ratio = analysis['mvrv_ratio']
        mvrv_z_score = analysis['mvrv_z_score']
        
        # Obtenir la dernière date/heure
        if 'data' in analysis and not analysis['data'].empty:
            timestamp = analysis['data'].index[-1]
        else:
            timestamp = datetime.now()
        
        # Prix fictif pour le signal (non utilisé dans cette stratégie)
        price = 0.0
        
        # Générer des signaux basés sur les conditions
        
        # 1. Signal d'achat fort: marché fortement sous-évalué
        if signal == MVRVSignal.STRONG_UNDERVALUED or (self.use_z_score and mvrv_z_score is not None and mvrv_z_score < -self.z_score_threshold):
            signals.append(TradeSignal(
                symbol=asset,
                timeframe="1d",  # Le MVRV est généralement analysé sur des données journalières
                timestamp=timestamp,
                signal_type=SignalType.STRONG_BUY,
                price=price,
                confidence=0.9,
                source=self.name,
                metadata={
                    "mvrv_ratio": mvrv_ratio,
                    "mvrv_z_score": mvrv_z_score,
                    "is_undervalued": is_undervalued,
                    "signal": signal.value
                }
            ))
        
        # 2. Signal d'achat: marché sous-évalué
        elif signal == MVRVSignal.UNDERVALUED or (self.use_z_score and mvrv_z_score is not None and mvrv_z_score < -1.0):
            signals.append(TradeSignal(
                symbol=asset,
                timeframe="1d",
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=price,
                confidence=0.7,
                source=self.name,
                metadata={
                    "mvrv_ratio": mvrv_ratio,
                    "mvrv_z_score": mvrv_z_score,
                    "is_undervalued": is_undervalued,
                    "signal": signal.value
                }
            ))
        
        # 3. Signal de vente fort: marché fortement surévalué
        elif signal == MVRVSignal.STRONG_OVERVALUED or (self.use_z_score and mvrv_z_score is not None and mvrv_z_score > self.z_score_threshold):
            signals.append(TradeSignal(
                symbol=asset,
                timeframe="1d",
                timestamp=timestamp,
                signal_type=SignalType.STRONG_SELL,
                price=price,
                confidence=0.9,
                source=self.name,
                metadata={
                    "mvrv_ratio": mvrv_ratio,
                    "mvrv_z_score": mvrv_z_score,
                    "is_overvalued": is_overvalued,
                    "signal": signal.value
                }
            ))
        
        # 4. Signal de vente: marché surévalué
        elif signal == MVRVSignal.OVERVALUED or (self.use_z_score and mvrv_z_score is not None and mvrv_z_score > 1.0):
            signals.append(TradeSignal(
                symbol=asset,
                timeframe="1d",
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=price,
                confidence=0.7,
                source=self.name,
                metadata={
                    "mvrv_ratio": mvrv_ratio,
                    "mvrv_z_score": mvrv_z_score,
                    "is_overvalued": is_overvalued,
                    "signal": signal.value
                }
            ))
        
        return signals
    
    def get_market_cycle_position(self, 
                                asset: str = "BTC", 
                                days: int = 365,
                                until: Optional[Union[str, datetime]] = None) -> Dict:
        """
        Détermine la position dans le cycle de marché basée sur le MVRV.
        
        Args:
            asset: Actif à analyser (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            Dictionnaire contenant la position dans le cycle de marché.
        """
        # Analyser les données avec le MVRV
        analysis = self.mvrv_indicator.analyze(asset=asset, days=days, until=until)
        
        if analysis['mvrv_ratio'] is None:
            logger.warning(f"Aucune donnée MVRV disponible pour {asset}")
            return {
                "cycle_position": "Indéterminé",
                "confidence": 0.0,
                "details": "Données insuffisantes"
            }
        
        # Extraire les données
        mvrv_ratio = analysis['mvrv_ratio']
        mvrv_z_score = analysis['mvrv_z_score']
        signal = analysis['signal']
        
        # Déterminer la position dans le cycle
        if signal == MVRVSignal.STRONG_UNDERVALUED:
            cycle_position = "Fond de marché"
            confidence = 0.9
            details = "Le marché est fortement sous-évalué, ce qui indique un fond de marché potentiel."
        elif signal == MVRVSignal.UNDERVALUED:
            cycle_position = "Accumulation"
            confidence = 0.7
            details = "Le marché est sous-évalué, ce qui correspond à une phase d'accumulation."
        elif signal == MVRVSignal.NEUTRAL:
            if mvrv_ratio < 1.5:
                cycle_position = "Début de tendance haussière"
                confidence = 0.6
                details = "Le marché est légèrement sous-évalué, ce qui peut indiquer le début d'une tendance haussière."
            else:
                cycle_position = "Milieu de cycle"
                confidence = 0.5
                details = "Le marché est dans une zone neutre, ce qui correspond au milieu d'un cycle."
        elif signal == MVRVSignal.OVERVALUED:
            cycle_position = "Distribution"
            confidence = 0.7
            details = "Le marché est surévalué, ce qui correspond à une phase de distribution."
        elif signal == MVRVSignal.STRONG_OVERVALUED:
            cycle_position = "Sommet de marché"
            confidence = 0.9
            details = "Le marché est fortement surévalué, ce qui indique un sommet de marché potentiel."
        else:
            cycle_position = "Indéterminé"
            confidence = 0.0
            details = "Impossible de déterminer la position dans le cycle."
        
        # Ajouter des détails basés sur le Z-score
        if mvrv_z_score is not None:
            if mvrv_z_score > 2.0:
                details += " Le Z-score élevé confirme une surévaluation significative."
            elif mvrv_z_score < -2.0:
                details += " Le Z-score bas confirme une sous-évaluation significative."
        
        return {
            "cycle_position": cycle_position,
            "confidence": confidence,
            "details": details,
            "mvrv_ratio": mvrv_ratio,
            "mvrv_z_score": mvrv_z_score,
            "signal": signal.value
        }
    
    def get_investment_recommendation(self, 
                                    asset: str = "BTC", 
                                    days: int = 365,
                                    until: Optional[Union[str, datetime]] = None) -> Dict:
        """
        Fournit des recommandations d'investissement basées sur le MVRV.
        
        Args:
            asset: Actif à analyser (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            until: Date de fin (format ISO ou objet datetime).
            
        Returns:
            Dictionnaire contenant les recommandations d'investissement.
        """
        # Analyser les données avec le MVRV
        analysis = self.mvrv_indicator.analyze(asset=asset, days=days, until=until)
        
        if analysis['mvrv_ratio'] is None:
            logger.warning(f"Aucune donnée MVRV disponible pour {asset}")
            return {
                "recommendation": "Attendre",
                "confidence": 0.0,
                "details": "Données insuffisantes pour formuler une recommandation."
            }
        
        # Extraire les données
        mvrv_ratio = analysis['mvrv_ratio']
        mvrv_z_score = analysis['mvrv_z_score']
        signal = analysis['signal']
        
        # Déterminer la recommandation
        if signal == MVRVSignal.STRONG_UNDERVALUED:
            recommendation = "Acheter agressivement"
            confidence = 0.9
            details = "Le marché est fortement sous-évalué. C'est une excellente opportunité d'achat à long terme."
            allocation = 0.8  # Pourcentage du capital à allouer
        elif signal == MVRVSignal.UNDERVALUED:
            recommendation = "Acheter"
            confidence = 0.7
            details = "Le marché est sous-évalué. C'est une bonne opportunité d'achat."
            allocation = 0.5
        elif signal == MVRVSignal.NEUTRAL:
            if mvrv_ratio < 1.5:
                recommendation = "Acheter progressivement"
                confidence = 0.5
                details = "Le marché est légèrement sous-évalué. Envisagez d'acheter progressivement."
                allocation = 0.3
            else:
                recommendation = "Conserver"
                confidence = 0.5
                details = "Le marché est dans une zone neutre. Conservez vos positions existantes."
                allocation = 0.0
        elif signal == MVRVSignal.OVERVALUED:
            recommendation = "Vendre partiellement"
            confidence = 0.7
            details = "Le marché est surévalué. Envisagez de prendre des bénéfices partiels."
            allocation = -0.5  # Négatif indique une réduction de position
        elif signal == MVRVSignal.STRONG_OVERVALUED:
            recommendation = "Vendre"
            confidence = 0.9
            details = "Le marché est fortement surévalué. C'est une bonne opportunité de vente."
            allocation = -0.8
        else:
            recommendation = "Attendre"
            confidence = 0.0
            details = "Impossible de formuler une recommandation."
            allocation = 0.0
        
        # Ajouter des détails basés sur le Z-score
        if mvrv_z_score is not None:
            if mvrv_z_score > 2.0:
                details += " Le Z-score élevé confirme une surévaluation significative."
            elif mvrv_z_score < -2.0:
                details += " Le Z-score bas confirme une sous-évaluation significative."
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "details": details,
            "allocation": allocation,
            "mvrv_ratio": mvrv_ratio,
            "mvrv_z_score": mvrv_z_score,
            "signal": signal.value
        }
