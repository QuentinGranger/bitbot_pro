#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système d'agrégation des signaux avec scoring dynamique pour la plateforme BitBotPro.
Ce module permet d'agréger les signaux provenant de multiples sources d'indicateurs
et adapte dynamiquement les pondérations en fonction du contexte du marché.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import logging
from dataclasses import dataclass

# Import des stratégies et indicateurs
from bitbot.strategie.indicators.ema_crossover_strategy import EMACrossoverStrategy
from bitbot.strategie.indicators.sma_crossover_strategy import SMACrossoverStrategy
from bitbot.strategie.indicators.rsi_strategy import RSIStrategy
from bitbot.strategie.indicators.vwap import VWAPStrategy
from bitbot.strategie.base.MVRVRatio import MVRVRatio
from bitbot.strategie.base.ExchangeNetflow import ExchangeNetflow
from bitbot.utils.logger import logger

# Définition des constantes pour les scores
class SignalStrength(Enum):
    STRONG_SELL = 0
    SELL = 15
    WEAK_SELL = 30
    NEUTRAL = 50
    WEAK_BUY = 70
    BUY = 85
    STRONG_BUY = 100

# Définition des types de signaux
class SignalCategory(Enum):
    TECHNICAL = "technical"           # Indicateurs techniques
    ON_CHAIN = "on_chain"             # Données on-chain
    SENTIMENT = "sentiment"           # Sentiment du marché
    ORDER_BOOK = "order_book"         # Carnets d'ordres
    VOLATILITY = "volatility"         # Mesures de volatilité

@dataclass
class Signal:
    """Classe représentant un signal individuel avec son score et sa catégorie."""
    name: str
    score: float  # Score entre 0 et 100
    category: SignalCategory
    timestamp: float  # Timestamp du signal
    confidence: float = 1.0  # Niveau de confiance (0-1)
    metadata: Dict[str, Any] = None  # Métadonnées supplémentaires
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Assurer que le score est dans la plage 0-100
        self.score = max(0, min(100, self.score))
        # Assurer que la confiance est dans la plage 0-1
        self.confidence = max(0, min(1, self.confidence))

@dataclass
class AggregatedSignal:
    """Classe représentant un signal agrégé final."""
    score: float  # Score composite final (0-100)
    components: Dict[str, Signal]  # Signaux composants
    timestamp: float  # Timestamp de l'agrégation
    category_weights: Dict[SignalCategory, float]  # Poids appliqués par catégorie
    recommendation: str  # Recommandation (BUY, SELL, HOLD)
    confidence: float  # Confiance globale du signal
    
    def __str__(self):
        return (f"Signal[{self.recommendation}] Score: {self.score:.2f} "
                f"Confidence: {self.confidence:.2f} @ {pd.to_datetime(self.timestamp, unit='s')}")

class SignalAggregator:
    """
    Classe principale pour l'agrégation des signaux.
    
    Cette classe permet de:
    - Collecter des signaux de différentes sources
    - Ajuster dynamiquement les pondérations selon le contexte de marché
    - Agréger les signaux en un score composite
    - Gérer les fallbacks en cas de données manquantes
    """
    
    def __init__(self, 
                 default_weights: Optional[Dict[SignalCategory, float]] = None,
                 signal_threshold_buy: float = 70.0,
                 signal_threshold_sell: float = 30.0,
                 signal_timeout: int = 3600,  # 1 heure par défaut
                 confidence_threshold: float = 0.5):
        """
        Initialiser l'agrégateur de signaux.
        
        Args:
            default_weights: Pondérations par défaut pour chaque catégorie de signal
            signal_threshold_buy: Seuil pour générer un signal d'achat (0-100)
            signal_threshold_sell: Seuil pour générer un signal de vente (0-100)
            signal_timeout: Délai d'expiration des signaux en secondes
            confidence_threshold: Seuil de confiance minimum pour considérer un signal
        """
        # Configuration des pondérations par défaut si non spécifiées
        self.default_weights = default_weights or {
            SignalCategory.TECHNICAL: 0.4,    # 40% du poids pour les indicateurs techniques
            SignalCategory.ON_CHAIN: 0.2,     # 20% pour les données on-chain
            SignalCategory.SENTIMENT: 0.1,    # 10% pour le sentiment du marché
            SignalCategory.ORDER_BOOK: 0.2,   # 20% pour l'analyse du carnet d'ordres
            SignalCategory.VOLATILITY: 0.1,   # 10% pour les mesures de volatilité
        }
        
        # S'assurer que les poids par défaut somment à 1
        total_weight = sum(self.default_weights.values())
        if abs(total_weight - 1.0) > 0.001:  # Petite marge d'erreur pour les erreurs d'arrondi
            for category in self.default_weights:
                self.default_weights[category] /= total_weight
                
        # Configuration des seuils
        self.signal_threshold_buy = signal_threshold_buy
        self.signal_threshold_sell = signal_threshold_sell
        self.signal_timeout = signal_timeout
        self.confidence_threshold = confidence_threshold
        
        # Stockage des signaux actifs
        self.active_signals: Dict[str, Signal] = {}
        
        # Dernière analyse de contexte du marché
        self.last_market_context: Dict[str, Any] = {}
        
        # Derniers poids appliqués par catégorie
        self.current_weights = self.default_weights.copy()
        
        # Indicateurs disponibles
        self.available_indicators = {}
        
        # Derniers scores agrégés
        self.last_aggregated_signal: Optional[AggregatedSignal] = None
        
        # Historique des signaux
        self.signal_history = []
        self.max_history_size = 100  # Taille maximale de l'historique
        
        # Configuration des fallbacks
        self.fallbacks_enabled = True
        self.missing_data_timeout = 600  # 10 minutes avant de considérer une source comme indisponible
        self.category_availability = {cat: True for cat in SignalCategory}
        
        # Initialiser le système de journalisation
        self.logger = logger
        self.logger.info("Système d'agrégation des signaux initialisé")
        
    def register_indicator(self, name: str, indicator: Any, category: SignalCategory):
        """
        Enregistrer un indicateur auprès de l'agrégateur.
        
        Args:
            name: Nom unique de l'indicateur
            indicator: Instance de l'indicateur
            category: Catégorie du signal fourni par l'indicateur
        """
        self.available_indicators[name] = {
            'instance': indicator,
            'category': category,
            'last_update': 0,
            'status': 'registered'
        }
        self.logger.info(f"Indicateur enregistré: {name} (catégorie: {category.value})")
    
    def register_default_indicators(self):
        """Enregistrer les indicateurs par défaut inclus dans BitBotPro."""
        # Indicateurs techniques
        self.register_indicator("ema_crossover", EMACrossoverStrategy(), SignalCategory.TECHNICAL)
        self.register_indicator("sma_crossover", SMACrossoverStrategy(), SignalCategory.TECHNICAL)
        self.register_indicator("rsi", RSIStrategy(), SignalCategory.TECHNICAL)
        self.register_indicator("vwap", VWAPStrategy(), SignalCategory.TECHNICAL)
        
        # Indicateurs on-chain
        self.register_indicator("mvrv_ratio", MVRVRatio(), SignalCategory.ON_CHAIN)
        self.register_indicator("exchange_netflow", ExchangeNetflow(), SignalCategory.ON_CHAIN)
        
        self.logger.info("Indicateurs par défaut enregistrés")
    
    def set_weights(self, weights: Dict[SignalCategory, float]):
        """
        Définir manuellement les pondérations des catégories de signaux.
        
        Args:
            weights: Dictionnaire de pondérations par catégorie
        """
        # Vérifier la validité des pondérations
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            self.logger.warning(f"Les poids spécifiés ne somment pas à 1.0 (somme: {total_weight}). Normalisation appliquée.")
            normalized_weights = {cat: weight/total_weight for cat, weight in weights.items()}
            self.current_weights = normalized_weights
        else:
            self.current_weights = weights.copy()
            
        # Assurer que toutes les catégories sont présentes
        for category in SignalCategory:
            if category not in self.current_weights:
                self.current_weights[category] = 0.0
                
        self.logger.info(f"Nouveaux poids définis: {self.current_weights}")
        
    def configure(self, 
                 signal_threshold_buy: Optional[float] = None,
                 signal_threshold_sell: Optional[float] = None,
                 signal_timeout: Optional[int] = None,
                 confidence_threshold: Optional[float] = None,
                 fallbacks_enabled: Optional[bool] = None):
        """
        Configurer les paramètres de l'agrégateur.
        
        Args:
            signal_threshold_buy: Seuil pour générer un signal d'achat (0-100)
            signal_threshold_sell: Seuil pour générer un signal de vente (0-100)
            signal_timeout: Délai d'expiration des signaux en secondes
            confidence_threshold: Seuil de confiance minimum pour considérer un signal
            fallbacks_enabled: Activer/désactiver les fallbacks automatiques
        """
        if signal_threshold_buy is not None:
            self.signal_threshold_buy = max(0, min(100, signal_threshold_buy))
            
        if signal_threshold_sell is not None:
            self.signal_threshold_sell = max(0, min(100, signal_threshold_sell))
            
        if signal_timeout is not None:
            self.signal_timeout = max(60, signal_timeout)  # Minimum 1 minute
            
        if confidence_threshold is not None:
            self.confidence_threshold = max(0, min(1, confidence_threshold))
            
        if fallbacks_enabled is not None:
            self.fallbacks_enabled = fallbacks_enabled
            
        self.logger.info(f"Configuration mise à jour: seuils achat={self.signal_threshold_buy}, "
                        f"vente={self.signal_threshold_sell}, timeout={self.signal_timeout}, "
                        f"confiance={self.confidence_threshold}, fallbacks={self.fallbacks_enabled}")
    
    def add_signal(self, signal: Signal):
        """
        Ajouter un signal au pool de signaux actifs.
        
        Args:
            signal: Signal à ajouter
        """
        # Vérifier si le signal a une confiance suffisante
        if signal.confidence < self.confidence_threshold:
            self.logger.debug(f"Signal {signal.name} ignoré (confiance {signal.confidence} < seuil {self.confidence_threshold})")
            return
        
        # Ajouter au pool de signaux actifs
        self.active_signals[signal.name] = signal
        self.logger.debug(f"Signal ajouté: {signal.name} (score: {signal.score}, catégorie: {signal.category.value})")
        
    def collect_signals_from_indicators(self, data: pd.DataFrame, asset: str = "BTC") -> int:
        """
        Collecter les signaux de tous les indicateurs enregistrés.
        
        Args:
            data: DataFrame contenant les données de marché (OHLCV)
            asset: Asset à analyser
            
        Returns:
            Nombre de signaux collectés
        """
        now = time.time()
        signals_count = 0
        
        for name, indicator_info in self.available_indicators.items():
            try:
                indicator = indicator_info['instance']
                category = indicator_info['category']
                
                # Obtenir le signal de l'indicateur
                result = None
                
                # Appeler la méthode appropriée selon le type d'indicateur
                if hasattr(indicator, 'analyze'):
                    # Pour les indicateurs ayant une méthode analyze générique
                    result = indicator.analyze(data=data, asset=asset)
                elif hasattr(indicator, 'get_signal'):
                    # Pour les indicateurs avec une méthode get_signal
                    result = indicator.get_signal(data=data, asset=asset)
                else:
                    self.logger.warning(f"L'indicateur {name} n'a pas de méthode analyze ou get_signal")
                    continue
                
                if not result:
                    self.logger.warning(f"L'indicateur {name} n'a pas fourni de résultat")
                    continue
                
                # Convertir le résultat en score normalisé (0-100)
                score = self._normalize_indicator_result(result, name)
                
                # Créer et ajouter le signal
                signal = Signal(
                    name=name,
                    score=score,
                    category=category,
                    timestamp=now,
                    metadata=result if isinstance(result, dict) else {'raw_result': result}
                )
                
                self.add_signal(signal)
                signals_count += 1
                
                # Mettre à jour l'état de l'indicateur
                indicator_info['last_update'] = now
                indicator_info['status'] = 'active'
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la collecte du signal de {name}: {str(e)}")
                indicator_info['status'] = 'error'
        
        self.logger.info(f"{signals_count} signaux collectés des indicateurs")
        return signals_count
    
    def _normalize_indicator_result(self, result: Any, indicator_name: str) -> float:
        """
        Normaliser le résultat d'un indicateur en score 0-100.
        
        Args:
            result: Résultat brut de l'indicateur
            indicator_name: Nom de l'indicateur
            
        Returns:
            Score normalisé entre 0 et 100
        """
        # Si le résultat est déjà un dictionnaire avec un score
        if isinstance(result, dict) and 'score' in result:
            return max(0, min(100, result['score']))
        
        # Si le résultat est un dictionnaire avec un signal spécifique
        if isinstance(result, dict) and 'signal' in result:
            signal_str = result['signal']
            
            # Correspondance des signaux textuels aux scores
            signal_map = {
                'STRONG_BUY': 100,
                'BUY': 85,
                'WEAK_BUY': 70,
                'NEUTRAL': 50,
                'WEAK_SELL': 30,
                'SELL': 15,
                'STRONG_SELL': 0
            }
            
            # Tentative de correspondance exacte
            if signal_str.upper() in signal_map:
                return signal_map[signal_str.upper()]
            
            # Tentative de correspondance partielle
            for key, score in signal_map.items():
                if key in signal_str.upper():
                    return score
            
            # Cas particuliers pour certains indicateurs
            if 'BULLISH' in signal_str.upper():
                return 85
            elif 'BEARISH' in signal_str.upper():
                return 15
            
            # Valeur par défaut si pas de correspondance
            return 50
            
        # Pour les résultats numériques directs (supposés être dans la plage -1 à 1)
        if isinstance(result, (int, float)):
            # Conversion de [-1, 1] à [0, 100]
            return (result + 1) * 50
        
        # Pour les RSI et indicateurs similaires
        if indicator_name == 'rsi' and isinstance(result, dict) and 'value' in result:
            rsi_value = result['value']
            # RSI de 70+ indique survente (signal de vente)
            # RSI de 30- indique surachat (signal d'achat)
            if rsi_value >= 70:
                return 20  # Signal de vente (survente)
            elif rsi_value <= 30:
                return 80  # Signal d'achat (surachat)
            else:
                # Interpolation linéaire entre les valeurs
                if rsi_value > 50:  # Entre 50 et 70
                    return 50 - (rsi_value - 50) * (30 / 20)  # De 50 à 20
                else:  # Entre 30 et 50
                    return 80 - (rsi_value - 30) * (30 / 20)  # De 80 à 50
        
        # Pour les autres cas non gérés
        self.logger.warning(f"Normalisation non définie pour l'indicateur {indicator_name} avec résultat {result}")
        return 50  # Valeur neutre par défaut
    
    def clean_expired_signals(self):
        """Supprimer les signaux expirés du pool de signaux actifs."""
        now = time.time()
        expired_signals = []
        
        for name, signal in list(self.active_signals.items()):
            if now - signal.timestamp > self.signal_timeout:
                expired_signals.append(name)
                del self.active_signals[name]
        
        if expired_signals:
            self.logger.debug(f"Signaux expirés supprimés: {', '.join(expired_signals)}")
    
    def get_recommendation_from_score(self, score: float) -> str:
        """
        Obtenir une recommandation textuelle basée sur le score.
        
        Args:
            score: Score entre 0 et 100
            
        Returns:
            Recommandation sous forme de texte (BUY, SELL, HOLD)
        """
        if score >= self.signal_threshold_buy:
            return "BUY"
        elif score <= self.signal_threshold_sell:
            return "SELL"
        else:
            return "HOLD"
    
    def compute_aggregated_confidence(self, signals: Dict[str, Signal], weights: Dict[SignalCategory, float]) -> float:
        """
        Calculer la confiance globale basée sur les confiances individuelles pondérées.
        
        Args:
            signals: Dictionnaire des signaux actifs
            weights: Pondération par catégorie
            
        Returns:
            Niveau de confiance entre 0 et 1
        """
        if not signals:
            return 0.0
        
        # Calculer la somme des confiances par catégorie
        category_confidences = {}
        category_weights = {}
        
        # Initialiser les accumulateurs
        for category in SignalCategory:
            category_confidences[category] = 0.0
            category_weights[category] = 0.0
        
        # Accumuler les confiances par catégorie
        for signal in signals.values():
            category = signal.category
            category_confidences[category] += signal.confidence
            category_weights[category] += 1.0
        
        # Calculer la moyenne de confiance par catégorie
        weighted_confidences = 0.0
        total_weight = 0.0
        
        for category in SignalCategory:
            if category_weights[category] > 0:
                avg_category_confidence = category_confidences[category] / category_weights[category]
                category_weight = weights.get(category.value, 0.0)  # Utiliser .value pour obtenir la chaîne
                weighted_confidences += avg_category_confidence * category_weight
                total_weight += category_weight
        
        # Retourner la confiance globale pondérée
        if total_weight > 0:
            return weighted_confidences / total_weight
        return 0.0
    
    def aggregate_signals(self) -> Optional[AggregatedSignal]:
        """
        Agréger tous les signaux actifs en un signal composite.
        
        Returns:
            Signal agrégé ou None si aucun signal actif
        """
        # Nettoyer les signaux expirés
        self.clean_expired_signals()
        
        # Vérifier s'il y a des signaux actifs
        if not self.active_signals:
            self.logger.warning("Aucun signal actif à agréger")
            return None
        
        # Recalculer les poids des catégories si nécessaire (ajustement dynamique)
        self._adjust_weights_based_on_market_context()
        
        # Agréger les scores par catégorie
        category_scores = {}
        category_counts = {}
        
        # Initialiser les accumulateurs
        for category in SignalCategory:
            category_scores[category] = 0.0
            category_counts[category] = 0
        
        # Accumuler les scores par catégorie
        for signal in self.active_signals.values():
            category = signal.category
            category_scores[category] += signal.score
            category_counts[category] += 1
        
        # Calculer la moyenne de score par catégorie
        category_avg_scores = {}
        for category in SignalCategory:
            if category_counts[category] > 0:
                category_avg_scores[category] = category_scores[category] / category_counts[category]
            else:
                category_avg_scores[category] = 50.0  # Score neutre pour les catégories sans signal
        
        # Calculer le score composite pondéré
        composite_score = 0.0
        total_weight = 0.0
        
        for category, avg_score in category_avg_scores.items():
            weight = self.current_weights.get(category.value, 0.0)  # Utiliser .value pour obtenir la chaîne
            composite_score += avg_score * weight
            total_weight += weight
        
        # Normaliser le score composite
        if total_weight > 0:
            composite_score = composite_score / total_weight
        else:
            composite_score = 50.0  # Score neutre par défaut
        
        # Arrondir le score à deux décimales
        composite_score = round(composite_score, 2)
        
        # Déterminer la recommandation
        recommendation = self.get_recommendation_from_score(composite_score)
        
        # Calculer la confiance globale
        confidence = self.compute_aggregated_confidence(self.active_signals, self.current_weights)
        
        # Créer le signal agrégé
        aggregated_signal = AggregatedSignal(
            score=composite_score,
            components=self.active_signals.copy(),
            timestamp=time.time(),
            category_weights=self.current_weights.copy(),
            recommendation=recommendation,
            confidence=confidence
        )
        
        # Stocker le dernier signal agrégé
        self.last_aggregated_signal = aggregated_signal
        
        # Ajouter à l'historique
        self.signal_history.append(aggregated_signal)
        if len(self.signal_history) > self.max_history_size:
            self.signal_history.pop(0)
        
        self.logger.info(f"Signal agrégé: {aggregated_signal}")
        return aggregated_signal

    def _adjust_weights_based_on_market_context(self):
        """
        Ajuster dynamiquement les poids en fonction du contexte du marché.
        Cette méthode est appelée avant chaque agrégation de signaux.
        """
        # Cette méthode pourrait être étendue à l'avenir
        
        # Par exemple, donner plus de poids aux indicateurs techniques pendant des périodes de forte volatilité
        technical_signals = [signal for signal in self.active_signals.values() 
                         if signal.category == SignalCategory.TECHNICAL]
        
        # Détecter si nous sommes dans une période de crash
        sell_signals = sum(1 for s in technical_signals if s.score <= 30)
        extreme_signals = sum(1 for s in technical_signals if s.score <= 20 or s.score >= 80)
        
        # Si beaucoup de signaux de vente, nous pourrions être dans un crash
        if sell_signals >= 3 or extreme_signals >= 3:
            # Augmenter le poids des signaux sentiment et on-chain qui peuvent être plus fiables
            self.current_weights[SignalCategory.TECHNICAL.value] = 0.5  # Augmenter à 50%
            self.current_weights[SignalCategory.SENTIMENT.value] = 0.2  # Augmenter à 20%
            self.current_weights[SignalCategory.ON_CHAIN.value] = 0.2   # Augmenter à 20%
            self.current_weights[SignalCategory.ORDER_BOOK.value] = 0.05 # Réduire à 5%
            self.current_weights[SignalCategory.VOLATILITY.value] = 0.05 # Réduire à 5%
            
            self.logger.debug("Contexte de marché: forte volatilité détectée, poids ajustés")
        else:
            # Revenir aux poids par défaut en dehors des périodes de forte volatilité
            self.current_weights = self.default_weights.copy()
            
        # Ajuster pour les fallbacks si nécessaire
        if self.fallbacks_enabled:
            self._adjust_for_fallbacks(self.current_weights)
    
    def _adjust_for_fallbacks(self, weights: Dict[SignalCategory, float]):
        """
        Ajuster les poids pour compenser les sources de données manquantes.
        
        Args:
            weights: Dictionnaire des poids à ajuster (modifié in-place)
        """
        # Vérifier la disponibilité de chaque catégorie de signaux
        now = time.time()
        missing_categories = set()
        
        # Vérifier les signaux actifs par catégorie
        category_has_signals = {cat: False for cat in SignalCategory}
        for signal in self.active_signals.values():
            category_has_signals[signal.category] = True
        
        # Identifier les catégories sans signaux
        for category, has_signals in category_has_signals.items():
            if not has_signals:
                # Vérifier si la catégorie est considérée comme manquante
                weight = weights.get(category, 0.0)
                if weight > 0.0:
                    missing_categories.add(category)
                    self.logger.warning(f"Catégorie de signal manquante: {category.value}")
        
        # Si aucune catégorie ne manque, ne rien ajuster
        if not missing_categories:
            return
        
        # Recalculer les poids pour compenser les catégories manquantes
        total_missing_weight = sum(weights.get(cat, 0.0) for cat in missing_categories)
        remaining_weight = 1.0 - total_missing_weight
        
        # Mettre à zéro les poids des catégories manquantes
        for category in missing_categories:
            weights[category] = 0.0
        
        # Redistribuer le poids manquant aux catégories restantes
        if remaining_weight > 0:
            # Calculer le facteur de mise à l'échelle
            scale_factor = 1.0 / remaining_weight
            
            # Mettre à l'échelle les poids des catégories disponibles
            for category in weights:
                if category not in missing_categories:
                    weights[category] *= scale_factor
        
        self.logger.info(f"Ajustement des poids pour compenser les catégories manquantes: {[cat.value for cat in missing_categories]}")
    
    def analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyser le contexte du marché pour ajuster dynamiquement les poids.
        
        Args:
            data: DataFrame contenant les données de marché (OHLCV)
            
        Returns:
            Dictionnaire contenant le contexte du marché
        """
        # S'assurer que les données nécessaires sont présentes
        required_columns = ['close', 'high', 'low', 'volume']
        if not all(col in data.columns for col in required_columns):
            self.logger.warning("Données insuffisantes pour analyser le contexte du marché.")
            return {}
        
        # Calculer la volatilité (ATR normalisé sur les X dernières périodes)
        window_size = min(14, len(data))
        if window_size < 5:
            self.logger.warning("Pas assez de données pour calculer la volatilité.")
            return {}
        
        # Calculer le True Range
        data = data.copy()
        data['prev_close'] = data['close'].shift(1)
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['prev_close'])
        data['tr3'] = abs(data['low'] - data['prev_close'])
        data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculer l'ATR
        atr = data['true_range'].tail(window_size).mean()
        
        # Normaliser par le prix actuel
        current_price = data['close'].iloc[-1]
        normalized_atr = atr / current_price
        
        # Calculer le changement de volume
        avg_volume = data['volume'].tail(window_size).mean()
        current_volume = data['volume'].iloc[-1]
        volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
        
        # Détecter si le marché est latéral (sideways)
        price_range = data['high'].tail(window_size).max() - data['low'].tail(window_size).min()
        price_range_pct = price_range / current_price
        
        sideways = price_range_pct < 0.05  # Marché considéré comme latéral si la plage de prix est < 5%
        
        # Calculer la force de la tendance
        # Simple: utiliser la direction et l'ampleur de la variation de prix récente
        start_price = data['close'].iloc[-window_size]
        price_change = (current_price - start_price) / start_price
        trend_strength = abs(price_change)
        
        # Stocker et retourner le contexte du marché
        context = {
            'volatility': normalized_atr,
            'volume_change': volume_change,
            'sideways': sideways,
            'trend_strength': trend_strength,
            'price_change': price_change,
            'current_price': current_price,
            'timestamp': time.time()
        }
        
        self.last_market_context = context
        
        self.logger.info(f"Contexte du marché analysé: volatilité={normalized_atr:.4f}, "
                        f"changement de volume={volume_change:.2f}, "
                        f"marché latéral={sideways}, "
                        f"force de tendance={trend_strength:.2f}")
        
        return context
    
    def get_current_signal(self) -> Dict[str, Any]:
        """
        Obtenir le signal actuel et des informations sur l'agrégation.
        
        Returns:
            Dictionnaire contenant le signal actuel et les méta-informations
        """
        if not self.last_aggregated_signal:
            return {
                'status': 'no_signal',
                'timestamp': time.time(),
                'message': "Aucun signal agrégé disponible."
            }
        
        # Extraire les informations du dernier signal agrégé
        last_signal = self.last_aggregated_signal
        
        # Extraire les composants par catégorie
        components_by_category = {}
        for name, signal in last_signal.components.items():
            category = signal.category.value
            if category not in components_by_category:
                components_by_category[category] = []
            
            components_by_category[category].append({
                'name': name,
                'score': signal.score,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp
            })
        
        # Formater la réponse
        return {
            'status': 'ok',
            'timestamp': last_signal.timestamp,
            'recommendation': last_signal.recommendation,
            'score': last_signal.score,
            'confidence': last_signal.confidence,
            'category_weights': {cat.value: weight for cat, weight in last_signal.category_weights.items()},
            'components_by_category': components_by_category,
            'signals_count': len(last_signal.components)
        }
    
    def get_signal_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtenir l'historique des signaux agrégés.
        
        Args:
            limit: Nombre maximum de signaux à retourner
            
        Returns:
            Liste des signaux historiques, du plus récent au plus ancien
        """
        history = []
        
        for signal in self.signal_history[-limit:]:
            history.append({
                'timestamp': signal.timestamp,
                'recommendation': signal.recommendation,
                'score': signal.score,
                'confidence': signal.confidence
            })
        
        return list(reversed(history))  # Du plus récent au plus ancien
    
    def reset(self):
        """Réinitialiser l'état de l'agrégateur de signaux."""
        self.active_signals = {}
        self.last_market_context = {}
        self.current_weights = self.default_weights.copy()
        self.last_aggregated_signal = None
        self.logger.info("État de l'agrégateur de signaux réinitialisé")
