#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de backtest multi-scénarios pour le système d'agrégation de signaux.
Ce script évalue la fiabilité du système d'agrégation sur différentes conditions de marché:
- Forte volatilité
- Marchés en range/latéraux
- Flash crashes
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import math
import random

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitbot.strategie.aggregation.signal_aggregator import (
    SignalAggregator, Signal, SignalCategory, AggregatedSignal
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('signal_aggregator_backtest')

class MarketScenarioGenerator:
    """Classe pour générer différents scénarios de marché pour les tests."""
    
    @staticmethod
    def generate_normal_market(days=30, start_price=10000.0, volatility=0.02, 
                              trend=0.001, random_seed=42):
        """
        Générer des données de marché normales avec une légère tendance haussière.
        
        Args:
            days: Nombre de jours de données
            start_price: Prix initial
            volatility: Niveau de volatilité (écart-type des rendements)
            trend: Tendance journalière moyenne (positif=haussier, négatif=baissier)
            random_seed: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame avec les données OHLCV
        """
        np.random.seed(random_seed)
        
        # Date de début
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Générer un prix qui suit une marche aléatoire
        price = start_price
        closes = [price]
        
        for i in range(1, days):
            # Mouvement aléatoire avec tendance
            change = np.random.normal(trend, volatility)
            price *= (1 + change)
            closes.append(price)
        
        # Générer high, low, open à partir des close
        highs = [close * (1 + abs(np.random.normal(0, volatility/2))) for close in closes]
        lows = [close * (1 - abs(np.random.normal(0, volatility/2))) for close in closes]
        opens = [low + (high - low) * np.random.random() for high, low in zip(highs, lows)]
        
        # Générer le volume
        volumes = [np.random.gamma(2.0, 1000000) for _ in range(days)]
        
        # Créer le DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        data.set_index('date', inplace=True)
        return data
    
    @staticmethod
    def generate_high_volatility_market(days=30, start_price=10000.0, base_volatility=0.02,
                                      volatility_spike_factor=5.0, spike_days=5, random_seed=43):
        """
        Générer des données de marché avec une période de forte volatilité.
        
        Args:
            days: Nombre de jours de données
            start_price: Prix initial
            base_volatility: Niveau de volatilité de base
            volatility_spike_factor: Facteur multiplicateur pour les pics de volatilité
            spike_days: Nombre de jours avec volatilité élevée
            random_seed: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame avec les données OHLCV
        """
        np.random.seed(random_seed)
        
        # Date de début
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Générer un prix qui suit une marche aléatoire avec des périodes de forte volatilité
        price = start_price
        closes = [price]
        
        # Déterminer quand commencer la période de forte volatilité (au milieu par défaut)
        spike_start = days // 2
        
        for i in range(1, days):
            # Déterminer si nous sommes dans une période de forte volatilité
            if spike_start <= i < spike_start + spike_days:
                current_volatility = base_volatility * volatility_spike_factor
            else:
                current_volatility = base_volatility
                
            # Mouvement aléatoire
            change = np.random.normal(0, current_volatility)
            price *= (1 + change)
            closes.append(price)
        
        # Générer high, low, open à partir des close avec volatilité variable
        highs = []
        lows = []
        opens = []
        
        for i, close in enumerate(closes):
            # Déterminer si nous sommes dans une période de forte volatilité
            if spike_start <= i < spike_start + spike_days:
                current_volatility = base_volatility * volatility_spike_factor
            else:
                current_volatility = base_volatility
                
            high = close * (1 + abs(np.random.normal(0, current_volatility)))
            low = close * (1 - abs(np.random.normal(0, current_volatility)))
            open_price = low + (high - low) * np.random.random()
            
            highs.append(high)
            lows.append(low)
            opens.append(open_price)
        
        # Générer le volume avec des pics pendant la période de haute volatilité
        volumes = []
        for i in range(days):
            if spike_start <= i < spike_start + spike_days:
                # Volume plus élevé pendant la période de forte volatilité
                volumes.append(np.random.gamma(5.0, 1000000))
            else:
                volumes.append(np.random.gamma(2.0, 1000000))
        
        # Créer le DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        data.set_index('date', inplace=True)
        return data
    
    @staticmethod
    def generate_sideways_market(days=30, start_price=10000.0, range_pct=0.03, 
                               random_seed=44):
        """
        Générer des données de marché en range/latéral.
        
        Args:
            days: Nombre de jours de données
            start_price: Prix initial
            range_pct: Plage de variation en pourcentage du prix
            random_seed: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame avec les données OHLCV
        """
        np.random.seed(random_seed)
        
        # Date de début
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Prix central autour duquel le marché oscille
        center_price = start_price
        # Limites haute et basse du range
        upper_limit = center_price * (1 + range_pct/2)
        lower_limit = center_price * (1 - range_pct/2)
        
        # Générer un prix qui oscille dans un range
        closes = []
        current_price = center_price
        
        for i in range(days):
            # Mouvement aléatoire à l'intérieur du range
            # Plus le prix est proche d'une limite, plus il a tendance à revenir vers le centre
            distance_to_center = (current_price - center_price) / center_price
            # Force de rappel proportionnelle à la distance du centre
            mean_reversion = -distance_to_center * 0.5
            
            # Mouvement aléatoire avec force de rappel
            change = np.random.normal(mean_reversion, 0.005)
            current_price *= (1 + change)
            
            # S'assurer que le prix reste dans les limites
            current_price = max(lower_limit, min(upper_limit, current_price))
            closes.append(current_price)
        
        # Générer high, low, open avec une faible volatilité
        highs = [close * (1 + abs(np.random.normal(0, 0.005))) for close in closes]
        lows = [close * (1 - abs(np.random.normal(0, 0.005))) for close in closes]
        opens = [low + (high - low) * np.random.random() for high, low in zip(highs, lows)]
        
        # Générer le volume (généralement faible dans les marchés en range)
        volumes = [np.random.gamma(1.5, 800000) for _ in range(days)]
        
        # Créer le DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        data.set_index('date', inplace=True)
        return data
    
    @staticmethod
    def generate_flash_crash_market(days=30, start_price=10000.0, crash_day=15,
                                  crash_pct=-0.15, recovery_days=3, random_seed=45):
        """
        Générer des données de marché avec un flash crash suivi d'une récupération.
        
        Args:
            days: Nombre de jours de données
            start_price: Prix initial
            crash_day: Jour où le crash se produit
            crash_pct: Pourcentage de baisse durant le crash (négatif)
            recovery_days: Nombre de jours pour la récupération
            random_seed: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame avec les données OHLCV
        """
        np.random.seed(random_seed)
        
        # Date de début
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Générer un prix normal jusqu'au crash
        price = start_price
        closes = []
        
        for i in range(days):
            if i == crash_day:
                # Jour du crash
                price *= (1 + crash_pct)
            elif crash_day < i <= crash_day + recovery_days:
                # Période de récupération
                recovery_progress = (i - crash_day) / recovery_days
                # Récupération non linéaire (plus rapide au début)
                recovery_factor = np.sqrt(recovery_progress)
                # Récupérer une partie de la perte
                recovery_pct = -crash_pct * recovery_factor * 0.7  # 70% de récupération
                price *= (1 + recovery_pct / recovery_days)
            else:
                # Jours normaux
                change = np.random.normal(0.0005, 0.01)
                price *= (1 + change)
            
            closes.append(price)
        
        # Générer high, low, open
        highs = []
        lows = []
        opens = []
        
        for i, close in enumerate(closes):
            if i == crash_day:
                # Plus grande volatilité le jour du crash
                pre_crash_price = closes[i-1] if i > 0 else close
                high = pre_crash_price * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.02)))
                open_price = pre_crash_price * (1 - abs(np.random.normal(0, 0.01)))
            elif crash_day < i <= crash_day + recovery_days:
                # Volatilité élevée pendant la récupération
                high = close * (1 + abs(np.random.normal(0, 0.02)))
                low = close * (1 - abs(np.random.normal(0, 0.02)))
                open_price = closes[i-1] * (1 + np.random.normal(0, 0.02))
            else:
                # Volatilité normale
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = low + (high - low) * np.random.random()
            
            highs.append(high)
            lows.append(low)
            opens.append(open_price)
        
        # Générer le volume (pic de volume pendant le crash et la récupération)
        volumes = []
        for i in range(days):
            if i == crash_day:
                # Volume très élevé le jour du crash
                volumes.append(np.random.gamma(10.0, 1000000))
            elif crash_day < i <= crash_day + recovery_days:
                # Volume élevé pendant la récupération
                volumes.append(np.random.gamma(5.0, 1000000))
            else:
                # Volume normal
                volumes.append(np.random.gamma(2.0, 1000000))
        
        # Créer le DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        data.set_index('date', inplace=True)
        return data

class SignalSimulator:
    """Classe pour simuler des signaux à partir de différentes stratégies."""
    
    @staticmethod
    def simulate_ema_crossover_signals(data: pd.DataFrame, fast_period=9, slow_period=21) -> List[Dict[str, Any]]:
        """
        Simuler les signaux générés par une stratégie de croisement d'EMA.
        
        Args:
            data: DataFrame OHLCV
            fast_period: Période de l'EMA rapide
            slow_period: Période de l'EMA lente
            
        Returns:
            Liste de signaux simulés avec timestamps
        """
        # Calculer les EMAs
        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Identifier les croisements
        data = data.copy()
        data['ema_fast'] = ema_fast
        data['ema_slow'] = ema_slow
        data['ema_fast_gt_slow'] = ema_fast > ema_slow
        
        # Détecter les croisements (changements de True à False ou de False à True)
        # Corrigé pour éviter l'erreur avec l'opérateur ~ sur des flottants
        data['prev_ema_fast_gt_slow'] = data['ema_fast_gt_slow'].shift(1).fillna(False)
        data['crossover_buy'] = (data['ema_fast_gt_slow'] & ~data['prev_ema_fast_gt_slow']).astype(bool)
        data['crossover_sell'] = (~data['ema_fast_gt_slow'] & data['prev_ema_fast_gt_slow']).astype(bool)
        
        # Remplir les valeurs NaN
        data = data.fillna(False)
        
        # Générer les signaux
        signals = []
        for i, row in data.iterrows():
            timestamp = i.timestamp()
            
            if row['crossover_buy']:
                signals.append({
                    'name': 'ema_crossover',
                    'timestamp': timestamp,
                    'score': 75,  # Signal d'achat modéré à fort
                    'category': SignalCategory.TECHNICAL,
                    'confidence': 0.75,
                    'metadata': {
                        'type': 'buy',
                        'fast_ema': row['ema_fast'],
                        'slow_ema': row['ema_slow'],
                        'close': row['close']
                    }
                })
            elif row['crossover_sell']:
                signals.append({
                    'name': 'ema_crossover',
                    'timestamp': timestamp,
                    'score': 25,  # Signal de vente modéré
                    'category': SignalCategory.TECHNICAL,
                    'confidence': 0.75,
                    'metadata': {
                        'type': 'sell',
                        'fast_ema': row['ema_fast'],
                        'slow_ema': row['ema_slow'],
                        'close': row['close']
                    }
                })
                
        return signals
    
    @staticmethod
    def simulate_sma_crossover_signals(data: pd.DataFrame, fast_period=9, slow_period=21) -> List[Dict[str, Any]]:
        """
        Simuler les signaux générés par une stratégie de croisement de SMA.
        
        Args:
            data: DataFrame OHLCV
            fast_period: Période de la SMA rapide
            slow_period: Période de la SMA lente
            
        Returns:
            Liste de signaux simulés avec timestamps
        """
        # Calculer les SMAs
        sma_fast = data['close'].rolling(window=fast_period).mean()
        sma_slow = data['close'].rolling(window=slow_period).mean()
        
        # Identifier les croisements
        data = data.copy()
        data['sma_fast'] = sma_fast
        data['sma_slow'] = sma_slow
        data['sma_fast_gt_slow'] = sma_fast > sma_slow
        
        # Détecter les croisements (changements de True à False ou de False à True)
        data['prev_sma_fast_gt_slow'] = data['sma_fast_gt_slow'].shift(1).fillna(False)
        data['crossover_buy'] = (data['sma_fast_gt_slow'] & ~data['prev_sma_fast_gt_slow']).astype(bool)
        data['crossover_sell'] = (~data['sma_fast_gt_slow'] & data['prev_sma_fast_gt_slow']).astype(bool)
        
        # Remplir les valeurs NaN
        data = data.fillna(False)
        
        # Générer les signaux
        signals = []
        for i, row in data.iterrows():
            timestamp = i.timestamp()
            
            if row['crossover_buy']:
                signals.append({
                    'name': 'sma_crossover',
                    'timestamp': timestamp,
                    'score': 70,  # Signal d'achat modéré
                    'category': SignalCategory.TECHNICAL,
                    'confidence': 0.7,
                    'metadata': {
                        'type': 'buy',
                        'fast_sma': row['sma_fast'],
                        'slow_sma': row['sma_slow'],
                        'close': row['close']
                    }
                })
            elif row['crossover_sell']:
                signals.append({
                    'name': 'sma_crossover',
                    'timestamp': timestamp,
                    'score': 30,  # Signal de vente modéré
                    'category': SignalCategory.TECHNICAL,
                    'confidence': 0.7,
                    'metadata': {
                        'type': 'sell',
                        'fast_sma': row['sma_fast'],
                        'slow_sma': row['sma_slow'],
                        'close': row['close']
                    }
                })
                
        return signals
    
    @staticmethod
    def simulate_rsi_signals(data: pd.DataFrame, period=14, overbought=70, oversold=30) -> List[Dict[str, Any]]:
        """
        Simuler les signaux générés par une stratégie RSI.
        
        Args:
            data: DataFrame OHLCV
            period: Période du RSI
            overbought: Seuil de surachat
            oversold: Seuil de survente
            
        Returns:
            Liste de signaux simulés avec timestamps
        """
        # Calculer le RSI
        data = data.copy()
        delta = data['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Identifier les signaux d'achat et de vente
        data['buy_signal'] = data['rsi'] < oversold
        data['sell_signal'] = data['rsi'] > overbought
        
        # Générer les signaux
        signals = []
        for i, row in data.iterrows():
            if pd.notna(row['rsi']):  # S'assurer que le RSI est calculé
                timestamp = i.timestamp()
                
                if row['buy_signal']:
                    signals.append({
                        'name': 'rsi',
                        'timestamp': timestamp,
                        'score': 80,  # Signal d'achat fort (survente)
                        'category': SignalCategory.TECHNICAL,
                        'confidence': 0.65,
                        'metadata': {
                            'type': 'buy',
                            'rsi': row['rsi'],
                            'close': row['close'],
                            'threshold': oversold
                        }
                    })
                elif row['sell_signal']:
                    signals.append({
                        'name': 'rsi',
                        'timestamp': timestamp,
                        'score': 20,  # Signal de vente fort (surachat)
                        'category': SignalCategory.TECHNICAL,
                        'confidence': 0.65,
                        'metadata': {
                            'type': 'sell',
                            'rsi': row['rsi'],
                            'close': row['close'],
                            'threshold': overbought
                        }
                    })
                    
        return signals
    
    @staticmethod
    def simulate_order_book_signals(data: pd.DataFrame, volatility_factor=1.0) -> List[Dict[str, Any]]:
        """
        Simuler les signaux provenant du carnet d'ordres (orderbook).
        
        Args:
            data: DataFrame OHLCV
            volatility_factor: Facteur de volatilité pour ajuster la fréquence des signaux
            
        Returns:
            Liste de signaux simulés
        """
        # Simuler des signaux de carnet d'ordres basés sur la volatilité et le volume
        signals = []
        
        data = data.copy()
        # Calculer les variations de prix en pourcentage
        data['price_change'] = data['close'].pct_change()
        # Calculer les variations de volume en pourcentage
        data['volume_change'] = data['volume'].pct_change()
        
        # Remplir les valeurs NaN
        data = data.fillna(0)
        
        for i, row in data.iterrows():
            timestamp = i.timestamp()
            
            # Déséquilibre simulé du carnet d'ordres basé sur des variations de prix et de volume
            # Les grandes variations de prix/volume peuvent indiquer un déséquilibre dans le carnet
            price_change = row['price_change']
            volume_change = row['volume_change']
            
            # Simuler un déséquilibre du carnet d'ordres
            if abs(price_change) > 0.01 * volatility_factor and volume_change > 0.1 * volatility_factor:
                # Fort mouvement de prix avec augmentation du volume = pression acheteur/vendeur
                imbalance_score = 70 if price_change > 0 else 30
                confidence = min(0.9, 0.5 + abs(price_change) * 10 * volatility_factor)
                
                signals.append({
                    'name': 'orderbook_imbalance',
                    'timestamp': timestamp,
                    'score': imbalance_score,
                    'category': SignalCategory.ORDER_BOOK,
                    'confidence': confidence,
                    'metadata': {
                        'price_change': price_change,
                        'volume_change': volume_change,
                        'close': row['close']
                    }
                })
                
        return signals
    
    @staticmethod
    def simulate_sentiment_signals(data: pd.DataFrame, frequency=0.3, volatility_factor=1.0) -> List[Dict[str, Any]]:
        """
        Simuler les signaux de sentiment du marché.
        
        Args:
            data: DataFrame OHLCV
            frequency: Fréquence des signaux (0-1)
            volatility_factor: Facteur de volatilité pour ajuster la force des signaux
            
        Returns:
            Liste de signaux simulés
        """
        signals = []
        np.random.seed(42)  # Pour la reproductibilité
        
        for i, row in data.iterrows():
            timestamp = i.timestamp()
            
            # Générer aléatoirement un signal selon la fréquence spécifiée
            if np.random.random() < frequency:
                # Générer un score de sentiment aléatoire avec une tendance légèrement positive
                sentiment_score = np.random.normal(55, 15 * volatility_factor)
                # Limiter le score entre 0 et 100
                sentiment_score = max(0, min(100, sentiment_score))
                
                # La confiance varie avec la distance par rapport au score neutre
                neutrality = abs(sentiment_score - 50) / 50
                confidence = 0.5 + neutrality * 0.4
                
                signals.append({
                    'name': 'market_sentiment',
                    'timestamp': timestamp,
                    'score': sentiment_score,
                    'category': SignalCategory.SENTIMENT,
                    'confidence': confidence,
                    'metadata': {
                        'source': 'simulated',
                        'volatility_factor': volatility_factor,
                        'close': row['close']
                    }
                })
                
        return signals
    
    @staticmethod
    def simulate_volatility_signals(data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Simuler les signaux basés sur la volatilité.
        
        Args:
            data: DataFrame OHLCV
            
        Returns:
            Liste de signaux simulés
        """
        signals = []
        
        data = data.copy()
        # Calculer l'ATR
        data['high_low'] = data['high'] - data['low']
        data['high_close'] = abs(data['high'] - data['close'].shift(1))
        data['low_close'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # Calculer l'ATR en pourcentage du prix
        data['atr_pct'] = data['atr'] / data['close']
        
        # Générer des signaux basés sur l'ATR
        for i, row in data.iterrows():
            if pd.notna(row['atr_pct']):
                timestamp = i.timestamp()
                
                # Classifier la volatilité
                if row['atr_pct'] > 0.03:  # Volatilité élevée
                    signals.append({
                        'name': 'volatility_high',
                        'timestamp': timestamp,
                        'score': 40,  # Signal légèrement baissier (prudence)
                        'category': SignalCategory.VOLATILITY,
                        'confidence': 0.7,
                        'metadata': {
                            'atr': row['atr'],
                            'atr_pct': row['atr_pct'],
                            'close': row['close']
                        }
                    })
                elif row['atr_pct'] < 0.01:  # Volatilité faible
                    signals.append({
                        'name': 'volatility_low',
                        'timestamp': timestamp,
                        'score': 55,  # Signal légèrement haussier
                        'category': SignalCategory.VOLATILITY,
                        'confidence': 0.6,
                        'metadata': {
                            'atr': row['atr'],
                            'atr_pct': row['atr_pct'],
                            'close': row['close']
                        }
                    })
                    
        return signals
    
    @staticmethod
    def simulate_onchain_signals(data: pd.DataFrame, frequency=0.2) -> List[Dict[str, Any]]:
        """
        Simuler les signaux on-chain pour les crypto-monnaies.
        
        Args:
            data: DataFrame OHLCV
            frequency: Fréquence des signaux (0-1)
            
        Returns:
            Liste de signaux simulés
        """
        signals = []
        np.random.seed(43)  # Différent des autres pour varier les patterns
        
        for i, row in data.iterrows():
            timestamp = i.timestamp()
            
            # Générer aléatoirement un signal selon la fréquence spécifiée
            if np.random.random() < frequency:
                # Simuler différents types de signaux on-chain
                signal_type = np.random.choice(['whale_transaction', 'exchange_netflow', 'mvrv_ratio'])
                
                if signal_type == 'whale_transaction':
                    # Transactions de baleine - signaux plutôt haussiers
                    score = np.random.normal(65, 10)
                    confidence = 0.6
                elif signal_type == 'exchange_netflow':
                    # Flux net d'échange - varie entre haussier et baissier
                    score = np.random.normal(50, 15)
                    confidence = 0.7
                else:  # mvrv_ratio
                    # MVRV Ratio - signaux plutôt baissiers dans les extrêmes
                    score = np.random.normal(45, 20)
                    confidence = 0.65
                
                # Limiter le score entre 0 et 100
                score = max(0, min(100, score))
                
                signals.append({
                    'name': signal_type,
                    'timestamp': timestamp,
                    'score': score,
                    'category': SignalCategory.ON_CHAIN,
                    'confidence': confidence,
                    'metadata': {
                        'type': signal_type,
                        'close': row['close']
                    }
                })
                
        return signals

class BacktestRunner:
    """Classe pour exécuter des backtests sur l'agrégateur de signaux."""
    
    def __init__(self, market_generator=MarketScenarioGenerator(), signal_simulator=SignalSimulator()):
        """
        Initialiser le runner de backtest.
        
        Args:
            market_generator: Instance du générateur de scénarios de marché
            signal_simulator: Instance du simulateur de signaux
        """
        self.market_generator = market_generator
        self.signal_simulator = signal_simulator
        
    def run_uptrend_backtest(self, days=30, aggregator_config=None, plot=False):
        """
        Exécuter un backtest sur un marché en tendance haussière.
        
        Args:
            days: Nombre de jours pour la simulation
            aggregator_config: Configuration personnalisée pour l'agrégateur
            plot: Si True, affiche un graphique des résultats
            
        Returns:
            tuple: (données de marché, résultats agrégés, mesures de performance)
        """
        # Générer des données de marché
        market_data = self.market_generator.generate_normal_market(days=days, trend=0.002)
        
        # Générer des signaux à partir de différentes stratégies
        ema_signals = self.signal_simulator.simulate_ema_crossover_signals(market_data)
        sma_signals = self.signal_simulator.simulate_sma_crossover_signals(market_data)
        rsi_signals = self.signal_simulator.simulate_rsi_signals(market_data)
        orderbook_signals = self.signal_simulator.simulate_order_book_signals(market_data)
        sentiment_signals = self.signal_simulator.simulate_sentiment_signals(market_data)
        volatility_signals = self.signal_simulator.simulate_volatility_signals(market_data)
        onchain_signals = self.signal_simulator.simulate_onchain_signals(market_data)
        
        # Configuration pour marché haussier
        default_weights = {
            SignalCategory.TECHNICAL: 0.45,  # Augmenter l'importance des techniques en tendance
            SignalCategory.ON_CHAIN: 0.15,   
            SignalCategory.SENTIMENT: 0.15,  # Le sentiment est important dans les marchés haussiers
            SignalCategory.ORDER_BOOK: 0.15,
            SignalCategory.VOLATILITY: 0.10,
        }
        
        # Si une configuration personnalisée est fournie, l'utiliser
        if aggregator_config:
            if 'default_weights' in aggregator_config:
                default_weights = aggregator_config['default_weights']
            
            # Initialiser l'agrégateur avec les paramètres configurables
            aggregator = SignalAggregator(
                default_weights=default_weights,
                signal_threshold_buy=aggregator_config.get('signal_threshold_buy', 70.0),
                signal_threshold_sell=aggregator_config.get('signal_threshold_sell', 30.0),
                signal_timeout=aggregator_config.get('signal_timeout', 36000),  # Augmenté à 10 heures pour les backtests
                confidence_threshold=aggregator_config.get('confidence_threshold', 0.5)
            )
        else:
            # Utiliser la configuration par défaut
            aggregator = SignalAggregator(default_weights=default_weights)
        
        # Désactiver temporairement la méthode clean_expired_signals pour les backtests
        aggregator.clean_expired_signals = lambda: None
        
        # Combiner tous les signaux
        all_signals = ema_signals + sma_signals + rsi_signals + orderbook_signals + \
                      sentiment_signals + volatility_signals + onchain_signals
        
        # Trier les signaux par timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Agréger les signaux
        results = []
        positions = []
        current_position = None
        
        for i, row in market_data.iterrows():
            timestamp = i.timestamp()
            
            # Filtrer les signaux jusqu'à ce point dans le temps
            signals_to_now = [s for s in all_signals if s['timestamp'] <= timestamp]
            
            # Convertir les signaux au format attendu par l'agrégateur
            for signal in signals_to_now:
                # Convertir chaque signal en objet Signal
                category = SignalCategory.TECHNICAL  # Par défaut
                if 'category' in signal:
                    category_name = signal['category']
                    if category_name == 'technical':
                        category = SignalCategory.TECHNICAL
                    elif category_name == 'on_chain':
                        category = SignalCategory.ON_CHAIN
                    elif category_name == 'sentiment':
                        category = SignalCategory.SENTIMENT
                    elif category_name == 'order_book':
                        category = SignalCategory.ORDER_BOOK
                    elif category_name == 'volatility':
                        category = SignalCategory.VOLATILITY
                
                # Ajouter le signal à l'agrégateur
                signal_obj = Signal(
                    name=signal['name'],
                    score=signal['score'],
                    category=category,
                    timestamp=signal['timestamp'],
                    confidence=signal.get('confidence', 1.0)
                )
                aggregator.add_signal(signal_obj)
            
            # Agréger les signaux
            if signals_to_now:
                aggregated_signal = aggregator.aggregate_signals()
                
                if aggregated_signal:
                    # Enregistrer le résultat avec le timestamp et le prix
                    result = {
                        'timestamp': timestamp,
                        'date': i,
                        'close': row['close'],
                        'aggregated_score': aggregated_signal.score,
                        'signal_count': len(signals_to_now),
                        'confidence': aggregated_signal.confidence
                    }
                    results.append(result)
                    
                    # Simuler des décisions de trading basées sur le score agrégé
                    if not current_position and aggregated_signal.score > 65:
                        # Signal d'achat
                        current_position = {
                            'type': 'buy',
                            'entry_price': row['close'],
                            'entry_date': i,
                            'entry_timestamp': timestamp,
                            'entry_score': aggregated_signal.score
                        }
                        positions.append(current_position)
                    elif current_position and aggregated_signal.score < 35:
                        # Signal de vente, fermer la position
                        current_position['exit_price'] = row['close']
                        current_position['exit_date'] = i
                        current_position['exit_timestamp'] = timestamp
                        current_position['exit_score'] = aggregated_signal.score
                        current_position['profit_pct'] = \
                            (row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
                        current_position = None
        
        # Fermer une position ouverte à la fin du backtest
        if current_position:
            last_row = market_data.iloc[-1]
            current_position['exit_price'] = last_row['close']
            current_position['exit_date'] = market_data.index[-1]
            current_position['exit_timestamp'] = market_data.index[-1].timestamp()
            current_position['exit_score'] = results[-1]['aggregated_score'] if results else 50
            current_position['profit_pct'] = \
                (last_row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
        
        # Calculer les métriques de performance
        performance = self._calculate_performance(positions, market_data)
        
        # Afficher le graphique si demandé
        if plot:
            self._plot_results(market_data, results, positions)
            
        return market_data, results, performance
    
    def run_downtrend_backtest(self, days=30, aggregator_config=None, plot=False):
        """
        Exécuter un backtest sur un marché en tendance baissière.
        """
        # Similaire à run_uptrend_backtest mais avec un marché en baisse
        market_data = self.market_generator.generate_normal_market(days=days, trend=-0.002)
        
        # Le reste de la logique est similaire à run_uptrend_backtest...
        # Pour éviter la duplication, nous pourrions créer une méthode _run_generic_backtest
        
        # Générer des signaux à partir de différentes stratégies
        ema_signals = self.signal_simulator.simulate_ema_crossover_signals(market_data)
        sma_signals = self.signal_simulator.simulate_sma_crossover_signals(market_data)
        rsi_signals = self.signal_simulator.simulate_rsi_signals(market_data)
        orderbook_signals = self.signal_simulator.simulate_order_book_signals(market_data)
        sentiment_signals = self.signal_simulator.simulate_sentiment_signals(market_data)
        volatility_signals = self.signal_simulator.simulate_volatility_signals(market_data)
        onchain_signals = self.signal_simulator.simulate_onchain_signals(market_data)
        
        # Configuration pour marché baissier
        default_weights = {
            SignalCategory.TECHNICAL: 0.40,
            SignalCategory.ON_CHAIN: 0.20,   # Augmenter l'importance des données on-chain
            SignalCategory.SENTIMENT: 0.10,
            SignalCategory.ORDER_BOOK: 0.20, # Importance du carnet d'ordres pour les supports
            SignalCategory.VOLATILITY: 0.10,
        }
        
        # Si une configuration personnalisée est fournie, l'utiliser
        if aggregator_config:
            if 'default_weights' in aggregator_config:
                default_weights = aggregator_config['default_weights']
            
            # Initialiser l'agrégateur avec les paramètres configurables
            aggregator = SignalAggregator(
                default_weights=default_weights,
                signal_threshold_buy=aggregator_config.get('signal_threshold_buy', 70.0),
                signal_threshold_sell=aggregator_config.get('signal_threshold_sell', 30.0),
                signal_timeout=aggregator_config.get('signal_timeout', 36000),  # Augmenté à 10 heures pour les backtests
                confidence_threshold=aggregator_config.get('confidence_threshold', 0.5)
            )
        else:
            # Utiliser la configuration par défaut
            aggregator = SignalAggregator(default_weights=default_weights)
        
        # Désactiver temporairement la méthode clean_expired_signals pour les backtests
        aggregator.clean_expired_signals = lambda: None
        
        # Combiner tous les signaux
        all_signals = ema_signals + sma_signals + rsi_signals + orderbook_signals + \
                      sentiment_signals + volatility_signals + onchain_signals
        
        # Trier les signaux par timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Agréger les signaux
        results = []
        positions = []
        current_position = None
        
        for i, row in market_data.iterrows():
            timestamp = i.timestamp()
            
            # Filtrer les signaux jusqu'à ce point dans le temps
            signals_to_now = [s for s in all_signals if s['timestamp'] <= timestamp]
            
            # Convertir les signaux au format attendu par l'agrégateur
            for signal in signals_to_now:
                # Convertir chaque signal en objet Signal
                category = SignalCategory.TECHNICAL  # Par défaut
                if 'category' in signal:
                    category_name = signal['category']
                    if category_name == 'technical':
                        category = SignalCategory.TECHNICAL
                    elif category_name == 'on_chain':
                        category = SignalCategory.ON_CHAIN
                    elif category_name == 'sentiment':
                        category = SignalCategory.SENTIMENT
                    elif category_name == 'order_book':
                        category = SignalCategory.ORDER_BOOK
                    elif category_name == 'volatility':
                        category = SignalCategory.VOLATILITY
                
                # Ajouter le signal à l'agrégateur
                signal_obj = Signal(
                    name=signal['name'],
                    score=signal['score'],
                    category=category,
                    timestamp=signal['timestamp'],
                    confidence=signal.get('confidence', 1.0)
                )
                aggregator.add_signal(signal_obj)
            
            # Agréger les signaux
            if signals_to_now:
                aggregated_signal = aggregator.aggregate_signals()
                
                if aggregated_signal:
                    # Enregistrer le résultat avec le timestamp et le prix
                    result = {
                        'timestamp': timestamp,
                        'date': i,
                        'close': row['close'],
                        'aggregated_score': aggregated_signal.score,
                        'signal_count': len(signals_to_now),
                        'confidence': aggregated_signal.confidence
                    }
                    results.append(result)
                    
                    # Simuler des décisions de trading basées sur le score agrégé
                    if not current_position and aggregated_signal.score < 35:
                        # Signal de vente à découvert
                        current_position = {
                            'type': 'sell',
                            'entry_price': row['close'],
                            'entry_date': i,
                            'entry_timestamp': timestamp,
                            'entry_score': aggregated_signal.score
                        }
                        positions.append(current_position)
                    elif current_position and aggregated_signal.score > 65:
                        # Signal d'achat, fermer la position short
                        current_position['exit_price'] = row['close']
                        current_position['exit_date'] = i
                        current_position['exit_timestamp'] = timestamp
                        current_position['exit_score'] = aggregated_signal.score
                        current_position['profit_pct'] = \
                            (current_position['entry_price'] - row['close']) / current_position['entry_price'] * 100
                        current_position = None
        
        # Fermer une position ouverte à la fin du backtest
        if current_position:
            last_row = market_data.iloc[-1]
            current_position['exit_price'] = last_row['close']
            current_position['exit_date'] = market_data.index[-1]
            current_position['exit_timestamp'] = market_data.index[-1].timestamp()
            current_position['exit_score'] = results[-1]['aggregated_score'] if results else 50
            current_position['profit_pct'] = \
                (current_position['entry_price'] - last_row['close']) / current_position['entry_price'] * 100 \
                if current_position['type'] == 'sell' else \
                (last_row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
        
        # Calculer les métriques de performance
        performance = self._calculate_performance(positions, market_data)
        
        # Afficher le graphique si demandé
        if plot:
            self._plot_results(market_data, results, positions)
            
        return market_data, results, performance
    
    def run_sideways_backtest(self, days=30, aggregator_config=None, plot=False):
        """
        Exécuter un backtest sur un marché en range/latéral.
        
        Args:
            days: Nombre de jours pour la simulation
            aggregator_config: Configuration personnalisée pour l'agrégateur
            plot: Si True, affiche un graphique des résultats
            
        Returns:
            tuple: (données de marché, résultats agrégés, mesures de performance)
        """
        # Générer des données de marché en range
        market_data = self.market_generator.generate_sideways_market(days=days)
        
        # Générer des signaux
        ema_signals = self.signal_simulator.simulate_ema_crossover_signals(market_data)
        sma_signals = self.signal_simulator.simulate_sma_crossover_signals(market_data)
        rsi_signals = self.signal_simulator.simulate_rsi_signals(market_data)
        orderbook_signals = self.signal_simulator.simulate_order_book_signals(market_data)
        sentiment_signals = self.signal_simulator.simulate_sentiment_signals(market_data)
        volatility_signals = self.signal_simulator.simulate_volatility_signals(market_data)
        onchain_signals = self.signal_simulator.simulate_onchain_signals(market_data)
        
        # Configuration personnalisée pour l'agrégateur pour les marchés latéraux
        default_weights = {
            SignalCategory.TECHNICAL: 0.35,  # Réduire l'importance des indicateurs techniques
            SignalCategory.ON_CHAIN: 0.15,
            SignalCategory.SENTIMENT: 0.15,  # Augmenter l'importance du sentiment
            SignalCategory.ORDER_BOOK: 0.25, # Augmenter l'importance du carnet d'ordres
            SignalCategory.VOLATILITY: 0.1,
        }
        
        # Si une configuration personnalisée est fournie, l'utiliser
        if aggregator_config:
            if 'default_weights' in aggregator_config:
                default_weights = aggregator_config['default_weights']
            
            # Initialiser l'agrégateur avec les paramètres configurables
            aggregator = SignalAggregator(
                default_weights=default_weights,
                signal_threshold_buy=aggregator_config.get('signal_threshold_buy', 70.0),
                signal_threshold_sell=aggregator_config.get('signal_threshold_sell', 30.0),
                signal_timeout=aggregator_config.get('signal_timeout', 36000),  # Augmenté à 10 heures pour les backtests
                confidence_threshold=aggregator_config.get('confidence_threshold', 0.5)
            )
        else:
            # Utiliser la configuration par défaut pour les marchés latéraux
            aggregator = SignalAggregator(default_weights=default_weights)
        
        # Désactiver temporairement la méthode clean_expired_signals pour les backtests
        aggregator.clean_expired_signals = lambda: None
        
        # Ajouter une configuration spécifique des signaux pour les marchés en range
        special_weights = {
            'rsi': {'weight': 1.5},  # Augmenter l'importance du RSI dans les marchés en range
            'order_book_ask_bid_ratio': {'weight': 1.5},  # Augmenter l'importance du ratio ask/bid
            'volatility_low': {'weight': 1.2},  # Augmenter l'importance des signaux de faible volatilité
        }
        
        # Combiner tous les signaux
        all_signals = ema_signals + sma_signals + rsi_signals + orderbook_signals + \
                      sentiment_signals + volatility_signals + onchain_signals
        
        # Trier les signaux par timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Agréger les signaux
        results = []
        positions = []
        current_position = None
        
        for i, row in market_data.iterrows():
            timestamp = i.timestamp()
            
            # Filtrer les signaux jusqu'à ce point dans le temps
            signals_to_now = [s for s in all_signals if s['timestamp'] <= timestamp]
            
            # Convertir les signaux au format attendu par l'agrégateur
            for signal in signals_to_now:
                # Convertir chaque signal en objet Signal
                category = SignalCategory.TECHNICAL  # Par défaut
                if 'category' in signal:
                    category_name = signal['category']
                    if category_name == 'technical':
                        category = SignalCategory.TECHNICAL
                    elif category_name == 'on_chain':
                        category = SignalCategory.ON_CHAIN
                    elif category_name == 'sentiment':
                        category = SignalCategory.SENTIMENT
                    elif category_name == 'order_book':
                        category = SignalCategory.ORDER_BOOK
                    elif category_name == 'volatility':
                        category = SignalCategory.VOLATILITY
                
                # Ajouter le signal à l'agrégateur
                signal_obj = Signal(
                    name=signal['name'],
                    score=signal['score'],
                    category=category,
                    timestamp=signal['timestamp'],
                    confidence=signal.get('confidence', 1.0)
                )
                aggregator.add_signal(signal_obj)
            
            # Agréger les signaux
            if signals_to_now:
                aggregated_signal = aggregator.aggregate_signals()
                
                if aggregated_signal:
                    # Enregistrer le résultat avec le timestamp et le prix
                    result = {
                        'timestamp': timestamp,
                        'date': i,
                        'close': row['close'],
                        'aggregated_score': aggregated_signal.score,
                        'signal_count': len(signals_to_now),
                        'confidence': aggregated_signal.confidence
                    }
                    results.append(result)
                    
                    # Simuler des décisions de trading basées sur le score agrégé
                    # Dans un marché en range, nous utilisons des seuils plus stricts
                    if not current_position and aggregated_signal.score > 70:
                        # Signal d'achat fort
                        current_position = {
                            'type': 'buy',
                            'entry_price': row['close'],
                            'entry_date': i,
                            'entry_timestamp': timestamp,
                            'entry_score': aggregated_signal.score
                        }
                        positions.append(current_position)
                    elif current_position and aggregated_signal.score < 30:
                        # Signal de vente fort, fermer la position
                        current_position['exit_price'] = row['close']
                        current_position['exit_date'] = i
                        current_position['exit_timestamp'] = timestamp
                        current_position['exit_score'] = aggregated_signal.score
                        current_position['profit_pct'] = \
                            (row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
                        current_position = None
        
        # Fermer une position ouverte à la fin du backtest
        if current_position:
            last_row = market_data.iloc[-1]
            current_position['exit_price'] = last_row['close']
            current_position['exit_date'] = market_data.index[-1]
            current_position['exit_timestamp'] = market_data.index[-1].timestamp()
            current_position['exit_score'] = results[-1]['aggregated_score'] if results else 50
            current_position['profit_pct'] = \
                (last_row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
        
        # Calculer les métriques de performance
        performance = self._calculate_performance(positions, market_data)
        
        # Afficher le graphique si demandé
        if plot:
            self._plot_results(market_data, results, positions)
            
        return market_data, results, performance
    
    def run_flash_crash_backtest(self, days=30, aggregator_config=None, plot=False):
        """
        Exécuter un backtest sur un marché avec un flash crash.
        
        Args:
            days: Nombre de jours pour la simulation
            aggregator_config: Configuration personnalisée pour l'agrégateur
            plot: Si True, affiche un graphique des résultats
            
        Returns:
            tuple: (données de marché, résultats agrégés, mesures de performance)
        """
        # Générer des données de marché avec flash crash
        market_data = self.market_generator.generate_flash_crash_market(days=days)
        
        # Générer des signaux à partir de différentes stratégies
        ema_signals = self.signal_simulator.simulate_ema_crossover_signals(market_data)
        sma_signals = self.signal_simulator.simulate_sma_crossover_signals(market_data)
        rsi_signals = self.signal_simulator.simulate_rsi_signals(market_data)
        orderbook_signals = self.signal_simulator.simulate_order_book_signals(market_data)
        sentiment_signals = self.signal_simulator.simulate_sentiment_signals(market_data)
        volatility_signals = self.signal_simulator.simulate_volatility_signals(market_data)
        onchain_signals = self.signal_simulator.simulate_onchain_signals(market_data)
        
        # Configuration pour marché avec flash crash
        default_weights = {
            SignalCategory.TECHNICAL: 0.30,
            SignalCategory.ON_CHAIN: 0.15,
            SignalCategory.SENTIMENT: 0.15,
            SignalCategory.ORDER_BOOK: 0.25,  # Donner plus d'importance au carnet d'ordres
            SignalCategory.VOLATILITY: 0.15,  # Et à la volatilité
        }
        
        # Si une configuration personnalisée est fournie, l'utiliser
        if aggregator_config:
            if 'default_weights' in aggregator_config:
                default_weights = aggregator_config['default_weights']
            
            # Initialiser l'agrégateur avec les paramètres configurables
            aggregator = SignalAggregator(
                default_weights=default_weights,
                signal_threshold_buy=aggregator_config.get('signal_threshold_buy', 70.0),
                signal_threshold_sell=aggregator_config.get('signal_threshold_sell', 30.0),
                signal_timeout=aggregator_config.get('signal_timeout', 36000),  # Augmenté à 10 heures pour les backtests
                confidence_threshold=aggregator_config.get('confidence_threshold', 0.5)
            )
        else:
            # Utiliser la configuration par défaut pour les marchés avec flash crash
            aggregator = SignalAggregator(default_weights=default_weights)
        
        # Désactiver temporairement la méthode clean_expired_signals pour les backtests
        aggregator.clean_expired_signals = lambda: None
        
        # Combiner tous les signaux
        all_signals = ema_signals + sma_signals + rsi_signals + orderbook_signals + \
                      sentiment_signals + volatility_signals + onchain_signals
        
        # Trier les signaux par timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Agréger les signaux
        results = []
        positions = []
        current_position = None
        
        for i, row in market_data.iterrows():
            timestamp = i.timestamp()
            
            # Filtrer les signaux jusqu'à ce point dans le temps
            signals_to_now = [s for s in all_signals if s['timestamp'] <= timestamp]
            
            # Convertir les signaux au format attendu par l'agrégateur
            for signal in signals_to_now:
                # Convertir chaque signal en objet Signal
                category = SignalCategory.TECHNICAL  # Par défaut
                if 'category' in signal:
                    category_name = signal['category']
                    if category_name == 'technical':
                        category = SignalCategory.TECHNICAL
                    elif category_name == 'on_chain':
                        category = SignalCategory.ON_CHAIN
                    elif category_name == 'sentiment':
                        category = SignalCategory.SENTIMENT
                    elif category_name == 'order_book':
                        category = SignalCategory.ORDER_BOOK
                    elif category_name == 'volatility':
                        category = SignalCategory.VOLATILITY
                
                # Ajouter le signal à l'agrégateur
                signal_obj = Signal(
                    name=signal['name'],
                    score=signal['score'],
                    category=category,
                    timestamp=signal['timestamp'],
                    confidence=signal.get('confidence', 1.0)
                )
                aggregator.add_signal(signal_obj)
            
            # Agréger les signaux
            if signals_to_now:
                aggregated_signal = aggregator.aggregate_signals()
                
                if aggregated_signal:
                    # Enregistrer le résultat avec le timestamp et le prix
                    result = {
                        'timestamp': timestamp,
                        'date': i,
                        'close': row['close'],
                        'aggregated_score': aggregated_signal.score,
                        'signal_count': len(signals_to_now),
                        'confidence': aggregated_signal.confidence
                    }
                    results.append(result)
                    
                    # Simuler des décisions de trading basées sur le score agrégé
                    if not current_position and aggregated_signal.score > 75:
                        # Signal d'achat très fort (opportunité après le crash)
                        current_position = {
                            'type': 'buy',
                            'entry_price': row['close'],
                            'entry_date': i,
                            'entry_timestamp': timestamp,
                            'entry_score': aggregated_signal.score
                        }
                        positions.append(current_position)
                    elif current_position and aggregated_signal.score < 30:
                        # Signal de vente fort, fermer la position
                        current_position['exit_price'] = row['close']
                        current_position['exit_date'] = i
                        current_position['exit_timestamp'] = timestamp
                        current_position['exit_score'] = aggregated_signal.score
                        current_position['profit_pct'] = \
                            (row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
                        current_position = None
        
        # Fermer une position ouverte à la fin du backtest
        if current_position:
            last_row = market_data.iloc[-1]
            current_position['exit_price'] = last_row['close']
            current_position['exit_date'] = market_data.index[-1]
            current_position['exit_timestamp'] = market_data.index[-1].timestamp()
            current_position['exit_score'] = results[-1]['aggregated_score'] if results else 50
            current_position['profit_pct'] = \
                (last_row['close'] - current_position['entry_price']) / current_position['entry_price'] * 100
        
        # Calculer les métriques de performance
        performance = self._calculate_performance(positions, market_data)
        
        # Afficher le graphique si demandé
        if plot:
            self._plot_results(market_data, results, positions)
            
        return market_data, results, performance
    
    def _calculate_performance(self, positions, market_data):
        """
        Calculer les métriques de performance pour un ensemble de positions.
        
        Args:
            positions: Liste des positions ouvertes et fermées
            market_data: DataFrame des données de marché
            
        Returns:
            dict: Métriques de performance
        """
        if not positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0
            }
        
        # Calculer les métriques de base
        total_trades = len(positions)
        profitable_trades = sum(1 for p in positions if p.get('profit_pct', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        profits = [p.get('profit_pct', 0) for p in positions]
        avg_profit = sum(profits) / len(profits) if profits else 0
        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        # Calculer le profit factor
        winning_profits = sum(p for p in profits if p > 0)
        losing_profits = abs(sum(p for p in profits if p < 0))
        profit_factor = winning_profits / losing_profits if losing_profits != 0 else float('inf')
        
        # Calculer le Sharpe Ratio (approx.)
        if len(profits) > 1:
            returns_mean = np.mean(profits)
            returns_std = np.std(profits)
            sharpe_ratio = returns_mean / returns_std if returns_std != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculer le drawdown
        if market_data is not None:
            prices = market_data['close'].values
            peak = prices[0]
            drawdowns = []
            
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns) * 100
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # en pourcentage
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _plot_results(self, market_data, results, positions):
        """
        Tracer les résultats du backtest.
        
        Args:
            market_data: DataFrame des données de marché
            results: Liste des résultats d'agrégation
            positions: Liste des positions
        """
        try:
            import matplotlib.pyplot as plt
            
            # Créer un DataFrame pour les résultats
            if results:
                results_df = pd.DataFrame(results)
                results_df.set_index('date', inplace=True)
            
            # Configurer le graphique
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Tracer le prix
            ax1.plot(market_data.index, market_data['close'], label='Prix', color='blue')
            
            # Marquer les positions
            for pos in positions:
                if 'entry_date' in pos and 'exit_date' in pos:
                    # Tracer l'entrée
                    ax1.plot(pos['entry_date'], pos['entry_price'], 'o', 
                            color='green' if pos['type'] == 'buy' else 'red', 
                            markersize=8)
                    
                    # Tracer la sortie
                    ax1.plot(pos['exit_date'], pos['exit_price'], 's', 
                            color='red' if pos['type'] == 'buy' else 'green', 
                            markersize=8)
                    
                    # Tracer une ligne reliant l'entrée et la sortie
                    ax1.plot([pos['entry_date'], pos['exit_date']], 
                            [pos['entry_price'], pos['exit_price']], 
                            color='green' if pos.get('profit_pct', 0) > 0 else 'red',
                            linestyle='--', linewidth=1)
            
            # Tracer le score agrégé
            if results:
                ax2.plot(results_df.index, results_df['aggregated_score'], label='Score agrégé', color='purple')
                
                # Tracer les lignes de seuil
                ax2.axhline(y=65, color='green', linestyle='--', alpha=0.7)
                ax2.axhline(y=35, color='red', linestyle='--', alpha=0.7)
            
            # Configurer les axes et légendes
            ax1.set_title('Backtest de l\'agrégateur de signaux')
            ax1.set_ylabel('Prix')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Score agrégé')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib n'est pas installé. Impossible de tracer les résultats.")


def run_comprehensive_backtests(plot=True):
    """
    Exécuter un ensemble complet de backtests avec différents scénarios de marché.
    
    Args:
        plot: Si True, affiche des graphiques pour chaque scénario
        
    Returns:
        dict: Résultats détaillés pour chaque scénario
    """
    runner = BacktestRunner()
    results = {}
    
    print("Exécution du backtest sur marché haussier...")
    market_data, aggregation_results, performance = runner.run_uptrend_backtest(days=60, plot=plot)
    results['uptrend'] = {
        'market_data': market_data,
        'aggregation_results': aggregation_results,
        'performance': performance
    }
    print(f"Performance marché haussier: {performance}")
    
    print("\nExécution du backtest sur marché baissier...")
    market_data, aggregation_results, performance = runner.run_downtrend_backtest(days=60, plot=plot)
    results['downtrend'] = {
        'market_data': market_data,
        'aggregation_results': aggregation_results,
        'performance': performance
    }
    print(f"Performance marché baissier: {performance}")
    
    print("\nExécution du backtest sur marché latéral...")
    market_data, aggregation_results, performance = runner.run_sideways_backtest(days=60, plot=plot)
    results['sideways'] = {
        'market_data': market_data,
        'aggregation_results': aggregation_results,
        'performance': performance
    }
    print(f"Performance marché latéral: {performance}")
    
    print("\nExécution du backtest sur marché avec flash crash...")
    market_data, aggregation_results, performance = runner.run_flash_crash_backtest(days=60, plot=plot)
    results['flash_crash'] = {
        'market_data': market_data,
        'aggregation_results': aggregation_results,
        'performance': performance
    }
    print(f"Performance marché avec flash crash: {performance}")
    
    return results


if __name__ == "__main__":
    # Lancer les backtests complets
    run_comprehensive_backtests(plot=True)
    print("Fin du programme.")