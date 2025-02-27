"""
Module de simulation de conditions de marché extrêmes pour tester la robustesse du système.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
import random
from enum import Enum

from bitbot.utils.logger import logger
from bitbot.backtest.engine import BacktestConfig, BacktestEngine
from bitbot.models.signal import SignalType
from bitbot.trading.strategy import BaseStrategy

class MarketCondition(Enum):
    """Types de conditions de marché extrêmes."""
    NORMAL = "normal"
    FLASH_CRASH = "flash_crash"
    PUMP_DUMP = "pump_dump"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"

class NetworkCondition(Enum):
    """Types de conditions réseau."""
    NORMAL = "normal"
    HIGH_LATENCY = "high_latency"
    INTERMITTENT = "intermittent"
    OFFLINE = "offline"

@dataclass
class StressTestConfig:
    """Configuration des tests de stress."""
    # Conditions de marché
    volatility_factor: float = 2.0  # Multiplicateur de volatilité
    flash_crash_probability: float = 0.001  # Probabilité de flash crash
    flash_crash_magnitude: float = 0.15  # Amplitude du crash (15%)
    pump_dump_probability: float = 0.001  # Probabilité de pump & dump
    pump_dump_magnitude: float = 0.20  # Amplitude du pump (20%)
    
    # Conditions réseau
    base_latency: int = 100  # Latence de base en ms
    max_latency: int = 2000  # Latence maximale en ms
    latency_volatility: float = 0.5  # Volatilité de la latence
    connection_loss_probability: float = 0.01  # Probabilité de perte de connexion
    max_offline_duration: int = 300  # Durée maximale hors ligne en secondes

class MarketStressSimulator:
    """Simulateur de conditions de marché extrêmes."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.current_condition = MarketCondition.NORMAL
        self.condition_start_time = None
        self.condition_duration = 0
    
    def apply_stress(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applique des conditions de stress aux données.
        
        Args:
            data: Données OHLCV originales
        
        Returns:
            Données modifiées avec stress
        """
        df = data.copy()
        
        for i in range(len(df)):
            # Déterminer la condition de marché
            if random.random() < self.config.flash_crash_probability:
                self.current_condition = MarketCondition.FLASH_CRASH
                self.condition_duration = random.randint(1, 5)  # 1-5 bougies
                self.condition_start_time = i
            elif random.random() < self.config.pump_dump_probability:
                self.current_condition = MarketCondition.PUMP_DUMP
                self.condition_duration = random.randint(3, 8)  # 3-8 bougies
                self.condition_start_time = i
            elif i - (self.condition_start_time or 0) >= (self.condition_duration or 0):
                self.current_condition = MarketCondition.NORMAL
            
            # Appliquer la condition
            if self.current_condition == MarketCondition.FLASH_CRASH:
                crash_progress = (i - self.condition_start_time) / self.condition_duration
                magnitude = self.config.flash_crash_magnitude * (1 - crash_progress)
                df.iloc[i] = self._apply_price_shock(df.iloc[i], -magnitude)
                
            elif self.current_condition == MarketCondition.PUMP_DUMP:
                pump_progress = (i - self.condition_start_time) / self.condition_duration
                if pump_progress < 0.5:  # Phase de pump
                    magnitude = self.config.pump_dump_magnitude * (pump_progress * 2)
                else:  # Phase de dump
                    magnitude = self.config.pump_dump_magnitude * ((1 - pump_progress) * 2)
                df.iloc[i] = self._apply_price_shock(df.iloc[i], magnitude)
            
            # Ajouter de la volatilité supplémentaire
            df.iloc[i] = self._add_volatility(df.iloc[i])
        
        return df
    
    def _apply_price_shock(self, row: pd.Series, magnitude: float) -> pd.Series:
        """
        Applique un choc de prix à une bougie.
        
        Args:
            row: Données OHLCV d'une bougie
            magnitude: Amplitude du choc (-1 à 1)
        
        Returns:
            Données modifiées
        """
        base_price = row['open']
        shock = base_price * magnitude
        
        row['high'] = max(row['open'], row['close']) + abs(shock)
        row['low'] = min(row['open'], row['close']) - abs(shock)
        row['close'] += shock
        
        # Augmenter le volume pendant les chocs
        row['volume'] *= (1 + abs(magnitude) * 5)
        
        return row
    
    def _add_volatility(self, row: pd.Series) -> pd.Series:
        """
        Ajoute de la volatilité supplémentaire à une bougie.
        
        Args:
            row: Données OHLCV d'une bougie
        
        Returns:
            Données avec volatilité accrue
        """
        volatility = np.random.normal(0, self.config.volatility_factor * 0.01)
        
        row['high'] *= (1 + abs(volatility))
        row['low'] *= (1 - abs(volatility))
        row['close'] *= (1 + volatility)
        
        return row

class NetworkStressSimulator:
    """Simulateur de conditions réseau dégradées."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.current_condition = NetworkCondition.NORMAL
        self.offline_until = None
        self.current_latency = config.base_latency
    
    def get_current_latency(self) -> int:
        """
        Calcule la latence actuelle en fonction des conditions.
        
        Returns:
            Latence en millisecondes
        """
        if self.current_condition == NetworkCondition.OFFLINE:
            return float('inf')
        
        if self.current_condition == NetworkCondition.HIGH_LATENCY:
            base = self.config.max_latency
        else:
            base = self.config.base_latency
        
        # Ajouter de la variation aléatoire
        jitter = np.random.normal(0, base * self.config.latency_volatility)
        latency = base + jitter
        
        return max(1, int(latency))
    
    def is_connected(self) -> bool:
        """
        Vérifie si la connexion est active.
        
        Returns:
            True si connecté
        """
        if self.current_condition == NetworkCondition.OFFLINE:
            if datetime.now() >= self.offline_until:
                self.current_condition = NetworkCondition.NORMAL
            else:
                return False
        
        if random.random() < self.config.connection_loss_probability:
            self.current_condition = NetworkCondition.OFFLINE
            self.offline_until = datetime.now() + timedelta(
                seconds=random.randint(1, self.config.max_offline_duration)
            )
            return False
        
        return True

class StressTestEngine(BacktestEngine):
    """Moteur de backtest avec simulation de conditions extrêmes."""
    
    def __init__(
        self,
        backtest_config: BacktestConfig,
        stress_config: StressTestConfig
    ):
        super().__init__(backtest_config)
        self.market_simulator = MarketStressSimulator(stress_config)
        self.network_simulator = NetworkStressSimulator(stress_config)
    
    async def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict:
        """
        Exécute le backtest avec conditions stressantes.
        
        Args:
            data: Données historiques
            strategy: Stratégie à tester
        
        Returns:
            Résultats du backtest
        """
        # Appliquer le stress de marché
        stressed_data = self.market_simulator.apply_stress(data)
        
        # Sauvegarder les conditions originales
        original_latency = self.config.latency
        
        try:
            for i in range(len(stressed_data)):
                # Simuler les conditions réseau
                if not self.network_simulator.is_connected():
                    logger.warning(f"Perte de connexion à {stressed_data.index[i]}")
                    continue
                
                # Mettre à jour la latence
                self.config.latency = self.network_simulator.get_current_latency()
                
                # Exécuter une itération du backtest
                await self._process_bar(stressed_data.iloc[i:i+1], strategy)
                
                # Loguer les conditions extrêmes
                if self.market_simulator.current_condition != MarketCondition.NORMAL:
                    logger.warning(
                        f"Condition de marché: {self.market_simulator.current_condition.value} "
                        f"à {stressed_data.index[i]}"
                    )
                
                if self.config.latency > original_latency * 2:
                    logger.warning(f"Latence élevée: {self.config.latency}ms")
            
            return self.results
            
        finally:
            # Restaurer la configuration originale
            self.config.latency = original_latency
    
    async def _process_bar(self, data: pd.DataFrame, strategy: BaseStrategy):
        """
        Traite une bougie de données avec gestion des erreurs.
        
        Args:
            data: Données d'une bougie
            strategy: Stratégie à exécuter
        """
        try:
            # Simuler la latence réseau
            await asyncio.sleep(self.config.latency / 1000)
            
            # Exécuter la logique de trading
            signal = await strategy.analyze(data)
            
            if signal:
                # Vérifier si l'ordre peut être exécuté
                if self.network_simulator.is_connected():
                    await self._execute_signal(signal, data.iloc[-1])
                else:
                    logger.error("Ordre non exécuté : perte de connexion")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement : {str(e)}")
            # Continuer malgré l'erreur pour la simulation
