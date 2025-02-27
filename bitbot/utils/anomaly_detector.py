"""
Détection d'anomalies dans les flux de données de marché.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass
from enum import Enum

from bitbot.utils.logger import logger


class AnomalyType(Enum):
    """Types d'anomalies détectables."""
    VOLUME_SPIKE = "volume_spike"
    PRICE_SPIKE = "price_spike"
    DATA_GAP = "data_gap"
    SEQUENCE_GAP = "sequence_gap"
    NEGATIVE_SPREAD = "negative_spread"
    TIMESTAMP_REVERSAL = "timestamp_reversal"
    INSTRUMENT_HALT = "instrument_halt"


@dataclass
class Anomaly:
    """Représentation d'une anomalie détectée."""
    
    anomaly_type: AnomalyType
    symbol: str
    timestamp: float
    severity: float  # 0.0 à 1.0, où 1.0 est le plus sévère
    details: Dict[str, Any]
    is_confirmed: bool = False
    recovery_action: Optional[str] = None
    
    def __str__(self) -> str:
        """Représentation textuelle de l'anomalie."""
        dt = datetime.fromtimestamp(self.timestamp)
        return (f"Anomalie {self.anomaly_type.value} pour {self.symbol} à {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}, "
                f"sévérité {self.severity:.2f}")


class AnomalyDetector:
    """
    Détecteur d'anomalies pour les flux de données de marché.
    
    Cette classe analyse les données de marché en temps réel pour détecter
    diverses anomalies comme les spikes de volume, les gaps de données, etc.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le détecteur d'anomalies.
        
        Args:
            config: Configuration optionnelle pour les seuils de détection
        """
        self.config = config or {}
        
        # Seuils par défaut
        self.default_thresholds = {
            "volume_spike_z_score": 3.0,  # Z-score pour considérer un spike de volume
            "price_spike_z_score": 4.0,   # Z-score pour considérer un spike de prix
            "data_gap_threshold": 2.0,    # Multiple de l'intervalle attendu pour considérer un gap
            "min_samples_for_baseline": 10,  # Minimum d'échantillons pour établir une baseline
            "max_history_window": 100,    # Taille maximale de la fenêtre d'historique
            "sequence_gap_threshold": 2,  # Seuil pour considérer un gap de séquence
        }
        
        # Fusionner avec la configuration fournie
        for key, value in self.default_thresholds.items():
            if key not in self.config:
                self.config[key] = value
        
        # Historique des données pour chaque symbole et timeframe
        self.data_history: Dict[str, List[Any]] = {}
        
        # Historique des anomalies détectées
        self.anomalies: List[Anomaly] = []
        
        # Statistiques de baseline pour chaque symbole/timeframe
        self.baselines: Dict[str, Dict[str, Any]] = {}
    
    def add_data_point(self, symbol: str, stream_type: str, data: Dict) -> List[Anomaly]:
        """
        Ajoute un point de données et vérifie les anomalies.
        
        Args:
            symbol: Symbole de trading
            stream_type: Type de flux (kline, trade, etc.)
            data: Données reçues
            
        Returns:
            Liste des anomalies détectées
        """
        key = f"{symbol.lower()}@{stream_type}"
        
        # Initialiser l'historique si nécessaire
        if key not in self.data_history:
            self.data_history[key] = []
            self.baselines[key] = {
                "volume_mean": None,
                "volume_std": None,
                "price_mean": None,
                "price_std": None,
                "last_timestamp": None,
                "expected_interval": None,
                "last_sequence": None,
            }
        
        # Ajouter les données à l'historique
        self.data_history[key].append(data)
        
        # Limiter la taille de l'historique
        if len(self.data_history[key]) > self.config["max_history_window"]:
            self.data_history[key] = self.data_history[key][-self.config["max_history_window"]:]
        
        # Détecter les anomalies
        detected_anomalies = []
        
        # Mettre à jour les baselines uniquement après avoir assez de données
        if len(self.data_history[key]) >= self.config["min_samples_for_baseline"]:
            self._update_baseline(key)
            
            # Exécuter les détecteurs appropriés selon le type de flux
            if stream_type == "kline":
                volume_anomalies = self._detect_volume_spikes(key, data)
                price_anomalies = self._detect_price_spikes(key, data)
                timestamp_anomalies = self._detect_timestamp_anomalies(key, data)
                
                detected_anomalies.extend(volume_anomalies)
                detected_anomalies.extend(price_anomalies)
                detected_anomalies.extend(timestamp_anomalies)
                
            elif stream_type in ("trade", "aggTrade"):
                price_anomalies = self._detect_price_spikes(key, data)
                detected_anomalies.extend(price_anomalies)
                
            elif stream_type == "depth":
                spread_anomalies = self._detect_negative_spread(key, data)
                detected_anomalies.extend(spread_anomalies)
        
        # Ajouter les anomalies détectées à l'historique
        if detected_anomalies:
            self.anomalies.extend(detected_anomalies)
            # Limiter la taille de l'historique des anomalies
            if len(self.anomalies) > 1000:
                self.anomalies = self.anomalies[-1000:]
        
        return detected_anomalies
    
    def _update_baseline(self, key: str) -> None:
        """
        Met à jour les statistiques de référence pour un symbole/timeframe.
        
        Args:
            key: Clé symbole@stream_type
        """
        data = self.data_history[key]
        
        # Extraire les volumes et prix des klines
        volumes = []
        prices = []
        timestamps = []
        
        for item in data:
            if 'k' in item:  # Kline data
                volumes.append(float(item['k']['v']))
                prices.append(float(item['k']['c']))
                timestamps.append(item['k']['t'] / 1000)  # Convertir en secondes
            elif 'p' in item:  # Trade data
                prices.append(float(item['p']))
                if 'q' in item:
                    volumes.append(float(item['q']))
                timestamps.append(item['T'] / 1000)  # Convertir en secondes
        
        # Calculer les statistiques de volume si disponibles
        if volumes:
            self.baselines[key]["volume_mean"] = np.mean(volumes)
            self.baselines[key]["volume_std"] = np.std(volumes)
        
        # Calculer les statistiques de prix
        if prices:
            self.baselines[key]["price_mean"] = np.mean(prices)
            self.baselines[key]["price_std"] = np.std(prices)
        
        # Déterminer l'intervalle attendu entre les données
        if len(timestamps) >= 2:
            diffs = np.diff(sorted(timestamps))
            # Utiliser la médiane pour être robuste aux outliers
            self.baselines[key]["expected_interval"] = np.median(diffs)
        
        # Enregistrer le dernier timestamp
        if timestamps:
            self.baselines[key]["last_timestamp"] = max(timestamps)
    
    def _detect_volume_spikes(self, key: str, data: Dict) -> List[Anomaly]:
        """
        Détecte les spikes anormaux de volume.
        
        Args:
            key: Clé symbole@stream_type
            data: Point de données actuel
            
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Vérifier que nous avons les données nécessaires
        if 'k' not in data or self.baselines[key]["volume_mean"] is None:
            return anomalies
        
        # Extraire le volume
        volume = float(data['k']['v'])
        symbol = data['s']
        timestamp = data['k']['t'] / 1000  # Convertir en secondes
        
        # Calculer le z-score du volume
        volume_mean = self.baselines[key]["volume_mean"]
        volume_std = self.baselines[key]["volume_std"]
        
        if volume_std > 0:
            z_score = (volume - volume_mean) / volume_std
            
            # Détecter un spike si le z-score dépasse le seuil
            if z_score > self.config["volume_spike_z_score"]:
                severity = min(1.0, (z_score - self.config["volume_spike_z_score"]) / 10)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    symbol=symbol,
                    timestamp=timestamp,
                    severity=severity,
                    details={
                        "volume": volume,
                        "mean_volume": volume_mean,
                        "std_volume": volume_std,
                        "z_score": z_score,
                        "threshold": self.config["volume_spike_z_score"]
                    }
                ))
        
        return anomalies
    
    def _detect_price_spikes(self, key: str, data: Dict) -> List[Anomaly]:
        """
        Détecte les spikes anormaux de prix.
        
        Args:
            key: Clé symbole@stream_type
            data: Point de données actuel
            
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Vérifier que nous avons les données nécessaires
        if self.baselines[key]["price_mean"] is None:
            return anomalies
        
        # Extraire le prix en fonction du type de données
        price = None
        symbol = None
        timestamp = None
        
        if 'k' in data:  # Kline data
            price = float(data['k']['c'])
            symbol = data['s']
            timestamp = data['k']['t'] / 1000
        elif 'p' in data:  # Trade data
            price = float(data['p'])
            symbol = data['s']
            timestamp = data['T'] / 1000
        
        if price is None or symbol is None or timestamp is None:
            return anomalies
        
        # Calculer le z-score du prix
        price_mean = self.baselines[key]["price_mean"]
        price_std = self.baselines[key]["price_std"]
        
        if price_std > 0:
            z_score = abs(price - price_mean) / price_std
            
            # Détecter un spike si le z-score dépasse le seuil
            if z_score > self.config["price_spike_z_score"]:
                severity = min(1.0, (z_score - self.config["price_spike_z_score"]) / 10)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    symbol=symbol,
                    timestamp=timestamp,
                    severity=severity,
                    details={
                        "price": price,
                        "mean_price": price_mean,
                        "std_price": price_std,
                        "z_score": z_score,
                        "threshold": self.config["price_spike_z_score"]
                    }
                ))
        
        return anomalies
    
    def _detect_timestamp_anomalies(self, key: str, data: Dict) -> List[Anomaly]:
        """
        Détecte les anomalies liées aux timestamps (gaps, inversions).
        
        Args:
            key: Clé symbole@stream_type
            data: Point de données actuel
            
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Vérifier que nous avons les données nécessaires
        if self.baselines[key]["last_timestamp"] is None:
            return anomalies
        
        # Extraire le timestamp
        if 'k' in data:
            timestamp = data['k']['t'] / 1000
            symbol = data['s']
        elif 'T' in data:
            timestamp = data['T'] / 1000
            symbol = data['s']
        else:
            return anomalies
        
        last_timestamp = self.baselines[key]["last_timestamp"]
        expected_interval = self.baselines[key]["expected_interval"]
        
        if expected_interval:
            # Détecter un gap de données
            time_diff = timestamp - last_timestamp
            if time_diff > expected_interval * self.config["data_gap_threshold"]:
                severity = min(1.0, (time_diff / expected_interval) / 10)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.DATA_GAP,
                    symbol=symbol,
                    timestamp=timestamp,
                    severity=severity,
                    details={
                        "gap_duration": time_diff,
                        "expected_interval": expected_interval,
                        "last_timestamp": last_timestamp,
                        "threshold": self.config["data_gap_threshold"]
                    },
                    recovery_action="retrieve_historical_data"
                ))
            
            # Détecter une inversion de timestamp
            elif time_diff < 0:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.TIMESTAMP_REVERSAL,
                    symbol=symbol,
                    timestamp=timestamp,
                    severity=0.7,  # Sévérité fixe pour ce type d'anomalie
                    details={
                        "current_timestamp": timestamp,
                        "previous_timestamp": last_timestamp,
                        "time_difference": time_diff
                    }
                ))
        
        return anomalies
    
    def _detect_negative_spread(self, key: str, data: Dict) -> List[Anomaly]:
        """
        Détecte un spread négatif dans le carnet d'ordres.
        
        Args:
            key: Clé symbole@stream_type
            data: Point de données actuel
            
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Vérifier si nous avons des données de profondeur
        if 'b' not in data or 'a' not in data:
            return anomalies
        
        # Extraire le meilleur bid et le meilleur ask
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        if not bids or not asks:
            return anomalies
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else float('inf')
        
        # Un spread négatif est une anomalie grave
        if best_bid > best_ask:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.NEGATIVE_SPREAD,
                symbol=key.split('@')[0],
                timestamp=time.time(),
                severity=1.0,  # Toujours maximale pour un spread négatif
                details={
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": best_bid - best_ask
                }
            ))
        
        return anomalies
    
    def get_recent_anomalies(self, symbol: Optional[str] = None, 
                           anomaly_type: Optional[AnomalyType] = None,
                           since: Optional[float] = None,
                           max_count: int = 100) -> List[Anomaly]:
        """
        Récupère les anomalies récentes filtrées par critères.
        
        Args:
            symbol: Filtrer par symbole
            anomaly_type: Filtrer par type d'anomalie
            since: Timestamp minimum
            max_count: Nombre maximum d'anomalies à retourner
            
        Returns:
            Liste des anomalies filtrées
        """
        filtered = self.anomalies
        
        if symbol:
            filtered = [a for a in filtered if a.symbol == symbol]
        
        if anomaly_type:
            filtered = [a for a in filtered if a.anomaly_type == anomaly_type]
        
        if since:
            filtered = [a for a in filtered if a.timestamp >= since]
        
        # Trier par timestamp décroissant
        filtered.sort(key=lambda a: a.timestamp, reverse=True)
        
        return filtered[:max_count]
    
    def get_anomaly_summary(self, window_seconds: int = 3600) -> Dict:
        """
        Génère un résumé des anomalies sur une période donnée.
        
        Args:
            window_seconds: Fenêtre de temps en secondes (défaut: 1 heure)
            
        Returns:
            Résumé des anomalies
        """
        since = time.time() - window_seconds
        recent = self.get_recent_anomalies(since=since)
        
        # Regrouper par type
        by_type = {}
        for anomaly in recent:
            atype = anomaly.anomaly_type.value
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly)
        
        # Regrouper par symbole
        by_symbol = {}
        for anomaly in recent:
            if anomaly.symbol not in by_symbol:
                by_symbol[anomaly.symbol] = []
            by_symbol[anomaly.symbol].append(anomaly)
        
        # Calculer les statistiques
        summary = {
            "total_anomalies": len(recent),
            "window_seconds": window_seconds,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_symbol": {k: len(v) for k, v in by_symbol.items()},
            "severity_distribution": {
                "low": len([a for a in recent if a.severity < 0.3]),
                "medium": len([a for a in recent if 0.3 <= a.severity < 0.7]),
                "high": len([a for a in recent if a.severity >= 0.7])
            },
            "most_severe": sorted(recent, key=lambda a: a.severity, reverse=True)[:5] if recent else []
        }
        
        return summary
