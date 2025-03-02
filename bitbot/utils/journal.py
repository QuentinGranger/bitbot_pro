"""
Module de journalisation avancée pour BitBotPro.

Ce module permet d'enregistrer de manière détaillée toutes les décisions,
actions et événements du bot de trading dans des journaux structurés
pour faciliter l'analyse post-mortem.
"""

import logging
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration du logger standard
logger = logging.getLogger(__name__)

class TradingJournal:
    """
    Journal de trading avancé qui enregistre toutes les décisions et actions
    du bot dans différents formats (JSON, CSV, etc.) pour analyse post-mortem.
    """
    
    def __init__(self, journal_dir: str = "journals"):
        """
        Initialise le journal de trading.
        
        Args:
            journal_dir: Répertoire où stocker les journaux
        """
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-répertoires pour différents types de journaux
        self.decisions_dir = self.journal_dir / "decisions"
        self.signals_dir = self.journal_dir / "signals"
        self.orders_dir = self.journal_dir / "orders"
        self.performance_dir = self.journal_dir / "performance"
        self.errors_dir = self.journal_dir / "errors"
        self.market_data_dir = self.journal_dir / "market_data"
        
        # Créer tous les sous-répertoires
        for directory in [
            self.decisions_dir, 
            self.signals_dir, 
            self.orders_dir, 
            self.performance_dir, 
            self.errors_dir,
            self.market_data_dir
        ]:
            directory.mkdir(exist_ok=True)
        
        # Initialiser les fichiers de journal pour la session actuelle
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.decisions_file = self.decisions_dir / f"decisions_{self.session_id}.jsonl"
        self.signals_file = self.signals_dir / f"signals_{self.session_id}.jsonl"
        self.orders_file = self.orders_dir / f"orders_{self.session_id}.jsonl"
        self.errors_file = self.errors_dir / f"errors_{self.session_id}.jsonl"
        
        # Dataframes pour la journalisation en mémoire avant écriture
        self.decisions_df = pd.DataFrame()
        self.signals_df = pd.DataFrame()
        self.orders_df = pd.DataFrame()
        self.errors_df = pd.DataFrame()
        
        # Compteurs pour les statistiques
        self.stats = {
            "decisions_count": 0,
            "signals_count": 0,
            "orders_count": 0,
            "errors_count": 0,
            "session_start": datetime.now()
        }
        
        logger.info(f"Journal de trading initialisé avec l'ID de session {self.session_id}")
    
    def log_decision(self, 
                    decision_type: str, 
                    symbol: str,
                    details: Dict[str, Any], 
                    reasons: List[str] = None,
                    indicators: Dict[str, Any] = None,
                    context: Dict[str, Any] = None) -> None:
        """
        Journalise une décision prise par le bot.
        
        Args:
            decision_type: Type de décision (ex: "market_entry", "position_close", etc.)
            symbol: Symbole concerné par la décision
            details: Détails de la décision
            reasons: Liste des raisons qui ont conduit à cette décision
            indicators: Valeurs des indicateurs utilisés pour prendre la décision
            context: Contexte supplémentaire (conditions de marché, etc.)
        """
        timestamp = datetime.now()
        
        decision_entry = {
            "timestamp": timestamp.isoformat(),
            "decision_type": decision_type,
            "symbol": symbol,
            "details": details,
            "reasons": reasons or [],
            "indicators": indicators or {},
            "context": context or {}
        }
        
        # Ajouter à la liste de décisions en mémoire
        with open(self.decisions_file, "a") as f:
            f.write(json.dumps(decision_entry) + "\n")
        
        self.stats["decisions_count"] += 1
        logger.debug(f"Décision journalisée: {decision_type} pour {symbol}")
    
    def log_signal(self, signal: Any) -> None:
        """
        Journalise un signal généré par une stratégie.
        
        Args:
            signal: Objet Signal ou dictionnaire contenant les détails du signal
        """
        timestamp = datetime.now()
        
        # Convertir l'objet Signal en dictionnaire si ce n'est pas déjà le cas
        if hasattr(signal, "__dict__"):
            signal_dict = signal.__dict__
        else:
            signal_dict = signal
        
        # Ajouter le timestamp
        signal_entry = {
            "timestamp": timestamp.isoformat(),
            "signal": signal_dict
        }
        
        # Écrire dans le fichier
        with open(self.signals_file, "a") as f:
            f.write(json.dumps(signal_entry) + "\n")
        
        self.stats["signals_count"] += 1
        logger.debug(f"Signal journalisé: {signal_dict.get('side', 'UNKNOWN')} pour {signal_dict.get('symbol', 'UNKNOWN')}")
    
    def log_order(self, order: Dict[str, Any]) -> None:
        """
        Journalise un ordre placé sur le marché.
        
        Args:
            order: Détails de l'ordre
        """
        timestamp = datetime.now()
        
        order_entry = {
            "timestamp": timestamp.isoformat(),
            "order": order
        }
        
        # Écrire dans le fichier
        with open(self.orders_file, "a") as f:
            f.write(json.dumps(order_entry) + "\n")
        
        self.stats["orders_count"] += 1
        logger.debug(f"Ordre journalisé: {order.get('side', 'UNKNOWN')} pour {order.get('symbol', 'UNKNOWN')}")
    
    def log_error(self, error_type: str, message: str, details: Dict[str, Any] = None) -> None:
        """
        Journalise une erreur rencontrée pendant l'exécution.
        
        Args:
            error_type: Type d'erreur
            message: Message d'erreur
            details: Détails supplémentaires sur l'erreur
        """
        timestamp = datetime.now()
        
        error_entry = {
            "timestamp": timestamp.isoformat(),
            "error_type": error_type,
            "message": message,
            "details": details or {}
        }
        
        # Écrire dans le fichier
        with open(self.errors_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        
        self.stats["errors_count"] += 1
        logger.error(f"Erreur journalisée: {error_type} - {message}")
    
    def log_market_data(self, symbol: str, timeframe: str, data: Any) -> None:
        """
        Journalise les données de marché pour analyse ultérieure.
        
        Args:
            symbol: Symbole de la paire de trading
            timeframe: Intervalle de temps
            data: Données de marché à journaliser
        """
        timestamp = datetime.now()
        
        # Créer un nom de fichier à partir du symbole, timeframe et timestamp
        filename = f"{symbol}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = self.market_data_dir / filename
        
        # Convertir en DataFrame si ce n'est pas déjà le cas
        if hasattr(data, "to_parquet"):
            df = data
        elif hasattr(data, "ohlcv") and hasattr(data.ohlcv, "to_parquet"):
            df = data.ohlcv
        else:
            # Essayer de convertir le dictionnaire en DataFrame
            try:
                df = pd.DataFrame(data)
            except:
                logger.error(f"Impossible de convertir les données de marché en DataFrame pour {symbol}_{timeframe}")
                return
        
        # Sauvegarder au format parquet compressé
        try:
            df.to_parquet(filepath, compression="gzip")
            logger.debug(f"Données de marché journalisées pour {symbol}_{timeframe}")
        except Exception as e:
            logger.error(f"Erreur lors de la journalisation des données de marché: {e}")
    
    def generate_session_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de session avec des statistiques sur l'activité du bot.
        
        Returns:
            Rapport de session
        """
        session_end = datetime.now()
        duration = (session_end - self.stats["session_start"]).total_seconds() / 60  # en minutes
        
        report = {
            "session_id": self.session_id,
            "session_start": self.stats["session_start"].isoformat(),
            "session_end": session_end.isoformat(),
            "duration_minutes": round(duration, 2),
            "decisions_count": self.stats["decisions_count"],
            "signals_count": self.stats["signals_count"],
            "orders_count": self.stats["orders_count"],
            "errors_count": self.stats["errors_count"],
            "decisions_per_minute": round(self.stats["decisions_count"] / max(1, duration), 2),
            "signals_per_minute": round(self.stats["signals_count"] / max(1, duration), 2),
            "orders_per_minute": round(self.stats["orders_count"] / max(1, duration), 2)
        }
        
        # Sauvegarder le rapport
        report_file = self.journal_dir / f"session_report_{self.session_id}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def analyze_decisions(self, symbol: str = None, decision_type: str = None, 
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Analyse les décisions journalisées.
        
        Args:
            symbol: Filtrer par symbole
            decision_type: Filtrer par type de décision
            start_date: Date de début (format ISO)
            end_date: Date de fin (format ISO)
            
        Returns:
            DataFrame contenant les décisions filtrées
        """
        # Lire toutes les décisions du fichier
        decisions = []
        
        try:
            with open(self.decisions_file, "r") as f:
                for line in f:
                    decisions.append(json.loads(line))
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier de décisions: {e}")
            return pd.DataFrame()
        
        # Convertir en DataFrame
        df = pd.DataFrame(decisions)
        
        if df.empty:
            return df
        
        # Convertir la colonne timestamp en datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Appliquer les filtres
        if symbol:
            df = df[df["symbol"] == symbol]
        
        if decision_type:
            df = df[df["decision_type"] == decision_type]
        
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        
        return df
    
    def find_error_patterns(self) -> Dict[str, Any]:
        """
        Analyse les erreurs pour identifier des motifs récurrents.
        
        Returns:
            Dictionnaire contenant les patterns d'erreurs identifiés
        """
        # Lire toutes les erreurs du fichier
        errors = []
        
        try:
            with open(self.errors_file, "r") as f:
                for line in f:
                    errors.append(json.loads(line))
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier d'erreurs: {e}")
            return {}
        
        if not errors:
            return {"message": "Aucune erreur journalisée"}
        
        # Convertir en DataFrame
        df = pd.DataFrame(errors)
        
        # Compter les types d'erreurs
        error_counts = df["error_type"].value_counts().to_dict()
        
        # Trouver les périodes avec beaucoup d'erreurs
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        hourly_errors = df.groupby("hour").size().to_dict()
        
        return {
            "error_counts": error_counts,
            "hourly_distribution": hourly_errors,
            "total_errors": len(errors),
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }
    
    def correlate_errors_with_decisions(self) -> Dict[str, Any]:
        """
        Corrèle les erreurs avec les décisions pour identifier les décisions problématiques.
        
        Returns:
            Dictionnaire contenant les corrélations identifiées
        """
        # Lire les erreurs et décisions
        errors = []
        decisions = []
        
        try:
            with open(self.errors_file, "r") as f:
                for line in f:
                    errors.append(json.loads(line))
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier d'erreurs: {e}")
            return {}
        
        try:
            with open(self.decisions_file, "r") as f:
                for line in f:
                    decisions.append(json.loads(line))
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier de décisions: {e}")
            return {}
        
        if not errors or not decisions:
            return {"message": "Données insuffisantes pour la corrélation"}
        
        # Convertir en DataFrame
        errors_df = pd.DataFrame(errors)
        decisions_df = pd.DataFrame(decisions)
        
        # Convertir les timestamps
        errors_df["timestamp"] = pd.to_datetime(errors_df["timestamp"])
        decisions_df["timestamp"] = pd.to_datetime(decisions_df["timestamp"])
        
        # Ajouter une colonne avec l'heure arrondie pour faciliter la corrélation
        errors_df["hour_rounded"] = errors_df["timestamp"].dt.floor("H")
        decisions_df["hour_rounded"] = decisions_df["timestamp"].dt.floor("H")
        
        # Compter les erreurs et décisions par heure
        hourly_errors = errors_df.groupby("hour_rounded").size()
        hourly_decisions = decisions_df.groupby("hour_rounded").size()
        
        # Fusionner les deux
        hourly_data = pd.DataFrame({
            "errors": hourly_errors,
            "decisions": hourly_decisions
        }).fillna(0)
        
        # Calculer la corrélation
        correlation = hourly_data["errors"].corr(hourly_data["decisions"])
        
        return {
            "correlation": correlation,
            "interpretation": "Corrélation positive forte" if correlation > 0.7 else 
                            "Corrélation positive modérée" if correlation > 0.3 else
                            "Corrélation négative forte" if correlation < -0.7 else
                            "Corrélation négative modérée" if correlation < -0.3 else
                            "Faible corrélation"
        }
    
    def get_decision_timeline(self, symbol: str = None, start_date: str = None, 
                             end_date: str = None) -> pd.DataFrame:
        """
        Récupère une chronologie des décisions pour analyse visuelle.
        
        Args:
            symbol: Filtrer par symbole
            start_date: Date de début (format ISO)
            end_date: Date de fin (format ISO)
            
        Returns:
            DataFrame contenant la chronologie des décisions
        """
        # Analyser les décisions avec les filtres fournis
        decisions_df = self.analyze_decisions(symbol, None, start_date, end_date)
        
        if decisions_df.empty:
            return pd.DataFrame()
        
        # Créer une chronologie simplifiée
        timeline = decisions_df[["timestamp", "symbol", "decision_type"]].copy()
        
        # Ajouter des détails simplifiés
        def extract_details(row):
            details = row.get("details", {})
            return ", ".join([f"{k}: {v}" for k, v in details.items() if k in ["price", "side", "amount", "action"]])
        
        timeline["details"] = decisions_df.apply(lambda row: extract_details(row), axis=1)
        
        return timeline.sort_values("timestamp")

# Créer une instance singleton du journal
trading_journal = TradingJournal()

# Fonctions d'accès simplifié pour l'API externe
def log_decision(*args, **kwargs):
    """Fonction d'accès simplifié pour journaliser une décision."""
    return trading_journal.log_decision(*args, **kwargs)

def log_signal(*args, **kwargs):
    """Fonction d'accès simplifié pour journaliser un signal."""
    return trading_journal.log_signal(*args, **kwargs)

def log_order(*args, **kwargs):
    """Fonction d'accès simplifié pour journaliser un ordre."""
    return trading_journal.log_order(*args, **kwargs)

def log_error(*args, **kwargs):
    """Fonction d'accès simplifié pour journaliser une erreur."""
    return trading_journal.log_error(*args, **kwargs)

def log_market_data(*args, **kwargs):
    """Fonction d'accès simplifié pour journaliser des données de marché."""
    return trading_journal.log_market_data(*args, **kwargs)

def generate_report():
    """Fonction d'accès simplifié pour générer un rapport de session."""
    return trading_journal.generate_session_report()

def analyze_decisions(*args, **kwargs):
    """Fonction d'accès simplifié pour analyser les décisions."""
    return trading_journal.analyze_decisions(*args, **kwargs)

# Exemple d'utilisation:
# from bitbot.utils.journal import log_decision, log_signal, log_order, log_error, log_market_data
# 
# log_decision(
#     decision_type="market_entry", 
#     symbol="BTCUSDT",
#     details={"side": "BUY", "price": 50000, "amount": 0.1},
#     reasons=["RSI oversold", "MACD bullish crossover"],
#     indicators={"rsi": 28, "macd_histogram": 0.5},
#     context={"market_trend": "bullish", "volatility": "low"}
# )
