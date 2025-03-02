"""
Module de surveillance des performances du bot de trading.

Ce module fournit des outils pour surveiller les performances du bot de trading
et générer des alertes en cas de problèmes détectés (drawdown anormal, 
exécutions d'ordres inhabituelles, etc.).
"""
import os
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType
from bitbot.utils.logger import logger


class PerformanceMonitor:
    """
    Moniteur de performance pour le bot de trading.
    
    Surveille différentes métriques de performance et génère des alertes
    en cas de comportement anormal.
    """
    
    def __init__(self, 
                initial_balance: float = 10000.0,
                max_drawdown_threshold: float = 5.0,  # 5% de drawdown maximum autorisé par défaut
                drawdown_alert_threshold: float = 2.0,  # Alerte à partir de 2% de drawdown
                order_frequency_window: int = 24,  # Fenêtre de 24 heures pour la fréquence des ordres
                historical_data_window: int = 30,  # 30 jours d'historique pour la comparaison
                ):
        """
        Initialise le moniteur de performance.
        
        Args:
            initial_balance: Balance initiale du portefeuille
            max_drawdown_threshold: Seuil maximum de drawdown toléré (en %)
            drawdown_alert_threshold: Seuil à partir duquel une alerte est générée (en %)
            order_frequency_window: Fenêtre de temps (en heures) pour surveiller la fréquence des ordres
            historical_data_window: Fenêtre de temps (en jours) pour l'historique de comparaison
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown_threshold = max_drawdown_threshold
        self.drawdown_alert_threshold = drawdown_alert_threshold
        self.order_frequency_window = order_frequency_window
        self.historical_data_window = historical_data_window
        
        # Historique des transactions et performances
        self.trades_history: List[Dict[str, Any]] = []
        self.balance_history: List[Dict[str, Any]] = []
        self.drawdown_history: List[Dict[str, Any]] = []
        
        # État de surveillance
        self.last_drawdown_alert_time = datetime.now() - timedelta(days=1)  # Pour éviter des alertes au démarrage
        self.last_order_frequency_alert_time = datetime.now() - timedelta(days=1)
        
        # Configuration
        self.order_frequency_threshold = 10  # Nombre d'ordres par heure considéré comme anormal
        
        logger.info("Moniteur de performance initialisé")
    
    def update_balance(self, new_balance: float, timestamp: Optional[datetime] = None) -> None:
        """
        Met à jour la balance du portefeuille et calcule le drawdown.
        
        Args:
            new_balance: Nouvelle valeur de la balance
            timestamp: Horodatage de la mise à jour (utilise l'heure actuelle si non spécifié)
        """
        timestamp = timestamp or datetime.now()
        
        # Mettre à jour les valeurs actuelles
        self.current_balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)
        
        # Calculer le drawdown actuel
        current_drawdown = 0.0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - new_balance) / self.peak_balance * 100
        
        # Enregistrer dans l'historique
        balance_entry = {
            "timestamp": timestamp,
            "balance": new_balance,
            "peak_balance": self.peak_balance,
            "drawdown": current_drawdown
        }
        
        self.balance_history.append(balance_entry)
        self.drawdown_history.append({
            "timestamp": timestamp,
            "drawdown": current_drawdown
        })
        
        # Vérifier si le drawdown nécessite une alerte
        self._check_drawdown_alert(current_drawdown, timestamp)
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Enregistre une transaction dans l'historique.
        
        Args:
            trade: Détails de la transaction (timestamp, symbol, side, price, quantity, etc.)
        """
        if "timestamp" not in trade:
            trade["timestamp"] = datetime.now()
            
        self.trades_history.append(trade)
        
        # Vérifier la fréquence des ordres
        self._check_order_frequency_alert()
    
    def _check_drawdown_alert(self, current_drawdown: float, timestamp: datetime) -> None:
        """
        Vérifie si le drawdown actuel nécessite l'envoi d'une alerte.
        
        Args:
            current_drawdown: Drawdown actuel en pourcentage
            timestamp: Horodatage actuel
        """
        # Éviter d'envoyer des alertes trop fréquemment (max une par heure)
        time_since_last_alert = timestamp - self.last_drawdown_alert_time
        if time_since_last_alert.total_seconds() < 3600:
            return
            
        # Générer différents niveaux d'alerte selon la gravité du drawdown
        if current_drawdown >= self.max_drawdown_threshold:
            # Drawdown critique dépassant le seuil maximum autorisé
            asyncio.create_task(self._send_drawdown_alert(
                current_drawdown, 
                NotificationPriority.CRITICAL,
                "DRAWDOWN CRITIQUE"
            ))
            self.last_drawdown_alert_time = timestamp
            
        elif current_drawdown >= self.drawdown_alert_threshold:
            # Drawdown important mais sous le seuil critique
            asyncio.create_task(self._send_drawdown_alert(
                current_drawdown, 
                NotificationPriority.HIGH,
                "Drawdown important"
            ))
            self.last_drawdown_alert_time = timestamp
    
    async def _send_drawdown_alert(self, drawdown: float, priority: NotificationPriority, 
                                  title: str) -> None:
        """
        Envoie une alerte de drawdown via le gestionnaire de notifications.
        
        Args:
            drawdown: Valeur du drawdown actuel (en %)
            priority: Niveau de priorité de la notification
            title: Titre de la notification
        """
        # Calculer des statistiques supplémentaires pour contextualiser l'alerte
        avg_drawdown = self._calculate_average_drawdown(days=7)
        
        details = {
            "drawdown_actuel": f"{drawdown:.2f}%",
            "drawdown_moyen_7j": f"{avg_drawdown:.2f}%",
            "balance_actuelle": f"{self.current_balance:.2f}",
            "balance_max": f"{self.peak_balance:.2f}",
            "perte_absolue": f"{self.peak_balance - self.current_balance:.2f}",
            "seuil_alerte": f"{self.drawdown_alert_threshold:.2f}%",
            "seuil_maximum": f"{self.max_drawdown_threshold:.2f}%"
        }
        
        emoji = "🔥" if priority == NotificationPriority.CRITICAL else "📉"
        
        message = (
            f"{emoji} *{title}* {emoji}\n\n"
            f"Drawdown actuel: *{drawdown:.2f}%*\n"
            f"Balance: {self.current_balance:.2f} (max: {self.peak_balance:.2f})\n"
            f"Perte absolue: {self.peak_balance - self.current_balance:.2f}\n\n"
            f"Ce drawdown est {(drawdown/avg_drawdown):.1f}x plus élevé que la moyenne sur 7 jours."
        )
        
        await notification_manager.notify(
            message=message,
            title=title,
            priority=priority,
            notification_type=NotificationType.ANOMALY,
            details=details
        )
        
        logger.warning(f"Alerte de drawdown envoyée: {drawdown:.2f}% (seuil: {self.drawdown_alert_threshold:.2f}%)")
    
    def _check_order_frequency_alert(self) -> None:
        """
        Vérifie si la fréquence des ordres récents est anormale.
        """
        now = datetime.now()
        
        # Éviter d'envoyer des alertes trop fréquemment (max une par 4 heures)
        time_since_last_alert = now - self.last_order_frequency_alert_time
        if time_since_last_alert.total_seconds() < 14400:  # 4 heures en secondes
            return
            
        # Compter les ordres dans la fenêtre de temps définie
        window_start = now - timedelta(hours=self.order_frequency_window)
        recent_orders = [
            trade for trade in self.trades_history 
            if isinstance(trade["timestamp"], datetime) and trade["timestamp"] >= window_start
        ]
        
        # Calculer le nombre d'ordres par heure
        order_count = len(recent_orders)
        if self.order_frequency_window > 0:
            orders_per_hour = order_count / self.order_frequency_window
        else:
            orders_per_hour = 0
            
        # Vérifier si la fréquence est anormalement élevée
        if orders_per_hour > self.order_frequency_threshold:
            asyncio.create_task(self._send_order_frequency_alert(
                order_count, orders_per_hour
            ))
            self.last_order_frequency_alert_time = now
    
    async def _send_order_frequency_alert(self, order_count: int, orders_per_hour: float) -> None:
        """
        Envoie une alerte de fréquence d'ordres anormale.
        
        Args:
            order_count: Nombre total d'ordres dans la fenêtre
            orders_per_hour: Fréquence d'ordres par heure
        """
        # Analyser les transactions récentes
        recent_symbols = {}
        for trade in self.trades_history[-order_count:]:
            symbol = trade.get("symbol", "Inconnu")
            if symbol in recent_symbols:
                recent_symbols[symbol] += 1
            else:
                recent_symbols[symbol] = 1
                
        # Trier les symboles par nombre d'ordres
        sorted_symbols = sorted(
            [(symbol, count) for symbol, count in recent_symbols.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Créer un résumé des symboles les plus actifs
        symbols_summary = "\n".join([
            f"- {symbol}: {count} ordres ({count/order_count*100:.1f}%)"
            for symbol, count in sorted_symbols[:5]  # Top 5 des symboles
        ])
        
        details = {
            "ordre_total": order_count,
            "ordres_par_heure": f"{orders_per_hour:.2f}",
            "seuil": self.order_frequency_threshold,
            "fenetre_heures": self.order_frequency_window,
            "ratio": f"{orders_per_hour/self.order_frequency_threshold:.1f}x"
        }
        
        message = (
            "⚠️ *Fréquence d'ordres anormale détectée* ⚠️\n\n"
            f"*{order_count}* ordres exécutés sur les dernières *{self.order_frequency_window}* heures\n"
            f"Fréquence: *{orders_per_hour:.2f}* ordres/heure\n"
            f"Seuil normal: {self.order_frequency_threshold} ordres/heure\n\n"
            "*Répartition par symbole:*\n"
            f"{symbols_summary}"
        )
        
        await notification_manager.notify(
            message=message,
            title="Alerte - Fréquence d'ordres anormale",
            priority=NotificationPriority.HIGH,
            notification_type=NotificationType.ANOMALY,
            details=details
        )
        
        logger.warning(f"Alerte de fréquence d'ordres envoyée: {orders_per_hour:.2f}/h (seuil: {self.order_frequency_threshold}/h)")
    
    def _calculate_average_drawdown(self, days: int = 7) -> float:
        """
        Calcule le drawdown moyen sur une période donnée.
        
        Args:
            days: Nombre de jours pour le calcul de la moyenne
            
        Returns:
            Drawdown moyen en pourcentage
        """
        if not self.drawdown_history:
            return 0.0
            
        # Filtrer les données sur la période spécifiée
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_drawdowns = [
            entry["drawdown"] for entry in self.drawdown_history
            if entry["timestamp"] >= cutoff_date
        ]
        
        if not recent_drawdowns:
            return 0.0
            
        return sum(recent_drawdowns) / len(recent_drawdowns)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet des performances du bot.
        
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        if not self.balance_history:
            return {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "profit_loss": 0.0,
                "profit_loss_pct": 0.0
            }
        
        # Calculer diverses métriques
        current_drawdown = 0.0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
            
        max_drawdown = max([entry["drawdown"] for entry in self.drawdown_history]) if self.drawdown_history else 0.0
        
        # Analyser les trades
        total_trades = len(self.trades_history)
        profitable_trades = sum(1 for trade in self.trades_history if trade.get("profit_loss", 0) > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        profit_loss = self.current_balance - self.initial_balance
        profit_loss_pct = (profit_loss / self.initial_balance * 100) if self.initial_balance > 0 else 0.0
        
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        }


# Instance singleton pour faciliter l'utilisation
performance_monitor = PerformanceMonitor()
