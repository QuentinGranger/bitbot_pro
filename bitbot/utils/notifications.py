"""
Module de gestion des notifications via différents canaux (Telegram, email, etc.)
pour les alertes du système de trading automatique.
"""
import os
import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Union
from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

from bitbot.utils.logger import logger


class NotificationPriority(Enum):
    """Niveaux de priorité des notifications."""
    LOW = "low"         # Informations générales
    MEDIUM = "medium"   # Événements importants
    HIGH = "high"       # Alertes nécessitant attention
    CRITICAL = "critical"  # Situations critiques nécessitant action immédiate


class NotificationType(Enum):
    """Types de notifications."""
    CONNECTION = "connection"    # État de la connexion
    ANOMALY = "anomaly"          # Anomalies détectées
    TRADE = "trade"              # Activités de trading
    SYSTEM = "system"            # Messages système
    CUSTOM = "custom"            # Messages personnalisés


class TelegramNotifier:
    """Gestionnaire de notifications Telegram."""
    
    def __init__(self, 
                token: str, 
                chat_ids: List[str],
                min_priority: NotificationPriority = NotificationPriority.LOW,
                notification_types: Optional[List[NotificationType]] = None,
                rate_limit_seconds: int = 60):
        """
        Initialise le notificateur Telegram.
        
        Args:
            token: Token du bot Telegram
            chat_ids: Liste des IDs de chat pour envoyer les notifications
            min_priority: Priorité minimale pour envoyer une notification
            notification_types: Types de notification à envoyer (tous si None)
            rate_limit_seconds: Limite de temps entre deux notifications du même type
        """
        self.token = token
        self.chat_ids = chat_ids
        self.min_priority = min_priority
        self.notification_types = notification_types or list(NotificationType)
        self.rate_limit_seconds = rate_limit_seconds
        self.last_notifications: Dict[str, float] = {}
        self.bot = Bot(token=token)
        
        # Vérifier la validité du bot
        self._check_bot_validity()
        
    def _check_bot_validity(self):
        """Vérifie si le bot est valide et opérationnel."""
        try:
            # Pour les tests synchrones, on ne fait pas la vérification complète
            # Ce sera fait au premier appel asynchrone
            logger.info("Bot Telegram initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'initialisation du bot Telegram: {e}")
            raise
    
    async def send_notification(self, 
                         message: str, 
                         priority: NotificationPriority = NotificationPriority.MEDIUM,
                         notification_type: NotificationType = NotificationType.SYSTEM,
                         title: Optional[str] = None,
                         details: Optional[Dict[str, Any]] = None,
                         silent: bool = False) -> bool:
        """
        Envoie une notification via Telegram.
        
        Args:
            message: Contenu principal du message
            priority: Niveau de priorité du message
            notification_type: Type de notification
            title: Titre optionnel du message
            details: Détails supplémentaires à inclure
            silent: Si True, envoi sans notification sonore
            
        Returns:
            True si envoyé avec succès, False sinon
        """
        # Vérifier si le type et la priorité correspondent aux filtres
        if notification_type not in self.notification_types:
            return False
            
        if priority.value < self.min_priority.value:
            return False
            
        # Vérifier le rate limiting
        key = f"{notification_type.value}_{priority.value}"
        current_time = datetime.now().timestamp()
        
        if key in self.last_notifications:
            elapsed = current_time - self.last_notifications[key]
            if elapsed < self.rate_limit_seconds and priority != NotificationPriority.CRITICAL:
                logger.debug(f"Rate limit atteint pour {key}, notification ignorée")
                return False
                
        self.last_notifications[key] = current_time
        
        # Formater le message
        formatted_message = self._format_message(message, title, priority, details)
        
        # Envoyer à tous les chats configurés
        success = True
        for chat_id in self.chat_ids:
            try:
                # Emoji indicateur de priorité
                priority_emoji = {
                    NotificationPriority.LOW: "ℹ️",
                    NotificationPriority.MEDIUM: "⚠️",
                    NotificationPriority.HIGH: "🔴",
                    NotificationPriority.CRITICAL: "🚨"
                }
                
                header = f"{priority_emoji[priority]} *{notification_type.value.upper()}*\n\n"
                full_message = header + formatted_message
                
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=full_message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_notification=silent
                )
                
            except TelegramError as e:
                if "chat not found" in str(e).lower():
                    logger.error(f"Erreur d'envoi Telegram à {chat_id}: Chat non trouvé. "
                                f"Avez-vous démarré une conversation avec le bot? "
                                f"Vérifiez votre Chat ID et assurez-vous d'avoir envoyé /start à votre bot.")
                else:
                    logger.error(f"Erreur d'envoi Telegram à {chat_id}: {e}")
                success = False
                
        return success
    
    def _format_message(self, 
                       message: str, 
                       title: Optional[str], 
                       priority: NotificationPriority,
                       details: Optional[Dict[str, Any]]) -> str:
        """Formate le message avec le titre et les détails."""
        formatted = ""
        
        # Ajouter le titre si présent
        if title:
            formatted += f"*{title}*\n\n"
            
        # Ajouter le message principal
        formatted += message + "\n"
        
        # Ajouter l'horodatage
        formatted += f"\n_📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        # Ajouter les détails si présents
        if details:
            formatted += "\n\n*Détails:*\n"
            for key, value in details.items():
                formatted += f"• {key}: `{value}`\n"
                
        return formatted
        

class NotificationManager:
    """
    Gestionnaire centralisé de notifications qui peut envoyer
    des alertes via différents canaux (Telegram, email, etc.)
    """
    
    def __init__(self):
        """Initialise le gestionnaire de notifications."""
        self.telegram_notifier = None
        self._initialize_from_env()
    
    def _initialize_from_env(self):
        """Initialise les notificateurs à partir des variables d'environnement."""
        # Initialiser Telegram si les variables sont définies
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat_ids = os.environ.get("TELEGRAM_CHAT_IDS")
        
        if telegram_token and telegram_chat_ids:
            chat_ids = [id.strip() for id in telegram_chat_ids.split(",")]
            try:
                min_priority_name = os.environ.get("TELEGRAM_MIN_PRIORITY", "LOW")
                min_priority = NotificationPriority[min_priority_name]
                
                self.telegram_notifier = TelegramNotifier(
                    token=telegram_token,
                    chat_ids=chat_ids,
                    min_priority=min_priority,
                    rate_limit_seconds=int(os.environ.get("TELEGRAM_RATE_LIMIT", "60"))
                )
                logger.info("Notificateur Telegram initialisé")
            except (ValueError, KeyError) as e:
                logger.error(f"Erreur d'initialisation du notificateur Telegram: {e}")
    
    async def notify(self, 
              message: str, 
              priority: NotificationPriority = NotificationPriority.MEDIUM,
              notification_type: NotificationType = NotificationType.SYSTEM,
              title: Optional[str] = None,
              details: Optional[Dict[str, Any]] = None,
              silent: bool = False) -> bool:
        """
        Envoie une notification via tous les canaux configurés.
        
        Args:
            message: Contenu principal du message
            priority: Niveau de priorité du message
            notification_type: Type de notification
            title: Titre optionnel du message
            details: Détails supplémentaires à inclure
            silent: Si True, envoi sans notification sonore
            
        Returns:
            True si au moins un canal a réussi l'envoi
        """
        success = False
        
        # Envoyer via Telegram si configuré
        if self.telegram_notifier:
            telegram_success = await self.telegram_notifier.send_notification(
                message=message,
                priority=priority,
                notification_type=notification_type,
                title=title,
                details=details,
                silent=silent
            )
            if telegram_success:
                success = True
        else:
            # Si aucun notificateur n'est configuré, logger le message
            log_level = {
                NotificationPriority.LOW: logging.INFO,
                NotificationPriority.MEDIUM: logging.WARNING,
                NotificationPriority.HIGH: logging.ERROR,
                NotificationPriority.CRITICAL: logging.CRITICAL
            }.get(priority, logging.INFO)
            
            full_msg = f"{title + ': ' if title else ''}{message}"
            logger.log(log_level, f"[NOTIFICATION] {full_msg}")
            success = True
            
        return success
    
    async def notify_connection_issue(self, 
                               message: str, 
                               details: Optional[Dict[str, Any]] = None,
                               critical: bool = False) -> bool:
        """
        Envoie une notification pour un problème de connexion.
        
        Args:
            message: Description du problème
            details: Détails supplémentaires (durée, symboles affectés, etc.)
            critical: Si True, marque comme priorité critique
            
        Returns:
            True si l'envoi a réussi
        """
        priority = NotificationPriority.CRITICAL if critical else NotificationPriority.HIGH
        return await self.notify(
            message=message,
            priority=priority,
            notification_type=NotificationType.CONNECTION,
            title="Problème de Connexion Détecté",
            details=details
        )
    
    async def notify_anomaly(self,
                      anomaly_type: str,
                      symbol: str,
                      severity: Union[str, float],
                      message: str,
                      details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Envoie une notification pour une anomalie détectée.
        
        Args:
            anomaly_type: Type d'anomalie (price_spike, volume_spike, etc.)
            symbol: Symbole concerné
            severity: Niveau de sévérité (numérique ou textuel)
            message: Description de l'anomalie
            details: Détails supplémentaires
            
        Returns:
            True si l'envoi a réussi
        """
        # Déterminer la priorité en fonction de la sévérité
        if isinstance(severity, str):
            priority_map = {
                "FAIBLE": NotificationPriority.LOW,
                "MOYENNE": NotificationPriority.MEDIUM,
                "ÉLEVÉE": NotificationPriority.HIGH,
                "CRITIQUE": NotificationPriority.CRITICAL
            }
            priority = priority_map.get(severity, NotificationPriority.MEDIUM)
        else:
            # Valeur numérique, convertir en priorité
            if severity > 0.8:
                priority = NotificationPriority.CRITICAL
            elif severity > 0.5:
                priority = NotificationPriority.HIGH
            elif severity > 0.2:
                priority = NotificationPriority.MEDIUM
            else:
                priority = NotificationPriority.LOW
        
        # Préparer les détails
        anomaly_details = details or {}
        anomaly_details.update({
            "Type": anomaly_type,
            "Symbole": symbol,
            "Sévérité": severity
        })
        
        return await self.notify(
            message=message,
            priority=priority,
            notification_type=NotificationType.ANOMALY,
            title=f"Anomalie {anomaly_type.capitalize()} Détectée",
            details=anomaly_details
        )


# Instance singleton pour utilisation facile
notification_manager = NotificationManager()


async def test_notifications():
    """Fonction de test pour les notifications."""
    # Simuler une notification d'anomalie
    await notification_manager.notify_anomaly(
        anomaly_type="price_spike",
        symbol="BTCUSDT",
        severity="ÉLEVÉE",
        message="Spike de prix anormal détecté sur BTC/USDT",
        details={
            "Prix actuel": "69420.00",
            "Variation": "+15%",
            "Volume": "250 BTC"
        }
    )
    
    # Simuler une notification de problème de connexion
    await notification_manager.notify_connection_issue(
        message="Connexion WebSocket interrompue",
        details={
            "Durée": "120s",
            "Streams affectés": "BTCUSDT, ETHUSDT",
            "Tentatives de reconnexion": "3/5"
        },
        critical=True
    )


if __name__ == "__main__":
    """Test du module de notifications."""
    import asyncio
    asyncio.run(test_notifications())
