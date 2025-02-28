"""
Module de gestion des alertes Telegram pour BitBotPro.
Ce module utilise un singleton pour faciliter l'envoi d'alertes depuis n'importe où dans le code.
"""
import os
import logging
import asyncio
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from dotenv import load_dotenv

# Charger les variables d'environnement
# Chercher le fichier .env dans le répertoire parent du projet
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Variables d'environnement chargées depuis {env_path}")
else:
    print(f"Fichier .env non trouvé à {env_path}")

# Configuration du logger
logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Niveaux de priorité des alertes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types d'alertes."""
    CONNECTION = "connection"  # Problèmes de connexion
    PRICE = "price"            # Alertes de prix
    VOLUME = "volume"          # Alertes de volume
    TRADE = "trade"            # Alertes de trading
    SYSTEM = "system"          # Alertes système
    ANOMALY = "anomaly"        # Anomalies détectées

class TelegramAlertManager:
    """
    Gestionnaire d'alertes Telegram implémenté comme un singleton pour
    faciliter l'accès depuis n'importe quel module.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelegramAlertManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialise le gestionnaire d'alertes Telegram."""
        if self._initialized:
            return
            
        self._initialized = True
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS", "")
        self.chat_ids = [id.strip() for id in chat_ids_str.split(",") if id.strip()]
        
        self.is_configured = bool(self.token and self.chat_ids)
        self.bot = None
        self.min_priority = AlertPriority.LOW
        self.last_alerts = {}
        self.rate_limit = int(os.environ.get("TELEGRAM_RATE_LIMIT", 60))
        
        logger.info(f"TelegramAlertManager: token configuré: {bool(self.token)}, chat IDs configurés: {len(self.chat_ids)}")
        
        # Initialiser le bot si la configuration est valide
        if self.is_configured:
            self.bot = Bot(token=self.token)
            logger.info(f"Alertes Telegram configurées pour {len(self.chat_ids)} destinataires avec le token {self.token[:5]}...{self.token[-5:]}")
        else:
            logger.warning(f"Alertes Telegram non configurées: token présent: {bool(self.token)}, chat_ids présents: {bool(self.chat_ids)}")
            
    async def send_alert(self, 
                    message: str, 
                    alert_type: AlertType = AlertType.SYSTEM,
                    priority: AlertPriority = AlertPriority.MEDIUM,
                    title: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None,
                    silent: bool = False) -> bool:
        """
        Envoie une alerte via Telegram.
        
        Args:
            message: Contenu principal du message
            alert_type: Type d'alerte
            priority: Niveau de priorité
            title: Titre optionnel
            details: Détails supplémentaires au format clé-valeur
            silent: Si True, envoi sans notification sonore
            
        Returns:
            True si envoyé avec succès
        """
        logger.info(f"Tentative d'envoi d'alerte Telegram: {alert_type.value} - {priority.value}")
        
        if not self.is_configured or not self.bot:
            logger.warning(f"Alerte {alert_type.value} non envoyée: Telegram non configuré")
            return False
            
        # Vérifier le rate limiting
        key = f"{alert_type.value}_{priority.value}"
        current_time = datetime.now().timestamp()
        
        if key in self.last_alerts:
            elapsed = current_time - self.last_alerts[key]
            if elapsed < self.rate_limit and priority != AlertPriority.CRITICAL:
                logger.debug(f"Rate limit atteint pour {key}, alerte ignorée")
                return False
        
        self.last_alerts[key] = current_time
        
        # Préparer le message
        formatted = self._format_message(message, title, priority, details)
        
        # Emoji par priorité
        priority_emojis = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🔴",
            AlertPriority.CRITICAL: "🚨"
        }
        
        # Échapper les caractères spéciaux dans le type d'alerte
        safe_type = alert_type.value.upper().replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")
        
        header = f"{priority_emojis[priority]} *{safe_type}*\n\n"
        full_message = header + formatted
        
        # Utiliser le mode HTML pour éviter les problèmes de formatage Markdown
        try:
            # Envoyer à tous les destinataires
            success = True
            for chat_id in self.chat_ids:
                try:
                    logger.info(f"Envoi d'alerte Telegram à chat_id {chat_id} en cours...")
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=full_message,
                        parse_mode=ParseMode.MARKDOWN_V2,
                        disable_notification=silent
                    )
                    logger.info(f"✅ Alerte envoyée avec succès à {chat_id}")
                except Exception as e:
                    # Si erreur avec MARKDOWN_V2, essayer sans formatage
                    if "parse" in str(e).lower() or "entity" in str(e).lower():
                        logger.warning(f"Échec du formatage Markdown, envoi sans formatage...")
                        try:
                            # Préparer un message sans formatage
                            plain_header = f"{priority_emojis[priority]} {alert_type.value.upper()}\n\n"
                            plain_message = message + "\n\n"
                            
                            if title:
                                plain_header += f"{title}\n\n"
                                
                            plain_details = ""
                            if details:
                                plain_details += "\nDétails:\n"
                                for key, value in details.items():
                                    plain_details += f"• {key}: {value}\n"
                                    
                            await self.bot.send_message(
                                chat_id=chat_id,
                                text=plain_header + plain_message + plain_details,
                                parse_mode=None,
                                disable_notification=silent
                            )
                            logger.info(f"✅ Alerte envoyée sans formatage avec succès à {chat_id}")
                        except Exception as inner_e:
                            logger.error(f"❌ Échec de l'envoi sans formatage: {type(inner_e).__name__}: {inner_e}")
                            success = False
                    else:
                        logger.error(f"❌ Erreur d'envoi d'alerte à {chat_id}: {type(e).__name__}: {e}")
                        success = False
                    
            return success
        except Exception as e:
            logger.error(f"❌ Erreur générale lors de l'envoi des alertes: {type(e).__name__}: {e}")
            return False
        
    def _format_message(self, 
                      message: str, 
                      title: Optional[str], 
                      priority: AlertPriority,
                      details: Optional[Dict[str, Any]]) -> str:
        """Formate le message avec titre et détails pour Markdown V2."""
        # Caractères à échapper pour Markdown V2: _ * [ ] ( ) ~ ` > # + - = | { } . !
        def escape_markdown(text):
            if not text:
                return ""
            text = str(text)
            for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                text = text.replace(char, f"\\{char}")
            return text
        
        formatted = ""
        
        # Ajouter le titre si présent
        if title:
            formatted += f"*{escape_markdown(title)}*\n\n"
            
        # Ajouter le message principal
        formatted += f"{escape_markdown(message)}\n"
        
        # Ajouter les détails si présents
        if details:
            formatted += "\n*Détails:*\n"
            for key, value in details.items():
                formatted += f"• {escape_markdown(key)}: `{escape_markdown(value)}`\n"
                
        return formatted

# Créer l'instance singleton
telegram_alerts = TelegramAlertManager()

# Fonction d'aide pour simplifier l'envoi d'alertes
async def send_telegram_alert(message: str, 
                        alert_type: AlertType = AlertType.SYSTEM,
                        priority: AlertPriority = AlertPriority.MEDIUM,
                        title: Optional[str] = None,
                        details: Optional[Dict[str, Any]] = None,
                        silent: bool = False) -> bool:
    """
    Fonction pratique pour envoyer une alerte Telegram.
    Utilise le singleton TelegramAlertManager.
    """
    return await telegram_alerts.send_alert(
        message=message,
        alert_type=alert_type,
        priority=priority,
        title=title,
        details=details,
        silent=silent
    )
