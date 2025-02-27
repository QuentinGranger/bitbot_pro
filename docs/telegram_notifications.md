# Notifications Telegram pour BitBotPro

Ce document explique comment configurer et utiliser les notifications Telegram avec BitBotPro pour recevoir des alertes en temps réel sur les déconnexions et les anomalies détectées.

## Prérequis

- Un compte Telegram
- Accès à internet pour le bot
- Variables d'environnement configurées (voir `.env-example`)

## Configuration du Bot Telegram

1. **Créer un nouveau bot Telegram**

   Ouvrez Telegram et recherchez "BotFather" (@BotFather).
   
   Envoyez la commande `/newbot` et suivez les instructions :
   - Donnez un nom à votre bot (ex: "BitBotPro Alerts")
   - Choisissez un nom d'utilisateur pour votre bot (doit se terminer par "bot", ex: "bitbotpro_alerts_bot")
   
   BotFather vous fournira un token d'API. Conservez-le précieusement, il sera nécessaire pour configurer les notifications.

2. **Obtenir votre Chat ID**

   Pour recevoir des notifications, vous devez connaître votre Chat ID.
   
   Méthode 1 : Utilisez @RawDataBot
   - Recherchez "@RawDataBot" dans Telegram et démarrez une conversation
   - Le bot vous enverra un message contenant toutes vos informations, incluant votre Chat ID
   
   Méthode 2 : Utilisez @userinfobot
   - Recherchez "@userinfobot" dans Telegram et démarrez une conversation
   - Le bot vous enverra directement votre Chat ID
   
   Si vous souhaitez envoyer des notifications à un groupe :
   - Ajoutez votre bot au groupe
   - Envoyez un message dans le groupe
   - Visitez l'URL : `https://api.telegram.org/bot<VOTRE_TOKEN>/getUpdates`
   - Recherchez l'ID du chat de groupe dans la réponse JSON (commencera généralement par "-")

3. **Configuration des variables d'environnement**

   Ajoutez les variables suivantes à votre fichier `.env` :
   
   ```
   TELEGRAM_BOT_TOKEN=<votre_token>
   TELEGRAM_CHAT_IDS=<chat_id1>,<chat_id2>  # Vous pouvez spécifier plusieurs IDs séparés par des virgules
   TELEGRAM_MIN_PRIORITY=MEDIUM  # LOW, MEDIUM, HIGH, CRITICAL
   TELEGRAM_RATE_LIMIT=60  # Temps minimum entre notifications similaires (secondes)
   ```

## Types de notifications

BitBotPro envoie les types de notifications suivants via Telegram :

### 1. Notifications de connexion

- **Déconnexion WebSocket** : Envoyée lorsque la connexion aux flux de données est perdue
- **Tentatives de reconnexion** : Informations sur les tentatives en cours
- **Reconnexion réussie** : Confirmation que la connexion a été rétablie
- **Échec de reconnexion** : Alerte critique si toutes les tentatives de reconnexion échouent

### 2. Notifications d'anomalies

- **Spikes de volume** : Détection de volumes de trading anormalement élevés
- **Spikes de prix** : Détection de variations de prix inhabituelles
- **Gaps de données** : Détection de périodes sans données
- **Spread négatif** : Détection d'une situation anormale où le prix d'achat est supérieur au prix de vente

## Niveaux de priorité

Les notifications sont envoyées avec différents niveaux de priorité :

- **LOW** (🔵) : Informations générales, non critiques
- **MEDIUM** (⚠️) : Événements importants à surveiller
- **HIGH** (🔴) : Alertes nécessitant attention
- **CRITICAL** (🚨) : Situations critiques nécessitant une action immédiate

Pour limiter le nombre de notifications, vous pouvez définir un niveau de priorité minimum avec `TELEGRAM_MIN_PRIORITY`.

## Test des notifications

Pour tester votre configuration Telegram, exécutez le script de test :

```bash
python scripts/test_telegram_notifications.py --token <votre_token> --chat-ids <votre_chat_id>
```

Ou si vos variables d'environnement sont déjà configurées :

```bash
python scripts/test_telegram_notifications.py --all
```

Options disponibles :
- `--connection` : Teste uniquement les notifications de connexion
- `--anomaly` : Teste uniquement les notifications d'anomalies
- `--all` : Teste tous les types de notifications

## Utilisation dans le code

Pour envoyer une notification depuis votre propre code :

```python
from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType

# Notification simple
await notification_manager.notify(
    message="Message important",
    priority=NotificationPriority.MEDIUM,
    notification_type=NotificationType.SYSTEM,
    title="Titre du message",
    details={"clé": "valeur", "autre_clé": "autre_valeur"}
)

# Notification d'anomalie
await notification_manager.notify_anomaly(
    anomaly_type="volume_spike",
    symbol="BTCUSDT",
    severity="ÉLEVÉE",
    message="Spike de volume détecté",
    details={"Volume": "1245.67", "Variation": "+200%"}
)

# Notification de problème de connexion
await notification_manager.notify_connection_issue(
    message="Connexion perdue",
    details={"Durée": "120s", "Raison": "Timeout"},
    critical=True  # Pour les problèmes urgents
)
```

## Dépannage

1. **Vous ne recevez pas de notifications**
   - Vérifiez que le token est valide
   - Assurez-vous d'avoir démarré une conversation avec votre bot
   - Vérifiez que le Chat ID est correct
   - Vérifiez les logs pour voir si des erreurs sont signalées

2. **Trop de notifications**
   - Augmentez la valeur de `TELEGRAM_MIN_PRIORITY` (MEDIUM ou HIGH)
   - Augmentez la valeur de `TELEGRAM_RATE_LIMIT` pour éviter les doublons

3. **Le bot ne répond pas**
   - Vérifiez la connectivité internet
   - Vérifiez si le bot a été désactivé par BotFather
