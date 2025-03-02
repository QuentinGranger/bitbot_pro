#!/usr/bin/env python3
"""
Test direct des notifications Telegram sans passer par notre classe de notification.
"""
import os
import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv

async def test_telegram_direct():
    """Envoie un message directement via l'API Telegram."""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer les variables d'environnement
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS")
    
    if not token or not chat_ids_str:
        print("❌ Erreur: TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_IDS non défini")
        print(f"TELEGRAM_BOT_TOKEN: {'Configuré' if token else 'Non configuré'}")
        print(f"TELEGRAM_CHAT_IDS: {'Configuré' if chat_ids_str else 'Non configuré'}")
        return
    
    chat_ids = [id.strip() for id in chat_ids_str.split(",")]
    print(f"✅ Configuration chargée: {len(chat_ids)} chat IDs trouvés")
    
    # Créer le bot
    bot = Bot(token=token)
    
    # Tenter d'obtenir les informations du bot pour vérifier si le token est valide
    try:
        bot_info = await bot.get_me()
        print(f"✅ Bot configuré correctement - Nom: {bot_info.first_name}, Username: @{bot_info.username}")
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du bot: {e}")
        return
    
    # Essayer d'envoyer un message à chaque chat ID
    for chat_id in chat_ids:
        try:
            message = (
                "*🚨 Test Direct de Notification Telegram*\n\n"
                "Ceci est un message de test direct via l'API Telegram.\n\n"
                "*Détails:*\n"
                "• Timestamp: Maintenant\n"
                "• Type: Test direct\n"
                "• Importance: Haute\n"
            )
            
            print(f"Envoi du message au chat ID: {chat_id}...")
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            print(f"✅ Message envoyé avec succès à {chat_id}")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi au chat ID {chat_id}: {e}")

if __name__ == "__main__":
    asyncio.run(test_telegram_direct())
