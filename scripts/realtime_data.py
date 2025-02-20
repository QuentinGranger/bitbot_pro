import os
import json
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
import websocket
import logging
import threading
import ssl
from dotenv import load_dotenv
from utils.data_cleaning import clean_kline_data, add_technical_features

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignorer les avertissements SSL
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

class RealtimeDataManager:
    """Gestionnaire de données temps réel pour Binance"""
    
    def __init__(self, symbol='btcusdt'):
        """Initialisation du gestionnaire de données temps réel"""
        # Charger les variables d'environnement
        load_dotenv()
        
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Initialiser le client Binance
        self.client = Client(
            api_key=api_key,
            api_secret=api_secret,
            tld='com',
            requests_params={'verify': False}  # Ignorer la vérification SSL
        )
        
        self.symbol = symbol.lower()  # Binance utilise des symboles en minuscules
        self.current_kline = None
        self.last_kline = None
        self.callbacks = []
        self.is_connected = False
        self.ws = None
        self.ws_thread = None
        
        # Buffer pour stocker les données en mémoire
        self.klines_1m = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'open': pd.Series(dtype='float64'),
            'high': pd.Series(dtype='float64'),
            'low': pd.Series(dtype='float64'),
            'close': pd.Series(dtype='float64'),
            'volume': pd.Series(dtype='float64')
        })
        
        self.klines_1h = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'open': pd.Series(dtype='float64'),
            'high': pd.Series(dtype='float64'),
            'low': pd.Series(dtype='float64'),
            'close': pd.Series(dtype='float64'),
            'volume': pd.Series(dtype='float64')
        })
    
    def add_callback(self, callback):
        """Ajouter une fonction de callback pour traiter les données en temps réel"""
        self.callbacks.append(callback)
    
    def clean_and_update_data(self, df: pd.DataFrame, new_data: dict, interval: str) -> pd.DataFrame:
        """Nettoie et met à jour les données"""
        # Ajouter les nouvelles données
        new_df = pd.concat([
            df,
            pd.DataFrame([new_data])
        ], ignore_index=True)
        
        # Nettoyer les données
        clean_df = clean_kline_data(
            new_df,
            remove_outliers=True,
            fill_missing=True,
            max_std_dev=3.0,
            min_volume=0.0
        )
        
        # Ajouter les indicateurs techniques
        clean_df = add_technical_features(clean_df)
        
        # Garder seulement les N dernières bougies
        max_rows = 1000 if interval == '1m' else 168  # 1000 minutes ou 1 semaine pour 1h
        clean_df = clean_df.tail(max_rows)
        
        return clean_df
    
    def on_message(self, ws, message):
        """Callback pour les messages WebSocket"""
        try:
            msg = json.loads(message)
            
            if msg.get('e') != 'kline':
                return
                
            kline = msg['k']
            
            # Créer un dictionnaire avec les données de la bougie
            kline_data = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            
            # Mettre à jour le buffer approprié avec nettoyage
            if kline['i'] == '1m':
                self.klines_1m = self.clean_and_update_data(
                    self.klines_1m,
                    kline_data,
                    '1m'
                )
            elif kline['i'] == '1h':
                self.klines_1h = self.clean_and_update_data(
                    self.klines_1h,
                    kline_data,
                    '1h'
                )
            
            # Appeler les callbacks avec les données nettoyées
            for callback in self.callbacks:
                callback(kline_data)
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {str(e)}")
    
    def on_error(self, ws, error):
        """Callback pour les erreurs WebSocket"""
        logger.error(f"Erreur WebSocket: {str(error)}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Callback pour la fermeture WebSocket"""
        logger.info("WebSocket fermé")
        self.is_connected = False
    
    def on_open(self, ws):
        """Callback pour l'ouverture WebSocket"""
        logger.info("WebSocket ouvert")
        self.is_connected = True
        
        # S'abonner aux streams kline
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [
                f"{self.symbol}@kline_1m",
                f"{self.symbol}@kline_1h"
            ],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        """Démarrer la réception des données"""
        try:
            logger.info(f"Démarrage du stream pour {self.symbol}...")
            
            # Créer la connexion WebSocket
            websocket.enableTrace(False)  # Désactiver les logs de débogage
            self.ws = websocket.WebSocketApp(
                "wss://stream.binance.com:9443/ws",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Démarrer le WebSocket dans un thread séparé
            self.ws_thread = threading.Thread(target=lambda: self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}))
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Attendre que la connexion soit établie
            time.sleep(2)  # Augmenter le délai d'attente
            
            if not self.is_connected:
                raise Exception("Impossible d'établir la connexion WebSocket")
            
            logger.info("Connexion établie avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du stream: {str(e)}")
            self.stop()
    
    def stop(self):
        """Arrêter la réception des données"""
        try:
            if self.ws and self.is_connected:
                logger.info("Arrêt des connexions...")
                self.ws.close()
                self.is_connected = False
                logger.info("Connexions fermées")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt des connexions: {str(e)}")

def print_kline_update(kline_data):
    """Fonction de callback pour afficher les mises à jour des bougies"""
    print(f"Nouvelle bougie: {kline_data}")

def main():
    """Fonction principale"""
    rtm = None
    try:
        # Créer le gestionnaire de données temps réel
        rtm = RealtimeDataManager()
        
        # Ajouter le callback d'affichage
        rtm.add_callback(print_kline_update)
        
        # Démarrer la réception des données
        rtm.start()
        
        logger.info("Appuyez sur Ctrl+C pour quitter")
        
        # Maintenir le programme en vie
        while rtm.is_connected:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
    finally:
        if rtm and rtm.is_connected:
            rtm.stop()
            # Attendre que le thread WebSocket se termine
            if rtm.ws_thread and rtm.ws_thread.is_alive():
                rtm.ws_thread.join(timeout=2)

if __name__ == "__main__":
    main()
