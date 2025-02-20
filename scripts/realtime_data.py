import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class RealtimeDataManager:
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.current_kline = None
        self.last_kline = None
        self.callbacks = []
        
        # Buffer pour stocker les données en mémoire
        self.klines_1m = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.klines_1h = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
    def add_callback(self, callback):
        """Ajouter une fonction de callback pour traiter les données en temps réel"""
        self.callbacks.append(callback)
        
    def process_kline_data(self, data):
        """Traiter les données de kline reçues"""
        k = data['k']
        
        # Créer un dictionnaire avec les données
        kline_data = {
            'timestamp': pd.to_datetime(k['t'], unit='ms'),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v']),
            'is_closed': k['x']
        }
        
        # Mettre à jour les données actuelles
        self.current_kline = kline_data
        
        # Si la bougie est fermée, mettre à jour le buffer
        if kline_data['is_closed']:
            self.last_kline = kline_data
            new_row = pd.DataFrame([kline_data])
            self.klines_1m = pd.concat([self.klines_1m, new_row]).tail(1000)  # Garder les 1000 dernières minutes
            
            # Si c'est une nouvelle heure, mettre à jour les données horaires
            if kline_data['timestamp'].minute == 0:
                self.klines_1h = pd.concat([self.klines_1h, new_row]).tail(168)  # Garder la dernière semaine
        
        # Exécuter tous les callbacks enregistrés
        for callback in self.callbacks:
            callback(kline_data)
    
    async def start_kline_socket(self):
        """Démarrer le socket pour les klines"""
        client = await AsyncClient.create(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        
        bm = BinanceSocketManager(client)
        # Écouter les klines 1 minute
        kline_stream = bm.kline_socket(symbol=self.symbol, interval='1m')
        
        async with kline_stream as stream:
            while True:
                msg = await stream.recv()
                if msg['e'] == 'kline':
                    self.process_kline_data(msg)

def print_kline_update(kline_data):
    """Exemple de callback pour afficher les mises à jour"""
    print(f"\rDernière mise à jour {kline_data['timestamp']}: {kline_data['close']}$ | Vol: {kline_data['volume']:.2f}", end='')

async def main():
    # Créer le gestionnaire de données
    manager = RealtimeDataManager()
    
    # Ajouter le callback d'affichage
    manager.add_callback(print_kline_update)
    
    print(f"Démarrage du flux de données en temps réel pour {manager.symbol}...")
    await manager.start_kline_socket()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nArrêt du flux de données...")
