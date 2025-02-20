import os
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API Binance
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

def fetch_historical_klines(symbol, interval, start_date, end_date=None):
    """
    Récupère les données historiques de Binance
    
    Args:
        symbol (str): Paire de trading (ex: 'BTCUSDT')
        interval (str): Intervalle de temps (ex: '1h', '1d')
        start_date (str): Date de début (format: 'YYYY-MM-DD')
        end_date (str): Date de fin (format: 'YYYY-MM-DD')
    
    Returns:
        pd.DataFrame: DataFrame contenant les données OHLCV
    """
    # Convertir les dates en timestamp
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    if end_date:
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)

    # Récupérer les données
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_ts,
        end_ts
    )

    # Convertir en DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Nettoyer le DataFrame
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df

def save_to_csv(df, symbol, interval, output_dir):
    """Sauvegarde les données dans un fichier CSV"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol}_{interval}_{df['timestamp'].min().strftime('%Y%m%d')}_{df['timestamp'].max().strftime('%Y%m%d')}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath

def main():
    # Configuration
    symbol = 'BTCUSDT'
    intervals = ['1h', '4h', '1d']
    start_date = '2023-01-01'  # Début 2023
    output_dir = 'data/raw'

    for interval in intervals:
        print(f"Récupération des données {symbol} pour l'intervalle {interval}...")
        df = fetch_historical_klines(symbol, interval, start_date)
        filepath = save_to_csv(df, symbol, interval, output_dir)
        print(f"Données sauvegardées dans {filepath}")
        
        # Ajouter au suivi DVC
        os.system(f"dvc add {filepath}")

if __name__ == "__main__":
    main()
