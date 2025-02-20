import os
import sys
import warnings
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# Ignorer les avertissements SSL
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API Binance
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, {"verify": False, "timeout": 20})

def get_latest_data_timestamp(filepath):
    """Récupère le dernier timestamp des données existantes"""
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    if df.empty:
        return None
    
    return pd.to_datetime(df['timestamp'].max())

def fetch_new_klines(symbol, interval, start_timestamp):
    """
    Récupère les nouvelles données depuis le dernier timestamp
    """
    # Convertir le timestamp en millisecondes
    start_ts = int(start_timestamp.timestamp() * 1000)
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

def update_data_file(symbol, interval, filepath):
    """Met à jour un fichier de données spécifique"""
    print(f"\nMise à jour de {symbol} ({interval})...")
    
    # Récupérer le dernier timestamp
    latest_ts = get_latest_data_timestamp(filepath)
    if latest_ts is None:
        print(f"Aucune donnée existante trouvée pour {symbol} ({interval})")
        return False

    # Ajouter un petit décalage pour éviter les doublons
    start_time = latest_ts + timedelta(minutes=1)
    
    # Si la dernière donnée est trop récente, on saute
    if datetime.now() - latest_ts < timedelta(hours=1):
        print(f"Données déjà à jour pour {symbol} ({interval})")
        return False

    # Récupérer les nouvelles données
    new_data = fetch_new_klines(symbol, interval, start_time)
    
    if new_data.empty:
        print(f"Aucune nouvelle donnée pour {symbol} ({interval})")
        return False

    # Charger les données existantes
    existing_data = pd.read_csv(filepath)
    existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])

    # Combiner les données
    updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['timestamp'])
    updated_data = updated_data.sort_values('timestamp')

    # Sauvegarder les données
    updated_data.to_csv(filepath, index=False)
    print(f"Ajout de {len(new_data)} nouvelles entrées pour {symbol} ({interval})")
    return True

def main():
    # Configuration
    symbol = 'BTCUSDT'
    intervals = ['1h', '4h', '1d']
    data_dir = 'data/raw'
    changes_made = False

    # Vérifier que le dossier existe
    if not os.path.exists(data_dir):
        print(f"Erreur: Le dossier {data_dir} n'existe pas")
        sys.exit(1)

    # Mettre à jour chaque fichier
    for interval in intervals:
        # Trouver le fichier correspondant
        files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_{interval}_") and f.endswith('.csv')]
        if not files:
            print(f"Aucun fichier trouvé pour {symbol} ({interval})")
            continue

        filepath = os.path.join(data_dir, files[0])
        if update_data_file(symbol, interval, filepath):
            changes_made = True
            # Mettre à jour DVC
            os.system(f"dvc add {filepath}")
            os.system("dvc push")

    if changes_made:
        # Commit Git si des changements ont été effectués
        os.system('git add data/raw/*.dvc')
        os.system('git commit -m "Update historical data"')
        os.system('git push')
        print("\nDonnées mises à jour et changements poussés vers le dépôt")
    else:
        print("\nAucune mise à jour nécessaire")

if __name__ == "__main__":
    main()
