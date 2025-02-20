import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import warnings
import time
import requests
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from sqlalchemy import create_engine, text, Table, Column, Integer, Float, String, MetaData, TIMESTAMP, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert
from utils.data_cleaning import clean_kline_data, add_technical_features

# Configuration du logging
import logging
from logging.handlers import RotatingFileHandler

# Créer le dossier logs s'il n'existe pas
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configuration du logger
logger = logging.getLogger('BitBot')
logger.setLevel(logging.INFO)

# Handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Handler pour le fichier avec rotation (5 fichiers de 50Mo max)
file_handler = RotatingFileHandler('logs/fetch_historical.log', maxBytes=50*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Supprimer le handler par défaut de logging
logger.propagate = False

# Ignorer les avertissements SSL
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def get_db_engine():
    """Créer la connexion à PostgreSQL"""
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'bitbot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    return create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

def show_database_stats(engine):
    """Affiche les statistiques de la base de données"""
    query = text("""
            SELECT interval, 
                   COUNT(*) as count,
                   MIN(open_time) as min_date,
                   MAX(open_time) as max_date
            FROM klines
            GROUP BY interval
        """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            if rows:
                logger.info("\nStatistiques de la base de données:")
                logger.info("--------------------------------")
                for row in rows:
                    logger.info(f"Intervalle {row.interval}:")
                    logger.info(f"  - Nombre de lignes: {row.count:,}")
                    logger.info(f"  - Période: du {row.min_date} au {row.max_date}")
                logger.info("--------------------------------")
            else:
                logger.info("Aucune donnée dans la base")
                
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage des statistiques: {str(e)}")

def create_tables(engine):
    """Créer les tables nécessaires"""
    try:
        metadata = MetaData()
        
        # Définition de la table
        klines = Table('klines', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String),
            Column('interval', String),
            Column('open_time', TIMESTAMP),
            Column('open', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('volume', Float),
            Column('close_time', TIMESTAMP),
            Column('quote_asset_volume', Float),
            Column('number_of_trades', Integer),
            Column('taker_buy_base_asset_volume', Float),
            Column('taker_buy_quote_asset_volume', Float),
            Column('ignore', Float),
            # Nouveaux champs pour les indicateurs techniques
            Column('volatility', Float, nullable=True),
            Column('volume_ma', Float, nullable=True),
            Column('return_1', Float, nullable=True),
            Column('return_5', Float, nullable=True),
            Column('return_15', Float, nullable=True),
            Column('return_30', Float, nullable=True),
            Column('return_60', Float, nullable=True),
            
            # Ajouter une contrainte unique
            UniqueConstraint('symbol', 'interval', 'open_time', name='uix_klines_symbol_interval_time')
        )
        
        # Créer la table
        with engine.begin() as connection:
            metadata.create_all(connection)
            
        logger.info("✓ Table klines créée ou mise à jour avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de la table: {str(e)}")
        raise

def insert_klines(df, symbol, interval):
    """Insère les données dans la base PostgreSQL"""
    try:
        engine = get_db_engine()
        
        # Préparer les données pour l'insertion
        data = []
        for _, row in df.iterrows():
            record = {
                'symbol': symbol,
                'interval': interval,
                'open_time': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'close_time': row['timestamp'],
                'quote_asset_volume': row['volume'],
                'number_of_trades': 0,  # Cette information n'est pas disponible dans les données nettoyées
                'taker_buy_base_asset_volume': 0,
                'taker_buy_quote_asset_volume': 0,
                'ignore': 0,
                'volatility': row.get('volatility'),
                'volume_ma': row.get('volume_ma'),
                'return_1': row.get('return_1'),
                'return_5': row.get('return_5'),
                'return_15': row.get('return_15'),
                'return_30': row.get('return_30'),
                'return_60': row.get('return_60')
            }
            data.append(record)
        
        # Insérer les données
        with engine.begin() as connection:
            table = Table('klines', MetaData(), autoload_with=engine)
            stmt = insert(table).values(data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'interval', 'open_time'],
                set_=dict(
                    open=stmt.excluded.open,
                    high=stmt.excluded.high,
                    low=stmt.excluded.low,
                    close=stmt.excluded.close,
                    volume=stmt.excluded.volume,
                    quote_asset_volume=stmt.excluded.quote_asset_volume,
                    number_of_trades=stmt.excluded.number_of_trades,
                    taker_buy_base_asset_volume=stmt.excluded.taker_buy_base_asset_volume,
                    taker_buy_quote_asset_volume=stmt.excluded.taker_buy_quote_asset_volume,
                    ignore=stmt.excluded.ignore,
                    volatility=stmt.excluded.volatility,
                    volume_ma=stmt.excluded.volume_ma,
                    return_1=stmt.excluded.return_1,
                    return_5=stmt.excluded.return_5,
                    return_15=stmt.excluded.return_15,
                    return_30=stmt.excluded.return_30,
                    return_60=stmt.excluded.return_60
                )
            )
            connection.execute(stmt)
        
        logger.info(f"✓ {len(data)} lignes insérées/mises à jour dans la base de données")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion des données: {str(e)}")
        raise

def get_chunks(client, symbol, interval, start_date, end_date, chunk_size, engine):
    """Générer les chunks de données à récupérer"""
    logger.info(f"Génération des chunks pour {symbol} {interval}")
    logger.info(f"Période: du {start_date.strftime('%Y-%m-%d %H:%M')} au {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    chunks = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        chunks.append({
            'symbol': symbol,
            'interval': interval,
            'start_date': current_start,
            'end_date': current_end,
            'engine': engine
        })
        current_start = current_end
    
    # Log uniquement le premier et dernier chunk
    logger.info(f"Premier chunk: {chunks[0]['start_date'].strftime('%Y-%m-%d %H:%M')} -> {chunks[0]['end_date'].strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Dernier chunk: {chunks[-1]['start_date'].strftime('%Y-%m-%d %H:%M')} -> {chunks[-1]['end_date'].strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Nombre total de chunks: {len(chunks)}")
    
    return chunks

def get_historical_klines(symbol, interval, start_date, end_date):
    """Récupérer les données historiques directement via l'API HTTP"""
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000,
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(end_date.timestamp() * 1000)
    }
    
    try:
        logger.info(f"Requête API pour {symbol} de {start_date.strftime('%Y-%m-%d %H:%M')} à {end_date.strftime('%Y-%m-%d %H:%M')}")
        start_time = time.time()
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        klines = response.json()
        elapsed_time = time.time() - start_time
        
        if klines:
            logger.info(f"✓ Réponse reçue en {elapsed_time:.1f}s - {len(klines)} klines")
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convertir les timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convertir les colonnes numériques
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_asset_volume', 'taker_buy_base_asset_volume',
                             'taker_buy_quote_asset_volume', 'ignore']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['number_of_trades'] = df['number_of_trades'].astype(int)
                
            # Nettoyer les données
            df = clean_kline_data(
                df,
                remove_outliers=True,
                fill_missing=True,
                max_std_dev=3.0,
                min_volume=0.0
            )
            
            # Ajouter les indicateurs techniques
            df = add_technical_features(df)
                
            return df
        else:
            logger.warning(f"❌ Aucune donnée reçue pour cette période")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"⚠️ Timeout après {time.time() - start_time:.1f}s")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"⚠️ Erreur API: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"⚠️ Erreur inattendue: {str(e)}")
        return None

def process_chunk(chunk, db_lock):
    """Traiter un chunk de données"""
    try:
        # Éviter de logger chaque chunk pour garder la console propre
        df = get_historical_klines(
            chunk['symbol'],
            chunk['interval'],
            chunk['start_date'],
            chunk['end_date']
        )
        
        if df is not None and not df.empty:
            with db_lock:
                insert_klines(df, chunk['symbol'], chunk['interval'])
            return df
        return None
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données: {str(e)}", exc_info=True)
        raise

def process_chunks_with_progress(chunks, symbol, interval, executor, db_lock):
    """Traiter les chunks avec une barre de progression"""
    total_chunks = len(chunks)
    completed_chunks = 0
    successful_chunks = 0
    total_rows = 0
    start_time = time.time()
    
    # Créer la barre de progression avec description plus claire
    with tqdm(total=total_chunks, 
             desc=f"Téléchargement {symbol}", 
             unit='chunks',
             position=0, 
             leave=True) as pbar:
        
        # Soumettre les tâches par groupes de 3 pour éviter de surcharger l'API
        for i in range(0, len(chunks), 3):
            chunk_group = chunks[i:i+3]
            futures = [executor.submit(process_chunk, chunk, db_lock) for chunk in chunk_group]
            
            # Traiter les résultats au fur et à mesure
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        successful_chunks += 1
                        total_rows += len(df)
                        
                    completed_chunks += 1
                    
                    # Calculer la vitesse et le temps restant estimé
                    elapsed = time.time() - start_time
                    speed = completed_chunks / elapsed if elapsed > 0 else 0
                    eta = (total_chunks - completed_chunks) / speed if speed > 0 else 0
                    
                    # Mettre à jour la barre de progression avec plus d'informations
                    pbar.set_description(
                        f"Téléchargement {symbol} ({successful_chunks}/{completed_chunks} réussis, "
                        f"{total_rows:,} lignes, {speed:.1f} chunks/s, ETA: {int(eta/60)}m {int(eta%60)}s)"
                    )
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement d'un chunk: {str(e)}")
                    completed_chunks += 1
                    pbar.update(1)
            
            # Pause entre les groupes de requêtes
            if i + 3 < len(chunks):
                time.sleep(2)
    
    return successful_chunks, total_rows

def main():
    """Fonction principale"""
    logger.info("Démarrage de la récupération des données historiques")
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Créer la connexion à PostgreSQL
    engine = get_db_engine()
    logger.info("Connexion à la base de données établie")
    
    # Créer les tables
    create_tables(engine)
    logger.info("Tables créées/vérifiées avec succès")
    
    # Configuration
    symbols = ['BTCUSDT']
    intervals = ['1m']
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    # Depuis le lancement de Binance
    start_date = datetime(2017, 7, 14)  # Date de lancement de Binance
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    chunk_size = timedelta(days=7)  # 7 jours par chunk
    max_workers = 3  # 3 workers en parallèle
    
    logger.info(f"Configuration de la récupération:")
    logger.info(f"- Symbole: {symbols[0]}")
    logger.info(f"- Intervalle: {intervals[0]}")
    logger.info(f"- Période: du {start_date.strftime('%Y-%m-%d %H:%M')} au {end_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"- Taille des chunks: {chunk_size}")
    logger.info(f"- Nombre de workers: {max_workers}")
    
    # Verrou global pour les opérations sur la base de données
    db_lock = Lock()
    
    # Barre de progression globale
    total_pairs = len(symbols) * len(intervals)
    with tqdm(total=total_pairs, desc="Progrès total", position=1, leave=True) as total_pbar:
        # Traiter chaque symbole et intervalle
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for symbol in symbols:
                for interval in intervals:
                    # Générer les chunks
                    chunks = get_chunks(None, symbol, interval, start_date, end_date, chunk_size, engine)
                    
                    # Traiter les chunks avec barre de progression
                    successful_chunks, total_rows = process_chunks_with_progress(chunks, symbol, interval, executor, db_lock)
                    
                    # Afficher le résumé
                    logger.info(f"\nRésumé pour {symbol} {interval}:")
                    logger.info(f"- Chunks réussis: {successful_chunks}/{len(chunks)}")
                    logger.info(f"- Total lignes: {total_rows}")
                    
                    # Mettre à jour la barre de progression globale
                    total_pbar.update(1)
    
    logger.info("\nRécupération des données terminée")
    show_database_stats(engine)
    logger.info("\nTerminé!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Erreur critique: {str(e)}", exc_info=True)
        raise
