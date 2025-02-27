"""
Client Binance pour la récupération de données historiques.
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
from pathlib import Path
import gzip
import json
import ssl
import pytz
from concurrent.futures import ThreadPoolExecutor

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger

class BinanceClient:
    """Client optimisé pour la récupération de données historiques Binance."""
    
    def __init__(self, verify_ssl: bool = False, data_dir: Union[str, Path] = './data'):
        """
        Args:
            verify_ssl: Si True, vérifie les certificats SSL
            data_dir: Répertoire de données
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limiter = RateLimiter(1200, 60)  # 1200 requêtes par minute
        self.session = None
        self.verify_ssl = verify_ssl
        self.data_dir = Path(data_dir)
        
        # Intervalles disponibles
        self.intervals = {
            "1m": timedelta(minutes=1),
            "3m": timedelta(minutes=3),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "4h": timedelta(hours=4),
            "6h": timedelta(hours=6),
            "8h": timedelta(hours=8),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),
        }
    
    async def _ensure_session(self):
        """S'assure qu'une session HTTP est disponible."""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            if not self.verify_ssl:
                # Créer un contexte SSL qui ignore la vérification
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                self.session = aiohttp.ClientSession(connector=connector)
            else:
                self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Effectue une requête GET avec gestion des erreurs.
        
        Args:
            endpoint: Endpoint de l'API
            params: Paramètres de la requête
        
        Returns:
            Données de la réponse
        """
        await self._ensure_session()
        await self.rate_limiter.wait()
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(3):
            try:
                async with self.session.get(url, params=params, ssl=False) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 5))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(5)
    
    def get_cache_file(self, symbol: str, interval: str) -> Path:
        """
        Retourne le chemin du fichier de cache pour un symbole et un intervalle donnés.

        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            Path: Chemin du fichier de cache
        """
        # Créer le répertoire de cache s'il n'existe pas
        cache_dir = self.data_dir / "binance_history/klines"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Retourner le chemin du fichier
        return cache_dir / f"{symbol}_{interval}_full.csv"

    async def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str) -> None:
        """
        Sauvegarde les données dans le cache.

        Args:
            df: DataFrame à sauvegarder
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)
        """
        if df.empty:
            return

        try:
            cache_file = self.get_cache_file(symbol, interval)
            df.to_csv(cache_file)
            logger.info(f"Données sauvegardées dans {cache_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde dans le cache: {e}")

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Récupère les données OHLCV depuis l'API Binance.

        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Date de début (optionnel)
            end_time: Date de fin (optionnel)

        Returns:
            DataFrame pandas avec les colonnes: open, high, low, close, volume
            Source des données ("binance")
        """
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        try:
            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1000
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            async with self.session.get(url, params=params, ssl=False) as response:
                if response.status != 200:
                    logger.error(f"Erreur lors de la récupération des données: {response.status} {response.reason}")
                    return pd.DataFrame(), "binance"
                
                data = await response.json()
                
                if not data:
                    logger.warning(f"Pas de données pour {symbol} {interval}")
                    return pd.DataFrame(), "binance"
                
                # Convertir les données en DataFrame
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_volume",
                    "taker_buy_quote_volume", "ignore"
                ])
                
                # Convertir les types
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Définir l'index
                df.set_index("timestamp", inplace=True)
                
                # Ne garder que les colonnes qui nous intéressent
                df = df[["open", "high", "low", "close", "volume"]]
                
                # Sauvegarder dans le cache
                await self.save_to_cache(df, symbol, interval)
                
                return df, "binance"
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame(), "binance"
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> dict:
        """
        Récupère le carnet d'ordres pour un symbole.

        Args:
            symbol: Paire de trading (ex: BTCUSDT)
            limit: Nombre de niveaux à récupérer (max 5000)
            
        Returns:
            dict: Carnet d'ordres avec bids et asks
        """
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        try:
            url = f"{self.base_url}/depth"
            params = {
                "symbol": symbol,
                "limit": limit
            }
            
            async with self.session.get(url, params=params, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "bids": [[float(p), float(q)] for p, q in data["bids"]],
                        "asks": [[float(p), float(q)] for p, q in data["asks"]],
                        "lastUpdateId": data["lastUpdateId"]
                    }
                else:
                    logger.error(f"Erreur lors de la récupération du carnet d'ordres: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres: {str(e)}")
            return None
    
    async def download_historical_data(
        self,
        symbol: str,
        interval: str,
        start_year: int,
        end_year: int,
        output_dir: Union[str, Path]
    ):
        """
        Télécharge les données historiques par année.

        Args:
            symbol: Symbole (ex: BTCUSDT)
            interval: Intervalle (1m, 1h, 1d, etc.)
            start_year: Année de début
            end_year: Année de fin
            output_dir: Répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Téléchargement des données {symbol} {interval} pour {year}")
            
            start_date = datetime(year, 1, 1)
            end_date = datetime(year + 1, 1, 1)
            
            if end_date > datetime.now():
                end_date = datetime.now()
            
            # Télécharger par morceaux
            chunk_start = start_date
            all_data = []
            
            while chunk_start < end_date:
                chunk_end = min(
                    chunk_start + timedelta(days=30),
                    end_date
                )
                
                try:
                    df, _ = await self.get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=chunk_start,
                        end_time=chunk_end,
                        limit=1000
                    )
                    all_data.append(df)
                    
                    # Attendre un peu pour éviter de surcharger l'API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(
                        f"Erreur téléchargement {symbol} {interval} "
                        f"du {chunk_start} au {chunk_end}: {str(e)}"
                    )
                    await asyncio.sleep(5)
                    continue
                
                chunk_start = chunk_end
            
            if all_data:
                # Concaténer et sauvegarder
                full_df = pd.concat(all_data)
                
                # Nettoyer les doublons
                full_df = full_df[~full_df.index.duplicated(keep='first')]
                
                # Sauvegarder en CSV compressé
                filename = output_dir / f"{symbol}_{interval}_{year}.csv.gz"
                full_df.to_csv(
                    filename,
                    compression='gzip'
                )
                
                logger.info(
                    f"Données {symbol} {interval} {year} sauvegardées: "
                    f"{len(full_df)} bougies"
                )
            
            # Attendre entre les années
            await asyncio.sleep(1)
    
    def merge_yearly_files(
        self,
        symbol: str,
        interval: str,
        data_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Fusionne les fichiers annuels en un seul DataFrame.

        Args:
            symbol: Symbole
            interval: Intervalle
            data_dir: Répertoire des données
            output_file: Fichier de sortie (optionnel)
        
        Returns:
            DataFrame avec toutes les données
        """
        data_dir = Path(data_dir)
        pattern = f"{symbol}_{interval}_*.csv.gz"
        
        all_data = []
        
        # Charger tous les fichiers
        for file in sorted(data_dir.glob(pattern)):
            df = pd.read_csv(
                file,
                compression='gzip',
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"Aucun fichier trouvé pour {symbol} {interval}")
        
        # Concaténer
        full_df = pd.concat(all_data)
        
        # Nettoyer et trier
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        full_df.sort_index(inplace=True)
        
        # Sauvegarder si demandé
        if output_file:
            full_df.to_csv(
                output_file,
                compression='gzip'
            )
        
        return full_df

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_str: Optional[str] = None
    ) -> List[List]:
        """
        Récupère les données historiques OHLCV de façon synchrone.
        
        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)
            limit: Nombre maximum de bougies à récupérer
            start_str: Date de début au format "1 day ago UTC", "1 month ago UTC", etc.
            
        Returns:
            Liste de listes contenant les données OHLCV
        """
        logger.info(f"Récupération des données historiques pour {symbol} ({interval})")
        
        # Conversion de start_str en objet datetime
        start_time = None
        if start_str:
            if "day" in start_str:
                days = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(days=days)
            elif "month" in start_str:
                months = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(days=30*months)
            elif "year" in start_str:
                years = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(days=365*years)
        
        # Création de la session pour les requêtes
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            # Préparation des paramètres
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)  # Maximum 1000 par requête
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            
            # Requête à l'API
            url = f"{self.base_url}/klines"
            
            # Exécution synchrone
            import requests
            response = requests.get(url, params=params, verify=False)
            response.raise_for_status()
            
            # Conversion en DataFrame
            data = response.json()
            
            # Si les données sont vides, on retourne une liste vide
            if not data:
                logger.warning(f"Pas de données pour {symbol} {interval}")
                return []
            
            # Construction du DataFrame
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            
            # Conversion des types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            df.set_index("timestamp", inplace=True)
            
            # Sauvegarder dans le cache
            cache_file = self.get_cache_file(symbol, interval)
            df.to_csv(cache_file)
            
            # Retourner les données au format attendu
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données historiques: {e}")
            return []
