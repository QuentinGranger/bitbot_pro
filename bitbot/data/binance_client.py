"""
Client Binance pour la récupération de données historiques.
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from pathlib import Path
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger

class BinanceClient:
    """Client optimisé pour la récupération de données historiques Binance."""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limiter = RateLimiter(1200, 60)  # 1200 requêtes par minute
        self.session = None
        
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
        """Assure qu'une session HTTP est active."""
        if self.session is None:
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
                async with self.session.get(url, params=params) as response:
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
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Récupère les données OHLCV.
        
        Args:
            symbol: Symbole (ex: BTCUSDT)
            interval: Intervalle (1m, 1h, 1d, etc.)
            start_time: Date de début
            end_time: Date de fin
            limit: Nombre maximum de bougies
        
        Returns:
            DataFrame avec les données OHLCV
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        data = await self._get("klines", params)
        
        # Convertir en DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convertir les types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                         'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Convertir les timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Définir l'index
        df.set_index('timestamp', inplace=True)
        
        return df
    
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
                    df = await self.get_klines(
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
