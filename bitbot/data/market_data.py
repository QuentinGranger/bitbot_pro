"""
Clients pour l'acquisition de données de marché depuis diverses sources.
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from decimal import Decimal

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger

@dataclass
class MarketDataConfig:
    """Configuration pour les clients de données."""
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None
    glassnode_api_key: Optional[str] = None
    cache_duration: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: int = 5

class CoinGeckoClient:
    """Client pour l'API CoinGecko."""
    
    def __init__(self, config: MarketDataConfig):
        """
        Args:
            config: Configuration du client
        """
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limiter = RateLimiter(50, 60)  # 50 requêtes par minute
        self.session = None
        self.cache = {}
    
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
        
        # Vérifier le cache
        cache_key = f"{endpoint}:{str(params)}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.config.cache_duration:
                return cached_data
        
        url = f"{self.base_url}/{endpoint}"
        if self.config.coingecko_api_key:
            params = params or {}
            params['x_cg_pro_api_key'] = self.config.coingecko_api_key
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Mettre en cache
                    self.cache[cache_key] = (data, time.time())
                    return data
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)
    
    async def get_price_history(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: Union[int, str] = "max",
        interval: str = "daily"
    ) -> pd.DataFrame:
        """
        Récupère l'historique des prix.
        
        Args:
            coin_id: ID de la crypto (ex: bitcoin)
            vs_currency: Devise de référence
            days: Nombre de jours ou "max"
            interval: Intervalle (daily, hourly)
        
        Returns:
            DataFrame avec l'historique
        """
        data = await self._get(
            f"coins/{coin_id}/market_chart",
            {
                "vs_currency": vs_currency,
                "days": days,
                "interval": interval
            }
        )
        
        # Convertir en DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ajouter le volume et la market cap
        df['volume'] = [x[1] for x in data['total_volumes']]
        df['market_cap'] = [x[1] for x in data['market_caps']]
        
        return df
    
    async def get_market_data(
        self,
        coin_id: str
    ) -> Dict:
        """
        Récupère les données de marché actuelles.
        
        Args:
            coin_id: ID de la crypto
        
        Returns:
            Données de marché
        """
        return await self._get(
            f"coins/{coin_id}",
            {
                "localization": "false",
                "tickers": "false",
                "community_data": "true",
                "developer_data": "false"
            }
        )

class CoinMarketCapClient:
    """Client pour l'API CoinMarketCap."""
    
    def __init__(self, config: MarketDataConfig):
        """
        Args:
            config: Configuration du client
        """
        self.config = config
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.rate_limiter = RateLimiter(30, 60)  # 30 requêtes par minute
        self.session = None
        self.cache = {}
        
        if not config.coinmarketcap_api_key:
            raise ValueError("Clé API CoinMarketCap requise")
    
    async def _ensure_session(self):
        """Assure qu'une session HTTP est active."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={'X-CMC_PRO_API_KEY': self.config.coinmarketcap_api_key}
            )
    
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
        
        # Vérifier le cache
        cache_key = f"{endpoint}:{str(params)}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.config.cache_duration:
                return cached_data
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Mettre en cache
                    self.cache[cache_key] = (data, time.time())
                    return data['data']
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)
    
    async def get_latest_quotes(
        self,
        symbol: str,
        convert: str = "USD"
    ) -> Dict:
        """
        Récupère les dernières cotations.
        
        Args:
            symbol: Symbole de la crypto
            convert: Devise de conversion
        
        Returns:
            Données de cotation
        """
        return await self._get(
            "cryptocurrency/quotes/latest",
            {
                "symbol": symbol,
                "convert": convert
            }
        )
    
    async def get_market_pairs(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Récupère les paires de trading.
        
        Args:
            symbol: Symbole de la crypto
            limit: Nombre maximum de résultats
        
        Returns:
            Liste des paires de trading
        """
        data = await self._get(
            "cryptocurrency/market-pairs/latest",
            {
                "symbol": symbol,
                "limit": limit
            }
        )
        return data.get('market_pairs', [])
