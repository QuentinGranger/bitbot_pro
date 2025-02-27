"""
Client pour l'API Glassnode avec focus sur les métriques on-chain.
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from dataclasses import dataclass

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger
from bitbot.data.market_data import MarketDataConfig

class GlassnodeClient:
    """Client pour l'API Glassnode avec focus sur les métriques on-chain."""
    
    def __init__(self, config: MarketDataConfig):
        """
        Args:
            config: Configuration du client
        """
        self.config = config
        self.base_url = "https://api.glassnode.com/v1"
        self.rate_limiter = RateLimiter(10, 60)  # 10 requêtes par minute (plan gratuit)
        self.session = None
        self.cache = {}
        
        if not config.glassnode_api_key:
            raise ValueError("Clé API Glassnode requise")
    
    async def _ensure_session(self):
        """Assure qu'une session HTTP est active."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Effectue une requête GET avec gestion des erreurs.
        
        Args:
            endpoint: Endpoint de l'API
            params: Paramètres de la requête
        
        Returns:
            DataFrame avec les données
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
        params = params or {}
        params['api_key'] = self.config.glassnode_api_key
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Convertir en DataFrame
                    df = pd.DataFrame(data)
                    if 't' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        df.drop('t', axis=1, inplace=True)
                    
                    # Mettre en cache
                    self.cache[cache_key] = (df, time.time())
                    return df
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)
    
    async def get_mvrv(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Récupère le ratio MVRV (Market Value to Realized Value).
        
        Args:
            asset: Actif (BTC, ETH)
            since: Date de début
            until: Date de fin
            interval: Intervalle
        
        Returns:
            DataFrame avec les données MVRV
        """
        params = {
            "a": asset,
            "i": interval
        }
        
        if since:
            params['s'] = int(since.timestamp())
        if until:
            params['u'] = int(until.timestamp())
        
        return await self._get("metrics/market/mvrv", params)
    
    async def get_exchange_netflow(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Récupère le flux net des exchanges.
        
        Args:
            asset: Actif (BTC, ETH)
            since: Date de début
            until: Date de fin
            interval: Intervalle
        
        Returns:
            DataFrame avec les données de flux
        """
        params = {
            "a": asset,
            "i": interval
        }
        
        if since:
            params['s'] = int(since.timestamp())
        if until:
            params['u'] = int(until.timestamp())
        
        return await self._get("metrics/transactions/exchanges_net_flow_volume", params)
    
    async def get_sopr(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Récupère le ratio SOPR (Spent Output Profit Ratio).
        
        Args:
            asset: Actif (BTC, ETH)
            since: Date de début
            until: Date de fin
            interval: Intervalle
        
        Returns:
            DataFrame avec les données SOPR
        """
        params = {
            "a": asset,
            "i": interval
        }
        
        if since:
            params['s'] = int(since.timestamp())
        if until:
            params['u'] = int(until.timestamp())
        
        return await self._get("metrics/indicators/sopr", params)
    
    async def get_active_addresses(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Récupère le nombre d'adresses actives.
        
        Args:
            asset: Actif (BTC, ETH)
            since: Date de début
            until: Date de fin
            interval: Intervalle
        
        Returns:
            DataFrame avec les données d'adresses actives
        """
        params = {
            "a": asset,
            "i": interval
        }
        
        if since:
            params['s'] = int(since.timestamp())
        if until:
            params['u'] = int(until.timestamp())
        
        return await self._get("metrics/addresses/active_count", params)
    
    async def get_supply_distribution(
        self,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Récupère la distribution de l'offre.
        
        Args:
            asset: Actif (BTC, ETH)
            since: Date de début
            until: Date de fin
            interval: Intervalle
        
        Returns:
            DataFrame avec les données de distribution
        """
        params = {
            "a": asset,
            "i": interval
        }
        
        if since:
            params['s'] = int(since.timestamp())
        if until:
            params['u'] = int(until.timestamp())
        
        return await self._get("metrics/distribution/balance_exchanges", params)
