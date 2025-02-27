"""
Client pour l'API CoinMarketCap.
"""

import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional
import pytz

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger

class CoinMarketCapClient:
    """Client pour l'API CoinMarketCap."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration avec l'API key
        """
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.api_key = config.coinmarketcap_api_key
        self.session = None
        self.rate_limiter = RateLimiter(30, 60)  # 30 requêtes par minute
        
        # Mapping des symboles
        self.symbol_mapping = {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            # Ajouter d'autres paires si nécessaire
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
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Récupère les données OHLCV.
        
        Args:
            symbol: Paire de trading (ex: BTCUSDT)
            interval: Intervalle (daily uniquement)
            start_time: Timestamp de début
            end_time: Timestamp de fin
            
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        try:
            # Convertir le symbole
            if symbol not in self.symbol_mapping:
                logger.warning(f"Symbole {symbol} non supporté par CoinMarketCap")
                return pd.DataFrame()
            
            coin_symbol = self.symbol_mapping[symbol]
            
            # Construire l'URL
            url = f"{self.base_url}/cryptocurrency/ohlcv/historical"
            params = {
                "symbol": coin_symbol,
                "convert": "USD",
                "interval": "daily",
                "count": 365  # Maximum 365 jours
            }
            
            headers = {
                "X-CMC_PRO_API_KEY": self.api_key
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Erreur API CoinMarketCap: {response.status}")
                data = await response.json()
                
                # Extraire les données
                quotes = data["data"]["quotes"]
                
                # Convertir en DataFrame
                records = []
                for quote in quotes:
                    record = {
                        "timestamp": pd.Timestamp(quote["timestamp"]),
                        "open": quote["quote"]["USD"]["open"],
                        "high": quote["quote"]["USD"]["high"],
                        "low": quote["quote"]["USD"]["low"],
                        "close": quote["quote"]["USD"]["close"],
                        "volume": quote["quote"]["USD"]["volume"]
                    }
                    records.append(record)
                
                df = pd.DataFrame.from_records(records)
                df.set_index("timestamp", inplace=True)
                
                # Ajouter le fuseau horaire UTC
                df.index = df.index.tz_localize(pytz.UTC)
                
                # Filtrer par date si nécessaire
                if start_time:
                    df = df[df.index >= pd.Timestamp(start_time)]
                if end_time:
                    df = df[df.index <= pd.Timestamp(end_time)]
                
                return df
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données CoinMarketCap: {str(e)}")
            return pd.DataFrame()
