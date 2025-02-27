"""
Client pour l'API CoinGecko.
"""

import logging
import aiohttp
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class CoinGeckoClient:
    """Client pour l'API CoinGecko."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration avec l'API key
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.verify_ssl = config.verify_ssl
        self.session = None
        self.symbol_mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            # Ajouter d'autres mappings au besoin
        }
    
    async def _ensure_session(self):
        """S'assure qu'une session HTTP est disponible."""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def get_klines(self, symbol: str, interval: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Tuple[pd.DataFrame, str]:
        """
        Récupère les données OHLCV depuis l'API CoinGecko.

        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: daily)
            start_time: Date de début (optionnel)
            end_time: Date de fin (optionnel)

        Returns:
            DataFrame pandas avec les colonnes: open, high, low, close, volume
            Source des données ("coingecko")
        """
        await self._ensure_session()
        
        try:
            # Convertir le symbole
            if symbol not in self.symbol_mapping:
                logger.warning(f"Symbole {symbol} non supporté par CoinGecko")
                return pd.DataFrame(), "coingecko"
            
            coin_id = self.symbol_mapping[symbol]
            
            # Construire l'URL
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                "vs_currency": "usd",
                "days": "1"  # Pour l'intervalle daily
            }
            
            async with self.session.get(url, params=params, ssl=self.verify_ssl) as response:
                if response.status != 200:
                    logger.error(f"Erreur lors de la récupération des données: {response.status} {response.reason}")
                    return pd.DataFrame(), "coingecko"
                
                data = await response.json()
                
                if not data:
                    logger.warning(f"Pas de données pour {symbol} {interval}")
                    return pd.DataFrame(), "coingecko"
                
                # Convertir les données en DataFrame
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
                
                # Convertir les types
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Ajouter une colonne volume (non disponible dans l'API gratuite)
                df["volume"] = 0.0
                
                # Définir l'index
                df.set_index("timestamp", inplace=True)
                
                # Filtrer par date si nécessaire
                if start_time:
                    df = df[df.index >= start_time]
                if end_time:
                    df = df[df.index <= end_time]
                
                return df, "coingecko"
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données CoinGecko: {e}")
            return pd.DataFrame(), "coingecko"
