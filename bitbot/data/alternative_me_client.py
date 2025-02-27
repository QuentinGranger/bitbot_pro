"""
Client pour l'API Alternative.me permettant de récupérer l'indice de peur et d'avidité.
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger
from bitbot.models.market_data import FearGreedIndex

class AlternativeMeClient:
    """Client pour l'API Alternative.me permettant de récupérer l'indice de peur et d'avidité."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration
        """
        self.base_url = "https://api.alternative.me/fng/"
        self.session = None
        self.rate_limiter = RateLimiter(10, 60)  # 10 requêtes par minute pour éviter de surcharger l'API gratuite
        self.cache = {}
        self.cache_ttl = 3600  # 1 heure en secondes
        self.verify_ssl = getattr(config, 'verify_ssl', False)  # Désactiver la vérification SSL par défaut
    
    async def _ensure_session(self):
        """Assure qu'une session HTTP est active."""
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=self.verify_ssl))
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_fear_greed_index(
        self, 
        limit: int = 1, 
        date_format: str = ""
    ) -> List[FearGreedIndex]:
        """
        Récupère l'indice de peur et d'avidité actuel ou historique.
        
        Args:
            limit: Nombre de jours à récupérer. Utilisez 0 pour toutes les données disponibles.
            date_format: Format de date (vide = unix timestamp, "us" = MM/DD/YYYY, 
                        "world" = DD/MM/YYYY, "cn" ou "kr" = YYYY/MM/DD)
        
        Returns:
            Liste d'indices de peur et d'avidité, du plus récent au plus ancien
        """
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        # Vérifier si les données sont en cache et toujours valides
        cache_key = f"fng_{limit}_{date_format}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now().timestamp() - cache_time < self.cache_ttl:
                logger.debug(f"Utilisation du cache pour Fear & Greed (limite: {limit})")
                return data
        
        params = {
            "limit": limit
        }
        
        if date_format:
            params["date_format"] = date_format
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Erreur API Alternative.me: {response.status}")
                    return []
                
                data = await response.json()
                
                if data.get("metadata", {}).get("error"):
                    logger.error(f"Erreur API Alternative.me: {data['metadata']['error']}")
                    return []
                
                result = [FearGreedIndex.from_api_response(item) for item in data["data"]]
                
                # Mettre en cache les résultats
                self.cache[cache_key] = (datetime.now().timestamp(), result)
                
                return result
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du Fear & Greed Index: {str(e)}")
            return []
    
    async def get_latest_fear_greed_index(self) -> Optional[FearGreedIndex]:
        """
        Récupère l'indice de peur et d'avidité le plus récent.
        
        Returns:
            Indice de peur et d'avidité ou None en cas d'erreur
        """
        indices = await self.get_fear_greed_index(limit=1)
        return indices[0] if indices else None
    
    async def get_fear_greed_dataframe(self, days: int = 30) -> pd.DataFrame:
        """
        Récupère l'historique de l'indice de peur et d'avidité sous forme de DataFrame.
        
        Args:
            days: Nombre de jours d'historique à récupérer
            
        Returns:
            DataFrame avec les colonnes date, value, classification, sentiment_score
        """
        indices = await self.get_fear_greed_index(limit=days)
        
        if not indices:
            return pd.DataFrame(columns=["date", "value", "classification", "sentiment_score"])
        
        data = {
            "date": [idx.timestamp for idx in indices],
            "value": [idx.value for idx in indices],
            "classification": [idx.classification for idx in indices],
            "sentiment_score": [idx.get_sentiment_score() for idx in indices]
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values("date")  # Trier par date croissante
        
        return df
