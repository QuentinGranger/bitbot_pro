"""
Clients pour l'acquisition de données de marché depuis diverses sources.
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from decimal import Decimal
import pytz
from pathlib import Path
import gzip

from bitbot.utils.rate_limiter import RateLimiter
from bitbot.utils.logger import logger
from bitbot.data.binance_client import BinanceClient
from bitbot.data.coingecko_client import CoinGeckoClient
from bitbot.data.coinmarketcap_client import CoinMarketCapClient
from bitbot.data.alternative_me_client import AlternativeMeClient
from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.models.market_data import FearGreedIndex, GoogleTrendsData

@dataclass
class MarketDataConfig:
    """Configuration pour les clients de données."""
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None
    cache_duration: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: int = 5
    verify_ssl: bool = True
    data_dir: str = "data"  # Répertoire racine des données
    enable_fear_greed_index: bool = True  # Activer l'indice de peur et d'avidité
    enable_google_trends: bool = True  # Activer les données de Google Trends

class MarketDataProvider:
    """
    Fournisseur de données de marché avec redondance.
    
    Utilise plusieurs sources dans l'ordre suivant :
    1. Binance (toutes les résolutions)
    2. CoinGecko (données journalières uniquement)
    3. CoinMarketCap (données journalières uniquement)
    4. Cache local en cas d'échec de toutes les sources en ligne
    
    Sources complémentaires :
    - Fear & Greed Index (Alternative.me) pour le sentiment du marché
    """
    
    def __init__(self, config: MarketDataConfig):
        """Initialise les clients et le mapping des intervalles."""
        self.config = config
        self.binance = BinanceClient(verify_ssl=config.verify_ssl)
        self.coingecko = CoinGeckoClient(config)
        self.coinmarketcap = CoinMarketCapClient(config) if config.coinmarketcap_api_key else None
        
        # Configuration du client AlternativeMeClient avec la même vérification SSL
        alternative_config = {
            "verify_ssl": config.verify_ssl
        }
        self.alternative_me = AlternativeMeClient(alternative_config) if config.enable_fear_greed_index else None
        
        # Configuration du client GoogleTrendsClient avec la même vérification SSL
        google_trends_config = {
            "verify_ssl": config.verify_ssl
        }
        self.google_trends = GoogleTrendsClient(google_trends_config) if config.enable_google_trends else None
        
        # Répertoire de données
        self.data_dir = Path(config.data_dir)
        
        # Mapping des intervalles pour chaque source
        self.interval_mapping = {
            "1m": {
                "binance": "1m",
                "coingecko": None,  # Non supporté
                "coinmarketcap": None  # Non supporté
            },
            "5m": {
                "binance": "5m",
                "coingecko": None,
                "coinmarketcap": None
            },
            "15m": {
                "binance": "15m",
                "coingecko": None,
                "coinmarketcap": None
            },
            "1h": {
                "binance": "1h",
                "coingecko": None,
                "coinmarketcap": None
            },
            "4h": {
                "binance": "4h",
                "coingecko": None,
                "coinmarketcap": None
            },
            "1d": {
                "binance": "1d",
                "coingecko": "daily",  # Corrigé
                "coinmarketcap": "daily"  # Corrigé
            }
        }
    
    async def close(self):
        """Ferme toutes les connexions."""
        await asyncio.gather(
            self.binance.close(),
            self.coingecko.close(),
            self.coinmarketcap.close() if self.coinmarketcap else asyncio.sleep(0),
            self.alternative_me.close() if self.alternative_me else asyncio.sleep(0),
            self.google_trends.close() if self.google_trends else asyncio.sleep(0)
        )
    
    def _get_cached_data(self, symbol: str, interval: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Récupère les données depuis le cache.
        
        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Date de début (optionnel)
            end_time: Date de fin (optionnel)
            
        Returns:
            DataFrame pandas avec les colonnes: open, high, low, close, volume
        """
        # Essayer de lire depuis le cache Binance
        if hasattr(self.binance, 'get_cache_file'):
            try:
                cache_file = self.binance.get_cache_file(symbol, interval)
                if cache_file and cache_file.exists():
                    df = pd.read_csv(cache_file, index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
                    
                    # Filtrer par date si nécessaire
                    if start_time:
                        df = df[df.index >= start_time]
                    if end_time:
                        df = df[df.index <= end_time]
                    
                    return df
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du cache Binance: {e}")
        
        # Essayer de lire depuis le cache CoinGecko
        if hasattr(self.coingecko, 'get_cache_file'):
            try:
                cache_file = self.coingecko.get_cache_file(symbol, interval)
                if cache_file and cache_file.exists():
                    df = pd.read_csv(cache_file, index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
                    
                    # Filtrer par date si nécessaire
                    if start_time:
                        df = df[df.index >= start_time]
                    if end_time:
                        df = df[df.index <= end_time]
                    
                    return df
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du cache CoinGecko: {e}")
        
        return pd.DataFrame()
        
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Récupère les données OHLCV depuis une source disponible.
        
        Args:
            symbol: Symbole de la paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (ex: 1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Date de début (optionnel)
            end_time: Date de fin (optionnel)
            
        Returns:
            Tuple[DataFrame, str]: Données et source utilisée
        """
        # Essayer Binance en premier
        logger.info(f"Tentative avec binance pour {symbol} {interval} -> {interval}")
        try:
            df, source = await self.binance.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            if not df.empty:
                return df, source
        except Exception as e:
            logger.warning(f"Erreur avec binance: {e}")
        
        # Essayer CoinGecko
        interval_map = {
            "1d": "daily",
            "1h": "hourly"
        }
        if interval in interval_map:
            logger.info(f"Tentative avec coingecko pour {symbol} {interval} -> {interval_map[interval]}")
            try:
                df, source = await self.coingecko.get_klines(
                    symbol=symbol,
                    interval=interval_map[interval],
                    start_time=start_time,
                    end_time=end_time
                )
                if not df.empty:
                    return df, source
            except Exception as e:
                logger.warning(f"Erreur avec coingecko: {e}")
        
        # Si toutes les sources en ligne ont échoué, essayer le cache
        logger.warning(f"Toutes les sources en ligne ont échoué, tentative avec le cache")
        df = self._get_cached_data(symbol, interval, start_time, end_time)
        if not df.empty:
            return df, "cache"
        
        logger.error(f"Aucune source disponible pour {symbol} {interval}")
        return pd.DataFrame(), "none"
    
    async def get_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> Optional[Dict]:
        """
        Récupère le carnet d'ordres (Binance uniquement).
        
        Args:
            symbol: Paire de trading
            limit: Profondeur du carnet
            
        Returns:
            Dict: Carnet d'ordres ou None
        """
        try:
            return await self.binance.get_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres: {str(e)}")
            return None

    async def get_latest_fear_greed_index(self) -> Optional[FearGreedIndex]:
        """
        Récupère l'indice de peur et d'avidité le plus récent.
        
        Returns:
            Dernier indice de peur et d'avidité ou None en cas d'erreur
        """
        if not self.alternative_me:
            logger.warning("Le client AlternativeMeClient n'est pas activé")
            return None
            
        try:
            return await self.alternative_me.get_latest_fear_greed_index()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du Fear & Greed Index: {str(e)}")
            return None
    
    async def get_fear_greed_history(self, days: int = 30) -> pd.DataFrame:
        """
        Récupère l'historique de l'indice de peur et d'avidité.
        
        Args:
            days: Nombre de jours d'historique à récupérer
            
        Returns:
            DataFrame avec l'historique de l'indice
        """
        if not self.alternative_me:
            logger.warning("Le client AlternativeMeClient n'est pas activé")
            return pd.DataFrame()
            
        try:
            return await self.alternative_me.get_fear_greed_dataframe(days=days)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique du Fear & Greed Index: {str(e)}")
            return pd.DataFrame()
    
    async def get_market_sentiment(self) -> Dict:
        """
        Récupère et analyse le sentiment général du marché basé sur l'indice de peur et d'avidité.
        
        Returns:
            Dictionnaire avec les indicateurs de sentiment :
                - value: Valeur brute de l'indice (0-100)
                - classification: Classification textuelle (ex: "Fear", "Greed")
                - sentiment_score: Score normalisé entre -1 et 1
                - market_phase: Phase du marché estimée
                - recommended_action: Action recommandée basée sur le sentiment
        """
        fear_greed = await self.get_latest_fear_greed_index()
        
        if not fear_greed:
            return {
                "value": 50,
                "classification": "Neutral",
                "sentiment_score": 0.0,
                "market_phase": "Indéterminée",
                "recommended_action": "Attendre et observer"
            }
        
        sentiment_score = fear_greed.get_sentiment_score()
        
        # Déterminer la phase du marché
        if sentiment_score < -0.5:  # Peur extrême
            market_phase = "Pessimisme extrême"
            recommended_action = "Considérer l'achat (sentiment très négatif)"
        elif sentiment_score < -0.2:  # Peur
            market_phase = "Pessimisme"
            recommended_action = "Surveiller les opportunités d'achat"
        elif sentiment_score < 0.2:  # Neutre
            market_phase = "Équilibre"
            recommended_action = "Maintenir la stratégie actuelle"
        elif sentiment_score < 0.5:  # Avidité
            market_phase = "Optimisme"
            recommended_action = "Prudence, envisager de prendre des bénéfices"
        else:  # Avidité extrême
            market_phase = "Optimisme extrême"
            recommended_action = "Considérer la vente (sentiment très positif)"
        
        return {
            "value": fear_greed.value,
            "classification": fear_greed.classification,
            "sentiment_score": sentiment_score,
            "market_phase": market_phase,
            "recommended_action": recommended_action,
            "timestamp": fear_greed.timestamp,
            "time_until_update": fear_greed.time_until_update
        }

    async def get_google_trends(self, keyword: str = "bitcoin", timeframe: str = "today 5-y") -> Optional[GoogleTrendsData]:
        """
        Récupère les données de Google Trends pour le Bitcoin.
        
        Args:
            keyword: Mot-clé à rechercher (par défaut: "bitcoin")
            timeframe: Période de temps (today 5-y, today 12-m, today 3-m, etc.)
            
        Returns:
            Données de Google Trends ou None en cas d'erreur
        """
        if not self.google_trends:
            logger.warning("Le client GoogleTrendsClient n'est pas activé")
            return None
            
        try:
            return await self.google_trends.get_interest_over_time(keyword, timeframe)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de Google Trends: {str(e)}")
            return None
    
    async def get_bitcoin_price_trends_correlation(self, days: int = 90) -> pd.DataFrame:
        """
        Calcule la corrélation entre les prix du Bitcoin et les recherches Google.
        
        Args:
            days: Nombre de jours de données à analyser
            
        Returns:
            DataFrame avec les données fusionnées et les corrélations
        """
        if not self.google_trends:
            logger.warning("Le client GoogleTrendsClient n'est pas activé")
            return pd.DataFrame()
            
        try:
            # Récupérer les données de prix du Bitcoin
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Récupérer les données OHLCV quotidiennes
            price_data = await self.get_klines("BTCUSDT", "1d", start_time, end_time)
            
            if price_data[0].empty:
                logger.error("Impossible de récupérer les données de prix pour le Bitcoin")
                return pd.DataFrame()
            
            # Déterminer le timeframe Google Trends approprié
            if days <= 30:
                timeframe = "today 1-m"
            elif days <= 90:
                timeframe = "today 3-m"
            elif days <= 180:
                timeframe = "today 6-m"
            elif days <= 365:
                timeframe = "today 12-m"
            else:
                timeframe = "today 5-y"
            
            # Récupérer les données de tendances
            trends_data = await self.get_google_trends("bitcoin", timeframe)
            
            if not trends_data:
                logger.error("Impossible de récupérer les données de Google Trends")
                return pd.DataFrame()
            
            # Calculer les corrélations
            return await self.google_trends.get_bitcoin_correlations(price_data[0], trends_data)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des corrélations: {str(e)}")
            return pd.DataFrame()
