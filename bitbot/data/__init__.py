"""Data acquisition and processing modules."""

from bitbot.data.market_data import MarketDataProvider
from bitbot.data.binance_client import BinanceClient
from bitbot.data.coingecko_client import CoinGeckoClient
from bitbot.data.coinmarketcap_client import CoinMarketCapClient
from bitbot.data.alternative_me_client import AlternativeMeClient

__all__ = [
    "MarketDataProvider",
    "BinanceClient",
    "CoinGeckoClient",
    "CoinMarketCapClient",
    "AlternativeMeClient"
]
