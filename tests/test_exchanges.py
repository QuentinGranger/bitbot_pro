"""
Tests des intégrations avec les exchanges.
"""

import pytest
from unittest.mock import Mock
import pandas as pd
from datetime import datetime, timezone

from bitbot.exchanges.base import BaseExchange
from bitbot.exchanges.binance import BinanceExchange
from bitbot.models.order import Order, OrderSide, OrderType
from bitbot.models.position import Position

@pytest.mark.asyncio
async def test_binance_fetch_ohlcv(mock_binance_api, mock_config):
    """Test de la récupération des données OHLCV."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    
    assert isinstance(ohlcv, pd.DataFrame)
    assert not ohlcv.empty
    assert all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Vérifier les valeurs
    assert ohlcv['open'].iloc[0] == 8100.0
    assert ohlcv['high'].iloc[0] == 8200.0
    assert ohlcv['low'].iloc[0] == 8000.0
    assert ohlcv['close'].iloc[0] == 8150.0
    assert ohlcv['volume'].iloc[0] == 10.0

@pytest.mark.asyncio
async def test_binance_fetch_ticker(mock_binance_api, mock_config):
    """Test de la récupération du ticker."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    ticker = await exchange.fetch_ticker("BTC/USDT")
    
    assert ticker['last'] == 8150.0
    assert ticker['bid'] == 8145.0
    assert ticker['ask'] == 8155.0
    assert ticker['volume'] == 10.0

@pytest.mark.asyncio
async def test_binance_fetch_order_book(mock_binance_api, mock_config):
    """Test de la récupération du carnet d'ordres."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    order_book = await exchange.fetch_order_book("BTC/USDT")
    
    assert len(order_book['bids']) == 2
    assert len(order_book['asks']) == 2
    assert order_book['bids'][0][0] == 8145.0
    assert order_book['asks'][0][0] == 8155.0

@pytest.mark.asyncio
async def test_create_order(mock_binance_api, mock_config):
    """Test de la création d'un ordre."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    order = Order(
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=1.0,
        price=8100.0
    )
    
    # Mock la réponse de l'API pour la création d'ordre
    mock_binance_api.add(
        responses.POST,
        "https://api.binance.com/api/v3/order",
        json={
            "symbol": "BTCUSDT",
            "orderId": 1,
            "price": "8100.0",
            "origQty": "1.0",
            "status": "NEW"
        }
    )
    
    result = await exchange.create_order(order)
    assert result['id'] == '1'
    assert result['status'] == 'NEW'

@pytest.mark.asyncio
async def test_cancel_order(mock_binance_api, mock_config):
    """Test de l'annulation d'un ordre."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    # Mock la réponse de l'API pour l'annulation d'ordre
    mock_binance_api.add(
        responses.DELETE,
        "https://api.binance.com/api/v3/order",
        json={
            "symbol": "BTCUSDT",
            "orderId": 1,
            "status": "CANCELED"
        }
    )
    
    result = await exchange.cancel_order("1", "BTC/USDT")
    assert result['status'] == 'CANCELED'

@pytest.mark.asyncio
async def test_fetch_balance(mock_binance_api, mock_config):
    """Test de la récupération du solde."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    # Mock la réponse de l'API pour le solde
    mock_binance_api.add(
        responses.GET,
        "https://api.binance.com/api/v3/account",
        json={
            "balances": [
                {
                    "asset": "BTC",
                    "free": "1.0",
                    "locked": "0.5"
                },
                {
                    "asset": "USDT",
                    "free": "1000.0",
                    "locked": "500.0"
                }
            ]
        }
    )
    
    balance = await exchange.fetch_balance()
    
    assert balance['BTC']['free'] == 1.0
    assert balance['BTC']['used'] == 0.5
    assert balance['USDT']['free'] == 1000.0
    assert balance['USDT']['used'] == 500.0

@pytest.mark.asyncio
async def test_error_handling(mock_binance_api, mock_config):
    """Test de la gestion des erreurs."""
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    # Mock une erreur API
    mock_binance_api.add(
        responses.GET,
        "https://api.binance.com/api/v3/klines",
        status=429,
        json={
            "code": -1003,
            "msg": "Too many requests"
        }
    )
    
    with pytest.raises(Exception) as exc_info:
        await exchange.fetch_ohlcv("BTC/USDT", "1m")
    assert "Too many requests" in str(exc_info.value)

@pytest.mark.asyncio
async def test_websocket_connection(mocker):
    """Test de la connexion WebSocket."""
    mock_ws = Mock()
    mocker.patch('websockets.connect', return_value=mock_ws)
    
    exchange = BinanceExchange(mock_config["exchanges"]["binance"])
    
    # Mock les messages WebSocket
    mock_ws.recv.side_effect = [
        '{"e":"trade","s":"BTCUSDT","p":"8150.0","q":"1.0"}',
        Exception("Connection closed")
    ]
    
    messages = []
    async def message_handler(msg):
        messages.append(msg)
    
    # Tester la reconnexion WebSocket
    await exchange.subscribe_trades("BTC/USDT", message_handler)
    
    assert len(messages) == 1
    assert messages[0]['price'] == 8150.0
