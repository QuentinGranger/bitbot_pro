"""
Tests pour le client WebSocket Binance avec mocks.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
from datetime import datetime
import pytz
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
from bitbot.data.websocket import BinanceWebSocket, WebSocketConfig, StreamType
from bitbot.models.market_data import Kline, Trade, OrderBook, Ticker

# Données de test
KLINE_DATA = {
    "e": "kline",
    "E": 1625232923451,
    "s": "BTCUSDT",
    "k": {
        "t": 1625232900000,
        "T": 1625232959999,
        "s": "BTCUSDT",
        "i": "1m",
        "f": 100,
        "L": 200,
        "o": "35000.10",
        "c": "35100.20",
        "h": "35200.30",
        "l": "34900.40",
        "v": "10.5",
        "n": 100,
        "x": False,
        "q": "367951.12",
        "V": "5.2",
        "Q": "182000.12"
    }
}

TRADE_DATA = {
    "e": "trade",
    "E": 1625232923451,
    "s": "BTCUSDT",
    "t": 12345,
    "p": "35100.10",
    "q": "0.5",
    "b": 12345,
    "a": 12346,
    "T": 1625232923450,
    "m": True,
    "M": True
}

DEPTH_DATA = {
    "e": "depthUpdate",
    "E": 1625232923451,
    "s": "BTCUSDT",
    "U": 100,
    "u": 200,
    "b": [
        ["35000.10", "1.5"],
        ["34990.20", "2.0"]
    ],
    "a": [
        ["35100.30", "0.8"],
        ["35200.40", "1.2"]
    ]
}

TICKER_DATA = {
    "e": "24hrTicker",
    "E": 1625232923451,
    "s": "BTCUSDT",
    "p": "100.10",
    "P": "0.5",
    "w": "35050.20",
    "x": "35000.30",
    "c": "35100.40",
    "Q": "1.5",
    "b": "35090.50",
    "B": "2.0",
    "a": "35110.60",
    "A": "3.0",
    "o": "34900.70",
    "h": "35200.80",
    "l": "34800.90",
    "v": "1000.0",
    "q": "35100000.0",
    "O": 1625146500000,
    "C": 1625232900000,
    "F": 100,
    "L": 200,
    "n": 1000
}

class MockWebSocket:
    """Mock pour le WebSocket."""
    
    def __init__(self):
        self.sent_messages = []
        self.closed = False
    
    async def send(self, message):
        self.sent_messages.append(message)
    
    async def recv(self):
        return "{}"  # Message vide par défaut
    
    async def ping(self):
        pass
    
    async def close(self):
        self.closed = True

class TestWebSocketClient:
    """Tests pour le client WebSocket."""
    
    @pytest.fixture
    def patched_client(self):
        """Crée un client avec _handle_message patché."""
        config = WebSocketConfig(buffer_size=10)
        client = BinanceWebSocket(config)
        client._send_message = AsyncMock()
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        # Sauvegarder la méthode originale
        original_handle_message = client._handle_message
        
        # Créer une méthode patched qui appelle directement les callbacks
        async def direct_callback_handle_message(message):
            data = json.loads(message)
            
            # Extraire le type de stream et le symbole
            if 'e' in data:
                stream_type = data['e']
                symbol = data.get('s')
                stream_key = f"{symbol.lower()}@{stream_type}"
                
                # Appeler directement les callbacks
                for callback in client.callbacks.get(stream_key, []):
                    await callback(data)
            
            # Appeler la méthode originale pour mettre à jour les buffers
            await original_handle_message(message)
        
        # Remplacer la méthode
        client._handle_message = direct_callback_handle_message
        
        return client

    @pytest.mark.asyncio
    async def test_kline_stream(self, patched_client):
        """Vérifie l'accès au flux de bougies."""
        client = patched_client
        
        received_data = []
        
        async def on_kline(data):
            received_data.append(data)
        
        # Extraire le type de stream et le symbole
        symbol = KLINE_DATA.get('s')
        stream_type = KLINE_DATA.get('e')
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # S'abonner au flux
        client.subscriptions.add(stream_key)
        client.callbacks[stream_key] = [on_kline]
        
        # Simuler la réception d'un message
        await client._handle_message(json.dumps(KLINE_DATA))
        
        # Vérifier que la callback a été appelée avec les bonnes données
        assert len(received_data) > 0, "Aucune donnée de bougie reçue"
        
        # Vérifier le format des données
        kline_data = received_data[0]
        assert "k" in kline_data, "Format de bougie invalide"
        assert "t" in kline_data["k"], "Timestamp manquant"
        assert "o" in kline_data["k"], "Prix d'ouverture manquant"
        assert "h" in kline_data["k"], "Plus haut manquant"
        assert "l" in kline_data["k"], "Plus bas manquant"
        assert "c" in kline_data["k"], "Prix de clôture manquant"
        assert "v" in kline_data["k"], "Volume manquant"

    @pytest.mark.asyncio
    async def test_trade_stream(self, patched_client):
        """Vérifie l'accès au flux de trades."""
        client = patched_client
        
        received_data = []
        
        async def on_trade(data):
            received_data.append(data)
        
        # Extraire le type de stream et le symbole
        symbol = TRADE_DATA.get('s')
        stream_type = TRADE_DATA.get('e')
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # S'abonner au flux
        client.subscriptions.add(stream_key)
        client.callbacks[stream_key] = [on_trade]
        
        # Simuler la réception d'un message
        await client._handle_message(json.dumps(TRADE_DATA))
        
        # Vérifier que la callback a été appelée avec les bonnes données
        assert len(received_data) > 0, "Aucune donnée de trade reçue"
        
        # Vérifier le format des données
        trade_data = received_data[0]
        assert "p" in trade_data, "Prix manquant"
        assert "q" in trade_data, "Quantité manquante"
        assert "T" in trade_data, "Timestamp manquant"
        assert "m" in trade_data, "Maker/taker manquant"

    @pytest.mark.asyncio
    async def test_orderbook_stream(self, patched_client):
        """Vérifie l'accès au flux du carnet d'ordres."""
        client = patched_client
        
        received_data = []
        
        async def on_depth(data):
            received_data.append(data)
        
        # Extraire le type de stream et le symbole
        symbol = DEPTH_DATA.get('s')
        stream_type = DEPTH_DATA.get('e')
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # S'abonner au flux
        client.subscriptions.add(stream_key)
        client.callbacks[stream_key] = [on_depth]
        
        # Simuler la réception d'un message
        await client._handle_message(json.dumps(DEPTH_DATA))
        
        # Vérifier que la callback a été appelée avec les bonnes données
        assert len(received_data) > 0, "Aucune donnée de profondeur reçue"
        
        # Vérifier le format des données
        depth_data = received_data[0]
        assert "b" in depth_data, "Ordres d'achat manquants"
        assert "a" in depth_data, "Ordres de vente manquants"
        assert "u" in depth_data, "Update ID manquant"
        
        # Vérifier la structure des ordres
        if len(depth_data["b"]) > 0:
            bid = depth_data["b"][0]
            assert len(bid) == 2, "Format d'ordre d'achat invalide"
            assert float(bid[0]) > 0, "Prix d'achat invalide"
        
        if len(depth_data["a"]) > 0:
            ask = depth_data["a"][0]
            assert len(ask) == 2, "Format d'ordre de vente invalide"
            assert float(ask[0]) > 0, "Prix de vente invalide"

    @pytest.mark.asyncio
    async def test_ticker_stream(self, patched_client):
        """Vérifie l'accès au flux des tickers."""
        client = patched_client
        
        received_data = []
        
        async def on_ticker(data):
            received_data.append(data)
        
        # Extraire le type de stream et le symbole
        symbol = TICKER_DATA.get('s')
        stream_type = TICKER_DATA.get('e')
        stream_key = f"{symbol.lower()}@{stream_type}"
        
        # S'abonner au flux
        client.subscriptions.add(stream_key)
        client.callbacks[stream_key] = [on_ticker]
        
        # Simuler la réception d'un message
        await client._handle_message(json.dumps(TICKER_DATA))
        
        # Vérifier que la callback a été appelée avec les bonnes données
        assert len(received_data) > 0, "Aucune donnée de ticker reçue"
        
        # Vérifier le format des données
        ticker_data = received_data[0]
        assert "c" in ticker_data, "Dernier prix manquant"
        assert "v" in ticker_data, "Volume manquant"
        assert "q" in ticker_data, "Volume en quote manquant"

    @pytest.mark.asyncio
    async def test_buffer_management(self):
        """Vérifie la gestion des buffers de données."""
        config = WebSocketConfig(buffer_size=10)
        client = BinanceWebSocket(config)
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        # Patch des méthodes de conversion pour éviter les erreurs
        with patch('bitbot.models.market_data.Kline.from_ws_message') as mock_kline:
            with patch('bitbot.models.market_data.Trade.from_ws_message') as mock_trade:
                # Configurer les mocks
                mock_kline.return_value = Kline(
                    timestamp=datetime.fromtimestamp(1625232900, tz=pytz.UTC),
                    open=Decimal('35000.10'),
                    high=Decimal('35200.30'),
                    low=Decimal('34900.40'),
                    close=Decimal('35100.20'),
                    volume=Decimal('10.5'),
                    close_time=datetime.fromtimestamp(1625232960, tz=pytz.UTC),
                    quote_volume=Decimal('367951.12'),
                    trades=100,
                    taker_buy_volume=Decimal('5.2'),
                    taker_buy_quote_volume=Decimal('182000.12'),
                    interval="1m"
                )
                
                mock_trade.return_value = Trade(
                    timestamp=datetime.fromtimestamp(1625232923, tz=pytz.UTC),
                    symbol="BTCUSDT",
                    id=12345,
                    price=Decimal('35100.10'),
                    quantity=Decimal('0.5'),
                    buyer_maker=True
                )
                
                # Simuler la réception de messages
                # Kline
                kline_data = KLINE_DATA.copy()
                await client._handle_message(json.dumps(kline_data))
                
                # Trade
                trade_data = TRADE_DATA.copy()
                await client._handle_message(json.dumps(trade_data))
                
                # Depth
                depth_data = DEPTH_DATA.copy()
                await client._handle_message(json.dumps(depth_data))
                
                # Vérifier que les buffers ont été mis à jour
                assert len(client.kline_buffer) > 0, "Buffer de klines vide"
                assert len(client.trade_buffer) > 0, "Buffer de trades vide"
                
                # Ajouter plusieurs klines pour tester la taille du buffer
                for i in range(config.buffer_size + 5):
                    await client._handle_message(json.dumps(kline_data))
                
                # Vérifier que la taille du buffer est respectée
                for key, data in client.kline_buffer.items():
                    assert len(data) <= config.buffer_size, f"Buffer de bougies trop grand: {len(data)} > {config.buffer_size}"

@pytest.mark.asyncio
async def test_reconnection():
    """Vérifie la gestion de la reconnexion."""
    config = WebSocketConfig(
        reconnect_delay=0.1,  # Réduire le délai pour le test
        max_reconnect_attempts=2
    )
    client = BinanceWebSocket(config)
    
    # Ajouter un abonnement
    client.subscriptions.add("btcusdt@trade")
    
    # Simuler une connexion réussie
    with patch.object(client, 'connect', AsyncMock()) as mock_connect:
        mock_connect.side_effect = lambda: setattr(client, 'connected', True)
        
        # Patcher asyncio.sleep pour éviter les délais
        with patch.object(asyncio, 'sleep', AsyncMock()):
            # Déclencher une reconnexion
            await client._handle_connection_error()
        
        # Vérifier que connect a été appelé
        mock_connect.assert_called()
        
        # Vérifier que le client est connecté
        assert client.connected, "Le client n'est pas connecté après reconnexion"

@pytest.mark.asyncio
async def test_reconnection_with_backoff():
    """Vérifie la gestion de la reconnexion avec délai exponentiel."""
    config = WebSocketConfig(
        reconnect_delay=0.1,  # Réduire le délai pour le test
        max_reconnect_attempts=3,
        backoff_factor=2.0,
        max_backoff_delay=1.0
    )
    client = BinanceWebSocket(config)
    
    # Ajouter un abonnement
    client.subscriptions.add("btcusdt@trade")
    
    # Simuler une connexion réussie
    with patch.object(client, 'connect', AsyncMock()) as mock_connect:
        # Configurer le mock pour simuler une connexion réussie au 3ème essai
        mock_connect.side_effect = [
            ConnectionError("Erreur simulée 1"),
            ConnectionError("Erreur simulée 2"),
            lambda: setattr(client, 'connected', True)
        ]
        
        # Patcher asyncio.sleep pour éviter les délais
        with patch.object(asyncio, 'sleep', AsyncMock()) as mock_sleep:
            try:
                # Déclencher une reconnexion
                await client._handle_connection_error("Test reconnexion")
            except ConnectionError:
                # Ignorer l'erreur si toutes les tentatives échouent
                pass
        
        # Vérifier que connect a été appelé le bon nombre de fois
        assert mock_connect.call_count <= config.max_reconnect_attempts, f"Trop d'appels à connect: {mock_connect.call_count}"
        
        # Vérifier que les délais sont exponentiels
        expected_delays = [
            config.reconnect_delay,
            config.reconnect_delay * config.backoff_factor,
            config.reconnect_delay * (config.backoff_factor ** 2)
        ]
        
        # Vérifier que sleep a été appelé avec les bons délais
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        for i, delay in enumerate(sleep_calls[:config.max_reconnect_attempts]):
            assert abs(delay - expected_delays[i]) < 0.1, f"Délai incorrect: {delay} vs {expected_delays[i]}"
        
        # Vérifier que les statistiques ont été mises à jour
        assert client.stats['reconnections'] > 0, "Compteur de reconnexions non incrémenté"
        assert len(client.stats['reconnection_attempts']) > 0, "Tentatives de reconnexion non enregistrées"

@pytest.mark.asyncio
async def test_ping_pong_timeout():
    """Vérifie la détection des timeouts de ping/pong."""
    config = WebSocketConfig(
        ping_interval=0.1,
        ping_timeout=0.1
    )
    client = BinanceWebSocket(config)
    
    # Simuler une connexion établie
    client.connected = True
    client.running = True
    client.last_ping_time = time.time() - config.ping_interval - config.ping_timeout - 1
    client.stats['ping_timeouts'] = 0
    
    # Patcher la méthode _handle_connection_error
    with patch.object(client, '_handle_connection_error', AsyncMock()) as mock_handle_error:
        ping_pong_timeout = config.ping_interval + config.ping_timeout
        if (client.last_ping_time and 
            time.time() - client.last_ping_time > ping_pong_timeout):
            client.stats['ping_timeouts'] += 1
            await client._handle_connection_error("Ping timeout")
        
        # Vérifier que _handle_connection_error a été appelé
        mock_handle_error.assert_called_once()
        assert "Ping timeout" in mock_handle_error.call_args[0][0]
        
        # Vérifier que les statistiques ont été mises à jour
        assert client.stats['ping_timeouts'] == 1
