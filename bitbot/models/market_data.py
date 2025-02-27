"""
Modèles de données pour les informations de marché.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Union, Any
from decimal import Decimal
import pandas as pd
import numpy as np
from datetime import timedelta

@dataclass
class Kline:
    """Chandelier (OHLCV)."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal
    trades: int
    taker_buy_volume: Decimal
    taker_buy_quote_volume: Decimal
    interval: str
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Kline':
        """
        Crée un Kline depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Kline
        """
        k = message['k']
        return cls(
            timestamp=datetime.fromtimestamp(k['t'] / 1000),
            open=Decimal(str(k['o'])),
            high=Decimal(str(k['h'])),
            low=Decimal(str(k['l'])),
            close=Decimal(str(k['c'])),
            volume=Decimal(str(k['v'])),
            close_time=datetime.fromtimestamp(k['T'] / 1000),
            quote_volume=Decimal(str(k['q'])),
            trades=k['n'],
            taker_buy_volume=Decimal(str(k['V'])),
            taker_buy_quote_volume=Decimal(str(k['Q'])),
            interval=k['i']
        )

@dataclass
class Trade:
    """Transaction."""
    timestamp: datetime
    symbol: str
    id: int
    price: Decimal
    quantity: Decimal
    buyer_maker: bool
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Trade':
        """
        Crée un Trade depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Trade
        """
        return cls(
            timestamp=datetime.fromtimestamp(message['T'] / 1000),
            symbol=message['s'],
            id=message['t'],
            price=Decimal(str(message['p'])),
            quantity=Decimal(str(message['q'])),
            buyer_maker=message['m']
        )

@dataclass
class OrderBook:
    """Carnet d'ordres."""
    symbol: str
    bids: Dict[float, float]  # prix -> quantité
    asks: Dict[float, float]  # prix -> quantité
    update_id: Optional[int] = None
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = {}
        self.asks = {}
        self.update_id = None
    
    def get_best_bid(self) -> tuple[float, float]:
        """
        Retourne le meilleur bid.
        
        Returns:
            (prix, quantité)
        """
        if not self.bids:
            return (0, 0)
        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])
    
    def get_best_ask(self) -> tuple[float, float]:
        """
        Retourne le meilleur ask.
        
        Returns:
            (prix, quantité)
        """
        if not self.asks:
            return (float('inf'), 0)
        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])
    
    def get_spread(self) -> float:
        """
        Calcule le spread.
        
        Returns:
            Spread en pourcentage
        """
        best_bid = self.get_best_bid()[0]
        best_ask = self.get_best_ask()[0]
        
        if best_bid == 0 or best_ask == float('inf'):
            return float('inf')
        
        return (best_ask - best_bid) / best_bid * 100

@dataclass
class Ticker:
    """Ticker 24h."""
    symbol: str
    price_change: Decimal
    price_change_percent: Decimal
    weighted_avg_price: Decimal
    prev_close_price: Decimal
    last_price: Decimal
    last_qty: Decimal
    bid_price: Decimal
    ask_price: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    open_time: datetime
    close_time: datetime
    first_id: int
    last_id: int
    count: int
    
    @classmethod
    def from_ws_message(cls, message: Dict) -> 'Ticker':
        """
        Crée un Ticker depuis un message WebSocket.
        
        Args:
            message: Message du WebSocket
        
        Returns:
            Instance de Ticker
        """
        return cls(
            symbol=message['s'],
            price_change=Decimal(str(message['p'])),
            price_change_percent=Decimal(str(message['P'])),
            weighted_avg_price=Decimal(str(message['w'])),
            prev_close_price=Decimal(str(message['x'])),
            last_price=Decimal(str(message['c'])),
            last_qty=Decimal(str(message['Q'])),
            bid_price=Decimal(str(message['b'])),
            ask_price=Decimal(str(message['a'])),
            open_price=Decimal(str(message['o'])),
            high_price=Decimal(str(message['h'])),
            low_price=Decimal(str(message['l'])),
            volume=Decimal(str(message['v'])),
            quote_volume=Decimal(str(message['q'])),
            open_time=datetime.fromtimestamp(message['O'] / 1000),
            close_time=datetime.fromtimestamp(message['C'] / 1000),
            first_id=message['F'],
            last_id=message['L'],
            count=message['n']
        )

@dataclass
class FearGreedIndex:
    """Indice de peur et d'avidité pour mesurer le sentiment du marché crypto."""
    timestamp: datetime
    value: int
    classification: str
    time_until_update: Optional[int] = None
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'FearGreedIndex':
        """
        Crée un FearGreedIndex depuis une réponse de l'API Alternative.me.
        
        Args:
            data: Données de l'API
        
        Returns:
            Instance de FearGreedIndex
        """
        return cls(
            timestamp=datetime.fromtimestamp(int(data['timestamp'])),
            value=int(data['value']),
            classification=data['value_classification'],
            time_until_update=int(data.get('time_until_update', 0)) if data.get('time_until_update') else None
        )
    
    def get_sentiment_score(self) -> float:
        """
        Convertit la valeur de l'indice en score de sentiment normalisé entre -1 et 1.
        -1 représente la peur extrême (0), 0 est neutre (50), 1 est l'avidité extrême (100).
        
        Returns:
            Score de sentiment normalisé
        """
        return (self.value - 50) / 50.0

class GoogleTrendsData:
    """
    Données de Google Trends pour un mot-clé spécifique.
    """
    
    def __init__(self, keyword: str, data: pd.DataFrame, related_queries: Dict = None, related_topics: Dict = None):
        """
        Initialise les données Google Trends.
        
        Args:
            keyword: Mot-clé pour lequel les données ont été récupérées
            data: DataFrame contenant les données d'intérêt au fil du temps
            related_queries: Requêtes associées (top et rising)
            related_topics: Sujets associés (top et rising)
        """
        self.keyword = keyword
        self.data = data
        self.related_queries = related_queries or {}
        self.related_topics = related_topics or {}
    
    def get_normalized_interest(self) -> pd.DataFrame:
        """
        Retourne les données d'intérêt normalisées.
        
        Returns:
            DataFrame avec les données d'intérêt normalisées
        """
        if self.data.empty:
            return pd.DataFrame()
        
        # Créer une copie pour ne pas modifier l'original
        df = self.data.copy()
        
        # Vérifier si la colonne du mot-clé existe
        if self.keyword in df.columns:
            # Normaliser les valeurs entre 0 et 1
            interest_values = df[self.keyword].values
            
            # Gérer les valeurs invalides
            valid_mask = ~(np.isnan(interest_values) | np.isinf(interest_values))
            if valid_mask.sum() > 0:
                min_val = np.min(interest_values[valid_mask])
                max_val = np.max(interest_values[valid_mask])
                
                # Éviter la division par zéro
                if max_val > min_val:
                    normalized = (interest_values - min_val) / (max_val - min_val)
                    # Remplacer les valeurs invalides par NaN
                    normalized[~valid_mask] = np.nan
                else:
                    normalized = np.zeros_like(interest_values)
                    normalized[~valid_mask] = np.nan
            else:
                # Si toutes les valeurs sont invalides
                normalized = np.zeros_like(interest_values)
                normalized[:] = np.nan
                
            df['normalized_interest'] = normalized
        
        return df
    
    def get_momentum_signal(self, window: int = 14) -> pd.DataFrame:
        """
        Calcule un signal de momentum basé sur les tendances Google.
        
        Args:
            window: Taille de la fenêtre pour le calcul du momentum
            
        Returns:
            DataFrame avec le signal de momentum
        """
        if self.data.empty:
            return pd.DataFrame()
        
        # Obtenir les données normalisées
        df = self.get_normalized_interest()
        
        if 'normalized_interest' not in df.columns:
            return pd.DataFrame()
        
        # Calculer la moyenne mobile
        df['sma'] = df['normalized_interest'].rolling(window=window).mean()
        
        # Calculer le momentum (différence entre la valeur actuelle et la moyenne mobile)
        df['momentum'] = df['normalized_interest'] - df['sma']
        
        # Normaliser le momentum entre -1 et 1
        valid_mask = ~(np.isnan(df['momentum']) | np.isinf(df['momentum']))
        if valid_mask.sum() > 0:
            max_abs = np.max(np.abs(df['momentum'][valid_mask]))
            if max_abs > 0:
                df['momentum_signal'] = df['momentum'] / max_abs
            else:
                df['momentum_signal'] = 0
        else:
            df['momentum_signal'] = np.nan
        
        return df
    
    def get_rising_topics(self, min_value: int = 0) -> List[Dict]:
        """
        Retourne les sujets en hausse filtrés par valeur minimale.
        
        Args:
            min_value: Valeur minimale pour filtrer les sujets
            
        Returns:
            Liste des sujets en hausse
        """
        if not self.related_topics or 'rising' not in self.related_topics:
            return []
        
        rising = self.related_topics.get('rising', [])
        if isinstance(rising, pd.DataFrame):
            rising = rising.to_dict('records')
        
        return [topic for topic in rising if topic.get('value', 0) >= min_value]
    
    def get_rising_queries(self, min_value: int = 0) -> List[Dict]:
        """
        Retourne les requêtes en hausse filtrées par valeur minimale.
        
        Args:
            min_value: Valeur minimale pour filtrer les requêtes
            
        Returns:
            Liste des requêtes en hausse
        """
        if not self.related_queries or 'rising' not in self.related_queries:
            return []
        
        rising = self.related_queries.get('rising', [])
        if isinstance(rising, pd.DataFrame):
            rising = rising.to_dict('records')
        
        return [query for query in rising if query.get('value', 0) >= min_value]

class NewsItem:
    """
    Représente un article d'actualité de CryptoPanic ou d'autres sources.
    """
    
    def __init__(self, 
                 id: str, 
                 title: str, 
                 url: str, 
                 published_at: datetime, 
                 source: str, 
                 currencies: List[str] = None, 
                 votes: Dict[str, int] = None, 
                 sentiment: str = None):
        """
        Initialise un article d'actualité.
        
        Args:
            id: Identifiant unique de l'article
            title: Titre de l'article
            url: URL de l'article
            published_at: Date et heure de publication
            source: Source de l'article (site, média)
            currencies: Liste des cryptomonnaies mentionnées dans l'article
            votes: Votes reçus par l'article (positifs, négatifs, importants, etc.)
            sentiment: Sentiment global de l'article (positif, négatif, neutre)
        """
        self.id = id
        self.title = title
        self.url = url
        self.published_at = published_at
        self.source = source
        self.currencies = currencies or []
        self.votes = votes or {}
        self.sentiment = sentiment
    
    @classmethod
    def from_cryptopanic(cls, data: Dict) -> 'NewsItem':
        """
        Crée un objet NewsItem à partir des données de CryptoPanic.
        
        Args:
            data: Données brutes de l'API CryptoPanic
            
        Returns:
            NewsItem créé à partir des données
        """
        # Extraire les devises mentionnées
        currencies = []
        if 'currencies' in data and data['currencies']:
            currencies = [c['code'] for c in data['currencies']]
        
        # Extraire les votes s'ils existent
        votes = {}
        if 'votes' in data and data['votes']:
            votes = {
                'positive': data['votes'].get('positive', 0),
                'negative': data['votes'].get('negative', 0),
                'important': data['votes'].get('important', 0),
                'liked': data['votes'].get('liked', 0),
                'disliked': data['votes'].get('disliked', 0),
                'lol': data['votes'].get('lol', 0),
                'toxic': data['votes'].get('toxic', 0),
                'saved': data['votes'].get('saved', 0)
            }
        
        # Extraire le sentiment si disponible
        sentiment = None
        if 'kind' in data:
            sentiment = data['kind']  # 'positive', 'negative' ou None
        
        return cls(
            id=str(data.get('id', '')),
            title=data.get('title', ''),
            url=data.get('url', ''),
            published_at=datetime.fromisoformat(data.get('published_at', '').replace('Z', '+00:00')),
            source=data.get('source', {}).get('title', 'Unknown'),
            currencies=currencies,
            votes=votes,
            sentiment=sentiment
        )
    
    def get_sentiment_score(self) -> float:
        """
        Calcule un score de sentiment basé sur les votes et le sentiment.
        
        Returns:
            Score entre -1.0 (très négatif) et 1.0 (très positif)
        """
        base_score = 0.0
        
        # Utiliser le sentiment si disponible
        if self.sentiment == 'positive':
            base_score = 0.5
        elif self.sentiment == 'negative':
            base_score = -0.5
        
        # Ajuster le score en fonction des votes
        vote_score = 0.0
        total_votes = 0
        
        if self.votes:
            positive_votes = self.votes.get('positive', 0) + self.votes.get('liked', 0)
            negative_votes = self.votes.get('negative', 0) + self.votes.get('disliked', 0) + self.votes.get('toxic', 0)
            important_votes = self.votes.get('important', 0)
            
            total_votes = positive_votes + negative_votes
            
            if total_votes > 0:
                vote_score = (positive_votes - negative_votes) / total_votes
                
                # Les votes "importants" augmentent l'ampleur du score sans changer le signe
                if important_votes > 0:
                    vote_multiplier = 1 + (important_votes / (total_votes + important_votes)) * 0.5
                    vote_score *= vote_multiplier
        
        # Combiner le score de base et celui des votes
        if total_votes > 0:
            final_score = (base_score + vote_score) / 2
        else:
            final_score = base_score
        
        # Assurer que le score est entre -1 et 1
        return max(-1.0, min(1.0, final_score))
    
    def is_relevant_to(self, currency: str) -> bool:
        """
        Vérifie si l'article est pertinent pour une monnaie donnée.
        
        Args:
            currency: Code de la monnaie à vérifier
            
        Returns:
            True si l'article mentionne la monnaie, False sinon
        """
        if not self.currencies:
            return False
        
        return currency.upper() in [c.upper() for c in self.currencies]
    
    def __str__(self) -> str:
        """Représentation textuelle de l'article."""
        sentiment_str = f" [{self.sentiment}]" if self.sentiment else ""
        return f"{self.published_at.strftime('%Y-%m-%d %H:%M')}{sentiment_str}: {self.title} ({self.source})"


class NewsCollection:
    """
    Collection d'articles d'actualité avec des méthodes pour analyser le sentiment.
    """
    
    def __init__(self, news_items: List[NewsItem] = None):
        """
        Initialise une collection d'articles.
        
        Args:
            news_items: Liste d'articles (optionnelle)
        """
        self.news_items = news_items or []
    
    def add_item(self, item: NewsItem):
        """
        Ajoute un article à la collection.
        
        Args:
            item: Article à ajouter
        """
        self.news_items.append(item)
    
    def get_relevant_news(self, currency: str, days: int = 1) -> List[NewsItem]:
        """
        Récupère les articles pertinents pour une monnaie dans une période donnée.
        
        Args:
            currency: Code de la monnaie
            days: Nombre de jours à considérer
            
        Returns:
            Liste des articles pertinents
        """
        cutoff_time = datetime.now().replace(tzinfo=None) - timedelta(days=days)
        
        relevant_news = []
        for item in self.news_items:
            # S'assurer que les deux datetimes sont dans le même format (avec ou sans timezone)
            item_time = item.published_at
            if item_time.tzinfo is not None:
                item_time = item_time.replace(tzinfo=None)
                
            if item.is_relevant_to(currency) and item_time >= cutoff_time:
                relevant_news.append(item)
                
        return relevant_news
    
    def get_sentiment_analysis(self, currency: str = None, days: int = 1) -> Dict[str, float]:
        """
        Analyse le sentiment global des articles, filtré par monnaie si spécifié.
        
        Args:
            currency: Code de la monnaie (optionnel)
            days: Nombre de jours à considérer
            
        Returns:
            Dictionnaire avec des statistiques de sentiment
        """
        # Filtrer les articles pertinents
        if currency:
            relevant_items = self.get_relevant_news(currency, days)
        else:
            cutoff_time = datetime.now() - timedelta(days=days)
            relevant_items = [item for item in self.news_items if item.published_at >= cutoff_time]
        
        if not relevant_items:
            return {
                'count': 0,
                'average_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        # Calculer les scores et les ratios
        scores = [item.get_sentiment_score() for item in relevant_items]
        positive_count = sum(1 for s in scores if s > 0.2)
        negative_count = sum(1 for s in scores if s < -0.2)
        neutral_count = len(scores) - positive_count - negative_count
        
        return {
            'count': len(relevant_items),
            'average_score': sum(scores) / len(scores),
            'positive_ratio': positive_count / len(scores),
            'negative_ratio': negative_count / len(scores),
            'neutral_ratio': neutral_count / len(scores)
        }
    
    def get_sentiment_signal(self, currency: str = None, days: int = 1) -> float:
        """
        Génère un signal de trading basé sur l'analyse de sentiment.
        
        Args:
            currency: Code de la monnaie (optionnel)
            days: Nombre de jours à considérer
            
        Returns:
            Signal entre -1.0 (très bearish) et 1.0 (très bullish)
        """
        sentiment = self.get_sentiment_analysis(currency, days)
        
        if sentiment['count'] == 0:
            return 0.0
        
        # Pondérer le score moyen par le nombre d'articles
        # Plus il y a d'articles, plus le signal est fort
        weight = min(1.0, sentiment['count'] / 20)  # Plafond à 20 articles pour un poids maximum
        
        return sentiment['average_score'] * weight

@dataclass
class Signal:
    """Signal de trading généré par une stratégie."""
    
    symbol: str
    timestamp: datetime
    signal_type: str  # Type de signal (ex: 'TREND', 'BREAKOUT', 'NEWS', etc.)
    direction: float  # Direction du signal (-1.0 à 1.0, où -1 est très baissier, 1 est très haussier)
    strength: float  # Force du signal (0.0 à 1.0)
    source: str  # Source du signal (ex: 'rsi', 'news', 'trends', etc.)
    metadata: Dict[str, Any] = None  # Métadonnées supplémentaires
    
    def __post_init__(self):
        """Valide les données après l'initialisation."""
        # S'assurer que direction est entre -1 et 1
        self.direction = max(-1.0, min(1.0, self.direction))
        
        # S'assurer que strength est entre 0 et 1
        self.strength = max(0.0, min(1.0, self.strength))
        
        # Initialiser metadata si non fourni
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        """Représentation textuelle du signal."""
        direction_str = "ACHAT" if self.direction > 0 else "VENTE"
        return f"Signal {direction_str} ({self.signal_type}) pour {self.symbol} avec force {self.strength:.2f} - {self.timestamp}"

class MarketData:
    """
    Classe représentant les données de marché pour un symbole.
    
    Cette classe contient les prix historiques (OHLCV), les indicateurs
    techniques, et des métadonnées sur le marché.
    """
    
    def __init__(self, symbol: str, timeframe: str = "1h"):
        """
        Initialise les données de marché.
        
        Args:
            symbol: Symbole de la paire de trading
            timeframe: Intervalle de temps des bougies
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.ohlcv = pd.DataFrame()  # Données OHLCV
        self.indicators = {}  # Indicateurs techniques
        self.metadata = {}  # Métadonnées
        self.signals = []  # Signaux générés
    
    def update_from_klines(self, klines: List[Kline]) -> None:
        """
        Met à jour les données OHLCV à partir d'une liste de Klines.
        
        Args:
            klines: Liste d'objets Kline
        """
        # Convertir les klines en DataFrame
        data = []
        
        for kline in klines:
            data.append({
                'timestamp': kline.timestamp,
                'open': float(kline.open),
                'high': float(kline.high),
                'low': float(kline.low),
                'close': float(kline.close),
                'volume': float(kline.volume)
            })
        
        if data:
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Mettre à jour les données
            self.ohlcv = df
    
    def add_indicator(self, name: str, data: Union[pd.Series, pd.DataFrame, List[float], float]) -> None:
        """
        Ajoute un indicateur technique aux données de marché.
        
        Args:
            name: Nom de l'indicateur
            data: Données de l'indicateur
        """
        self.indicators[name] = data
    
    def add_signal(self, signal: Signal) -> None:
        """
        Ajoute un signal aux données de marché.
        
        Args:
            signal: Signal à ajouter
        """
        self.signals.append(signal)
    
    def get_latest_price(self) -> Optional[float]:
        """
        Récupère le dernier prix disponible.
        
        Returns:
            Dernier prix ou None si pas de données
        """
        if self.ohlcv.empty:
            return None
        
        return self.ohlcv['close'].iloc[-1]
    
    def get_price_change(self, periods: int = 1) -> Optional[float]:
        """
        Calcule la variation de prix sur un nombre de périodes.
        
        Args:
            periods: Nombre de périodes pour le calcul
            
        Returns:
            Variation en pourcentage ou None si pas assez de données
        """
        if self.ohlcv.empty or len(self.ohlcv) <= periods:
            return None
        
        last_price = self.ohlcv['close'].iloc[-1]
        prev_price = self.ohlcv['close'].iloc[-periods-1]
        
        return ((last_price - prev_price) / prev_price) * 100.0
    
    def to_dict(self) -> Dict:
        """
        Convertit les données de marché en dictionnaire.
        
        Returns:
            Dictionnaire des données de marché
        """
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'latest_price': self.get_latest_price(),
            'price_change_24h': self.get_price_change(24),
            'indicators': {k: str(v) for k, v in self.indicators.items()},
            'metadata': self.metadata,
            'signals': [str(signal) for signal in self.signals]
        }
