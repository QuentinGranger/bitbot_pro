"""
Client pour l'API CryptoPanic.

Ce module fournit une interface pour récupérer les actualités liées aux cryptomonnaies
via l'API CryptoPanic et effectuer une analyse de sentiment basée sur ces données.
"""

import aiohttp
import asyncio
import json
import logging
import os
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import time
import random
import urllib.parse

from bitbot.utils.logger import logger
from bitbot.models.market_data import NewsItem, NewsCollection

class CryptoPanicClient:
    """
    Client pour l'API CryptoPanic permettant de récupérer des actualités
    et analyses de sentiment pour les cryptomonnaies.
    """
    
    BASE_URL = "https://cryptopanic.com/api/v1"
    
    def __init__(self, api_key: str = None):
        """
        Initialise le client CryptoPanic.
        
        Args:
            api_key: Clé API pour l'authentification
        """
        # Utiliser la clé API fournie en paramètre, sinon utiliser la valeur par défaut
        self.api_key = api_key or "915117cb34ddead50902329a6132ffa818566c6f"
        
        if not self.api_key:
            logger.warning("API key for CryptoPanic not provided. Some features may be limited.")
            
        # Collection d'actualités pour le cache interne
        self.news_collection = NewsCollection()
        
        # Répertoire de cache
        self.cache_dir = Path("data/cryptopanic")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Durée de validité du cache (en secondes)
        self.cache_ttl = 1800  # 30 minutes
    
    async def get_news_async(self, 
                             currencies: Union[str, List[str]] = None,
                             filter_type: str = None,
                             kind: str = None,
                             regions: Union[str, List[str]] = None,
                             public: bool = True,
                             page: int = 1,
                             force_refresh: bool = False) -> List[NewsItem]:
        """
        Récupère les actualités de façon asynchrone.
        
        Args:
            currencies: Monnaie(s) pour filtrer les actualités
            filter_type: Type de filtre (rising, hot, bullish, bearish, important, saved, lol)
            kind: Type d'actualités (news, media)
            regions: Région(s) des actualités (en, de, nl, etc.)
            public: Si True, utilise l'API publique
            page: Numéro de page
            force_refresh: Force la récupération des données même si le cache est valide
            
        Returns:
            Liste d'articles d'actualités
        """
        # Créer une session avec l'option pour désactiver la vérification SSL
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        conn = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=conn) as session:
            params = {
                'auth_token': self.api_key
            }
            
            # Ajouter les paramètres optionnels
            if public:
                params['public'] = 'true'
                
            if currencies:
                if isinstance(currencies, list):
                    params['currencies'] = ','.join(currencies)
                else:
                    params['currencies'] = currencies
                    
            if filter_type:
                params['filter'] = filter_type
                
            if kind:
                params['kind'] = kind
                
            if regions:
                if isinstance(regions, list):
                    params['regions'] = ','.join(regions)
                else:
                    params['regions'] = regions
                    
            if page > 1:
                params['page'] = str(page)
            
            # Construire l'URL
            url = f"{self.BASE_URL}/posts/"
            
            # Faire la requête
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_news_response(data)
                    else:
                        logger.error(f"Erreur lors de la récupération des actualités: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Erreur lors de la requête à l'API CryptoPanic: {str(e)}")
                return []
    
    def get_news(self, 
                currencies: Union[str, List[str]] = None,
                filter_type: str = None,
                kind: str = None,
                regions: Union[str, List[str]] = None,
                public: bool = True,
                page: int = 1,
                force_refresh: bool = False) -> List[NewsItem]:
        """
        Récupère les actualités (version synchrone).
        
        Args:
            currencies: Monnaie(s) pour filtrer les actualités
            filter_type: Type de filtre (rising, hot, bullish, bearish, important, saved, lol)
            kind: Type d'actualités (news, media)
            regions: Région(s) des actualités (en, de, nl, etc.)
            public: Si True, utilise l'API publique
            page: Numéro de page
            force_refresh: Force la récupération des données même si le cache est valide
            
        Returns:
            Liste d'articles d'actualités
        """
        # Normalisation du paramètre currencies pour le nom du cache
        currency_str = None
        if currencies:
            if isinstance(currencies, list):
                currency_str = '_'.join(sorted(currencies))
            else:
                currency_str = currencies
        
        # Créer la clé de cache
        cache_key = f"news_{currency_str or 'all'}_{filter_type or 'all'}_{regions or 'all'}_{page}"
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        # Vérifier le cache si on ne force pas le rafraîchissement
        if not force_refresh and cache_path.exists():
            # Vérifier l'âge du cache
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < self.cache_ttl:
                logger.debug(f"Utilisation du cache pour CryptoPanic: {cache_key}")
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    return self._process_news_response(cached_data)
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture du cache CryptoPanic: {str(e)}")
        
        try:
            # Créer un événement asyncio et exécuter la requête asynchrone
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Si pas de boucle d'événements, en créer une nouvelle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            result = loop.run_until_complete(
                self.get_news_async(
                    currencies=currencies,
                    filter_type=filter_type,
                    kind=kind,
                    regions=regions,
                    public=public,
                    page=page,
                    force_refresh=force_refresh
                )
            )
            
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des actualités: {str(e)}")
            
            # En cas d'erreur avec l'API, utiliser les données de démonstration
            logger.info("Génération de données d'actualités de démonstration pour CryptoPanic")
            
            # Définir la liste des devises à inclure
            currencies_list = []
            if currency_str:
                currencies_list = currency_str.split(',')
            elif isinstance(currencies, list):
                currencies_list = currencies
            
            # Si aucune devise n'est spécifiée, utiliser BTC par défaut
            if not currencies_list:
                currencies_list = ['BTC']
            
            # Générer des données simulées
            demo_data = self._generate_demo_data(currencies_list, page)
            
            # Sauvegarder dans le cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(demo_data, f, ensure_ascii=False, indent=2)
                
            return self._process_news_response(demo_data)
    
    def _generate_demo_data(self, currencies: List[str], page: int = 1) -> Dict:
        """
        Génère des données d'actualités de démonstration.
        
        Args:
            currencies: Liste des devises à inclure dans les actualités
            page: Numéro de page (pour simuler la pagination)
            
        Returns:
            Données simulées au format de réponse de l'API
        """
        # Liste des titres de démonstration pour chaque sentiment
        positive_titles = [
            "Les analystes prédisent une hausse de 20% pour {currency} d'ici la fin du mois",
            "Nouvelle adoption institutionnelle majeure pour {currency}",
            "{currency} franchit un nouveau record historique",
            "La mise à jour du réseau {currency} améliore considérablement les performances",
            "Un grand détaillant commence à accepter {currency} comme moyen de paiement"
        ]
        
        negative_titles = [
            "Chute brutale du cours de {currency} suite à des annonces réglementaires",
            "Faille de sécurité découverte dans un portefeuille {currency}",
            "Le gouvernement envisage de taxer lourdement les gains en {currency}",
            "Un important exchange suspend temporairement les transactions en {currency}",
            "Les mineurs de {currency} font face à des coûts d'électricité croissants"
        ]
        
        neutral_titles = [
            "Rapport trimestriel sur les performances de {currency}",
            "Comparaison entre {currency} et les autres cryptomonnaies",
            "Guide d'achat et de stockage de {currency}",
            "L'histoire de {currency} : des débuts à aujourd'hui",
            "Analyse technique : les niveaux à surveiller pour {currency}"
        ]
        
        # Sources d'actualités simulées
        sources = [
            {"domain": "coindesk.com", "title": "CoinDesk"},
            {"domain": "cointelegraph.com", "title": "CoinTelegraph"},
            {"domain": "cryptonews.com", "title": "CryptoNews"},
            {"domain": "bitcoinist.com", "title": "Bitcoinist"},
            {"domain": "decrypt.co", "title": "Decrypt"},
            {"domain": "newsbtc.com", "title": "NewsBTC"}
        ]
        
        # Génération des résultats
        results = []
        base_time = datetime.now() - timedelta(days=7)
        items_per_page = 10
        
        for i in range(items_per_page):
            # Déterminer si cette actualité concerne toutes les devises ou une seule
            if len(currencies) > 1 and i % 3 == 0:
                # Actualité concernant plusieurs devises
                news_currencies = [{"code": c} for c in currencies]
                currency_for_title = currencies[0]  # Utiliser la première pour le titre
            else:
                # Actualité concernant une seule devise
                currency_idx = i % len(currencies)
                news_currencies = [{"code": currencies[currency_idx]}]
                currency_for_title = currencies[currency_idx]
            
            # Déterminer le sentiment de l'actualité
            sentiment_rnd = i % 5
            if sentiment_rnd < 2:
                kind = "positive"
                titles = positive_titles
            elif sentiment_rnd < 4:
                kind = "negative"
                titles = negative_titles
            else:
                kind = None  # neutre
                titles = neutral_titles
            
            # Choisir un titre aléatoire
            title_idx = (i + page) % len(titles)
            title = titles[title_idx].format(currency=currency_for_title)
            
            # Choisir une source aléatoire
            source_idx = (i + page) % len(sources)
            source = sources[source_idx]
            
            # Générer des votes simulés
            votes = None
            if i % 2 == 0:  # Seulement certains articles ont des votes
                if kind == "positive":
                    votes = {
                        "positive": 10 + i,
                        "negative": 2,
                        "important": 5,
                        "liked": 8,
                        "disliked": 1
                    }
                elif kind == "negative":
                    votes = {
                        "positive": 3,
                        "negative": 12 + i,
                        "important": 4,
                        "liked": 2,
                        "disliked": 7
                    }
                else:
                    votes = {
                        "positive": 4,
                        "negative": 4,
                        "important": 2,
                        "liked": 3,
                        "disliked": 3
                    }
            
            # Générer la date de publication
            published_at = (base_time + timedelta(hours=i*3 + (page-1)*30)).isoformat()
            
            # Générer l'URL
            url = f"https://{source['domain']}/news/{i+page*10}/{title.replace(' ', '-').lower()}"
            
            # Créer l'item d'actualité
            news_item = {
                "id": i + page * 100,
                "title": title,
                "published_at": published_at,
                "url": url,
                "source": source,
                "domain": source["domain"],
                "currencies": news_currencies,
                "kind": kind,
                "votes": votes
            }
            
            results.append(news_item)
        
        # Construire la réponse complète
        response = {
            "count": 50,  # Nombre total simulé
            "next": f"https://cryptopanic.com/api/v1/posts/?public=true&page={page+1}" if page < 5 else None,
            "previous": f"https://cryptopanic.com/api/v1/posts/?public=true&page={page-1}" if page > 1 else None,
            "results": results
        }
        
        return response
    
    def _process_news_response(self, data: Dict) -> List[NewsItem]:
        """
        Traite la réponse de l'API et crée des objets NewsItem.
        
        Args:
            data: Réponse JSON de l'API
            
        Returns:
            Liste d'objets NewsItem
        """
        news_items = []
        
        if 'results' not in data:
            logger.warning("Format de réponse CryptoPanic inattendu: 'results' non trouvé")
            return news_items
        
        for item_data in data['results']:
            try:
                news_item = NewsItem.from_cryptopanic(item_data)
                news_items.append(news_item)
                
                # Ajouter à la collection interne
                self.news_collection.add_item(news_item)
            except Exception as e:
                logger.error(f"Erreur lors du traitement d'un article: {str(e)}")
        
        return news_items
    
    def get_sentiment_analysis(self, 
                               currency: str = 'BTC', 
                               days: int = 1, 
                               force_refresh: bool = False) -> Dict[str, float]:
        """
        Récupère et analyse le sentiment des actualités pour une cryptomonnaie.
        
        Args:
            currency: Code de la cryptomonnaie
            days: Nombre de jours à analyser
            force_refresh: Force la récupération des actualités même si le cache est valide
            
        Returns:
            Dictionnaire contenant l'analyse de sentiment
        """
        # Si peu d'articles dans la collection, récupérer plus d'actualités
        relevant_news = self.news_collection.get_relevant_news(currency, days)
        if len(relevant_news) < 5:
            for page in range(1, 3):  # Limiter à 2 pages pour éviter d'être bloqué par l'API
                self.get_news(
                    currencies=currency,
                    page=page,
                    force_refresh=force_refresh
                )
        
        # Analyser le sentiment
        sentiment = self.news_collection.get_sentiment_analysis(currency, days)
        
        return sentiment
    
    def get_sentiment_signal(self,
                            currency: str = 'BTC',
                            days: int = 1,
                            force_refresh: bool = False) -> float:
        """
        Génère un signal de trading basé sur l'analyse de sentiment.
        
        Args:
            currency: Code de la cryptomonnaie
            days: Nombre de jours à analyser
            force_refresh: Force la récupération des actualités même si le cache est valide
            
        Returns:
            Signal entre -1.0 (très bearish) et 1.0 (très bullish)
        """
        # Récupérer l'analyse de sentiment
        sentiment = self.get_sentiment_analysis(currency, days, force_refresh)
        
        # Calculer le signal
        return self.news_collection.get_sentiment_signal(currency, days)
