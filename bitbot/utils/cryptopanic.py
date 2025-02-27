"""
Simple client for the CryptoPanic API
https://cryptopanic.com/api/
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import requests
from requests.exceptions import RequestException


class CryptoPanicClient:
    """Client pour l'API CryptoPanic"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client CryptoPanic
        
        Args:
            api_key: Clé API CryptoPanic. Si non fournie, utilise CRYPTOPANIC_API_KEY depuis les variables d'environnement
        """
        self.api_key = api_key or os.getenv('CRYPTOPANIC_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set CRYPTOPANIC_API_KEY environment variable or pass api_key to constructor")
        
        self.base_url = "https://cryptopanic.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
        })

    def get_news(self, 
                 currencies: Optional[List[str]] = None,
                 regions: Optional[List[str]] = None,
                 kind: str = 'news',
                 filter: str = 'hot',
                 since: Optional[datetime] = None,
                 limit: int = 50) -> Dict:
        """
        Récupère les actualités crypto depuis CryptoPanic
        
        Args:
            currencies: Liste des symboles de cryptomonnaies (ex: ['BTC', 'ETH'])
            regions: Liste des régions (ex: ['en', 'fr'])
            kind: Type de contenu ('news', 'media')
            filter: Filtre ('hot', 'rising', 'important')
            since: Date de début pour les actualités
            limit: Nombre maximum d'actualités à récupérer
            
        Returns:
            Dict contenant les actualités
        """
        params = {
            'limit': min(limit, 50),  # API limite à 50 max
            'kind': kind,
            'filter': filter,
        }
        
        if currencies:
            params['currencies'] = ','.join(currencies)
        if regions:
            params['regions'] = ','.join(regions)
        if since:
            params['since'] = since.isoformat()
            
        try:
            response = self.session.get(f"{self.base_url}/posts/", params=params)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise Exception(f"Error fetching news from CryptoPanic: {str(e)}")

    def get_portfolio(self) -> Dict:
        """Récupère le portfolio de l'utilisateur"""
        try:
            response = self.session.get(f"{self.base_url}/portfolio/")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise Exception(f"Error fetching portfolio from CryptoPanic: {str(e)}")

    def get_market_news(self, market: str = 'bitcoin') -> Dict:
        """
        Récupère les actualités pour un marché spécifique
        
        Args:
            market: Nom du marché (ex: 'bitcoin', 'ethereum')
            
        Returns:
            Dict contenant les actualités du marché
        """
        try:
            response = self.session.get(f"{self.base_url}/markets/{market}/")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise Exception(f"Error fetching market news from CryptoPanic: {str(e)}")


# Exemple d'utilisation:
if __name__ == "__main__":
    client = CryptoPanicClient()
    
    # Récupérer les dernières actualités importantes pour BTC et ETH
    news = client.get_news(
        currencies=['BTC', 'ETH'],
        regions=['en'],
        filter='important',
        since=datetime.now() - timedelta(days=1)
    )
    
    for item in news.get('results', []):
        print(f"[{item['created_at']}] {item['title']}")
        print(f"Source: {item['source']['title']}")
        print(f"URL: {item['url']}\n")
