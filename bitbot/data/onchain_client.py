"""
Module pour interagir avec diverses API gratuites de données blockchain on-chain.

Ce module fournit des fonctionnalités pour récupérer des données on-chain
depuis diverses API gratuites, notamment CoinGecko, CoinLore et BlockCypher.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union, Any
import logging
from dotenv import load_dotenv
import json

from bitbot.utils.logger import logger

class OnChainClient:
    """
    Client pour interagir avec diverses API gratuites de données blockchain.
    
    Cette classe permet de récupérer des données on-chain depuis diverses API gratuites,
    et de calculer des métriques comme le ratio MVRV approximatif.
    """
    
    # URL des API
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    COINLORE_BASE_URL = "https://api.coinlore.net/api"
    BLOCKCHAIN_INFO_URL = "https://blockchain.info"
    BLOCKCHAIR_BASE_URL = "https://api.blockchair.com"
    BLOCKCYPHER_BASE_URL = "https://api.blockcypher.com/v1"
    
    # Limites de taux pour les API gratuites
    COINGECKO_RATE_LIMIT = 30  # requêtes par minute avec Demo API Key
    COINLORE_RATE_LIMIT = 30   # requêtes par minute (estimation conservative)
    
    def __init__(self):
        """Initialise le client de données on-chain."""
        # Charger les variables d'environnement
        load_dotenv()
        
        # Récupérer la clé API CoinGecko
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        
        if not self.coingecko_api_key:
            logger.warning("Aucune clé API CoinGecko trouvée. Utilisation de CoinLore comme alternative.")
        else:
            logger.info("Clé API CoinGecko trouvée.")
        
        self.last_request_time = {
            'coingecko': 0,
            'coinlore': 0,
            'blockchain_info': 0,
            'blockchair': 0,
            'blockcypher': 0
        }
        
        logger.info("Client de données on-chain initialisé")
    
    def _respect_rate_limit(self, api: str):
        """
        Respecte les limites de taux pour les API gratuites.
        
        Args:
            api: Nom de l'API.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time[api]
        
        if api == 'coingecko' and elapsed < 60/self.COINGECKO_RATE_LIMIT:
            sleep_time = 60/self.COINGECKO_RATE_LIMIT - elapsed
            logger.debug(f"Respect de la limite de taux pour {api}, attente de {sleep_time:.2f} secondes")
            time.sleep(sleep_time)
        
        if api == 'coinlore' and elapsed < 60/self.COINLORE_RATE_LIMIT:
            sleep_time = 60/self.COINLORE_RATE_LIMIT - elapsed
            logger.debug(f"Respect de la limite de taux pour {api}, attente de {sleep_time:.2f} secondes")
            time.sleep(sleep_time)
        
        self.last_request_time[api] = time.time()
    
    def _make_coingecko_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Effectue une requête à l'API CoinGecko.
        
        Args:
            endpoint: Point de terminaison de l'API.
            params: Paramètres de la requête.
            
        Returns:
            Réponse de l'API sous forme de dictionnaire.
            
        Raises:
            Exception: Si la requête échoue.
        """
        self._respect_rate_limit('coingecko')
        
        url = f"{self.COINGECKO_BASE_URL}/{endpoint}"
        
        # Préparer les paramètres
        if params is None:
            params = {}
        
        # Ajouter la clé API à l'en-tête (méthode recommandée)
        headers = {}
        if self.coingecko_api_key:
            headers = {'x-cg-demo-api-key': self.coingecko_api_key}
            
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API CoinGecko: {e}")
            if response.status_code == 429:
                logger.error("Limite de requêtes CoinGecko atteinte.")
            elif response.status_code == 401:
                logger.error("Clé API CoinGecko invalide ou non fournie.")
            raise
    
    def _make_coinlore_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Effectue une requête à l'API CoinLore.
        
        Args:
            endpoint: Point de terminaison de l'API.
            params: Paramètres de la requête.
            
        Returns:
            Réponse de l'API sous forme de dictionnaire.
            
        Raises:
            Exception: Si la requête échoue.
        """
        self._respect_rate_limit('coinlore')
        
        url = f"{self.COINLORE_BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API CoinLore: {e}")
            raise
    
    def _make_blockchain_info_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Effectue une requête à l'API Blockchain.info.
        
        Args:
            endpoint: Point de terminaison de l'API.
            params: Paramètres de la requête.
            
        Returns:
            Réponse de l'API sous forme de dictionnaire.
            
        Raises:
            Exception: Si la requête échoue.
        """
        self._respect_rate_limit('blockchain_info')
        
        url = f"{self.BLOCKCHAIN_INFO_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API Blockchain.info: {e}")
            raise
    
    def _make_blockcypher_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Effectue une requête à l'API BlockCypher.
        
        Args:
            endpoint: Point de terminaison de l'API.
            params: Paramètres de la requête.
            
        Returns:
            Réponse de l'API sous forme de dictionnaire.
            
        Raises:
            Exception: Si la requête échoue.
        """
        self._respect_rate_limit('blockcypher')
        
        url = f"{self.BLOCKCYPHER_BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API BlockCypher: {e}")
            raise
    
    def get_market_data(self, asset: str, days: int = 365) -> pd.DataFrame:
        """
        Récupère les données historiques de marché pour un actif.
        
        Args:
            asset: Symbole de l'actif (BTC, ETH, etc.).
            days: Nombre de jours de données à récupérer.
            
        Returns:
            DataFrame contenant les données de marché.
        """
        asset = asset.lower()
        
        # Mapping des symboles aux identifiants
        asset_ids = {
            'btc': '90',
            'eth': '80',
            'ltc': '1',
            'xrp': '58',
            'bch': '2321',
            'bnb': '2710',
            'ada': '257',
            'sol': '48543',
            'dot': '42159',
            'doge': '2'
        }
        
        try:
            logger.info(f"Récupération des données de marché pour {asset}")
            
            # Si nous utilisons CoinGecko et que nous avons une clé API
            if self.coingecko_api_key:
                try:
                    # Correspondance entre les symboles et les identifiants CoinGecko
                    coin_ids = {
                        'btc': 'bitcoin',
                        'eth': 'ethereum',
                        'ltc': 'litecoin',
                        'xrp': 'ripple',
                        'bch': 'bitcoin-cash',
                        'bnb': 'binancecoin',
                        'ada': 'cardano',
                        'sol': 'solana',
                        'dot': 'polkadot',
                        'doge': 'dogecoin'
                    }
                    
                    # Vérifier si le symbole est pris en charge
                    if asset not in coin_ids:
                        logger.error(f"Symbole d'actif {asset} non pris en charge par CoinGecko")
                        raise ValueError(f"Symbole d'actif {asset} non pris en charge")
                    
                    coin_id = coin_ids[asset]
                    
                    # Récupérer les données de marché
                    # Pour les calculs du MVRV, nous avons besoin de données historiques plus longues
                    # pour estimer correctement la capitalisation réalisée
                    extended_days = min(days * 4, 2000)  # Extension pour le calcul du realized cap
                    
                    params = {
                        'vs_currency': 'usd',
                        'days': extended_days,
                        'interval': 'daily'
                    }
                    
                    market_data = self._make_coingecko_request(f"coins/{coin_id}/market_chart", params)
                    
                    # Traitement des données
                    prices = market_data.get('prices', [])
                    market_caps = market_data.get('market_caps', [])
                    total_volumes = market_data.get('total_volumes', [])
                    
                    if not prices or not market_caps or not total_volumes:
                        logger.warning(f"Données de marché incomplètes pour {asset} sur CoinGecko")
                        raise ValueError("Données incomplètes")
                    
                    # Création du DataFrame
                    data = []
                    for i in range(min(len(prices), len(market_caps), len(total_volumes))):
                        timestamp = datetime.fromtimestamp(prices[i][0] / 1000)
                        price = prices[i][1]
                        market_cap = market_caps[i][1]
                        volume = total_volumes[i][1]
                        
                        data.append({
                            'timestamp': timestamp,
                            'price': price,
                            'market_cap': market_cap,
                            'volume': volume
                        })
                    
                    df = pd.DataFrame(data)
                    
                    # Filtrer pour le nombre de jours demandé
                    cutoff_date = datetime.now() - timedelta(days=days)
                    df = df[df['timestamp'] >= cutoff_date]
                    
                    return df
                
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des données de marché via CoinGecko: {e}")
                    # Nous essaierons CoinLore en cas d'échec
            
            # Utiliser CoinLore comme alternative
            logger.info(f"Utilisation de CoinLore comme source de données pour {asset}")
            
            # Vérifier si le symbole est pris en charge
            if asset not in asset_ids:
                logger.error(f"Symbole d'actif {asset} non pris en charge par CoinLore")
                raise ValueError(f"Symbole d'actif {asset} non pris en charge")
            
            coin_id = asset_ids[asset]
            
            # Récupérer les données actuelles de l'actif
            ticker_data = self._make_coinlore_request(f"ticker/?id={coin_id}")
            
            if not ticker_data or len(ticker_data) == 0:
                logger.error(f"Aucune donnée récupérée pour {asset} sur CoinLore")
                raise ValueError(f"Aucune donnée pour {asset}")
            
            # CoinLore ne fournit pas de données historiques directement
            # Nous allons créer un DataFrame avec seulement les données actuelles
            coin_data = ticker_data[0]
            
            current_data = {
                'timestamp': datetime.now(),
                'price': float(coin_data.get('price_usd', 0)),
                'market_cap': float(coin_data.get('market_cap_usd', 0)),
                'volume': float(coin_data.get('volume24', 0))
            }
            
            # Pour simuler des données historiques, nous allons utiliser les changements de pourcentage
            # fournis pour estimer les valeurs passées
            data = [current_data]
            
            # Estimation des prix précédents à partir des changements de pourcentage
            if 'percent_change_24h' in coin_data:
                data.append({
                    'timestamp': datetime.now() - timedelta(days=1),
                    'price': current_data['price'] / (1 + float(coin_data['percent_change_24h']) / 100),
                    'market_cap': current_data['market_cap'] / (1 + float(coin_data['percent_change_24h']) / 100),
                    'volume': current_data['volume']  # Pas de changement de volume disponible
                })
            
            if 'percent_change_7d' in coin_data:
                data.append({
                    'timestamp': datetime.now() - timedelta(days=7),
                    'price': current_data['price'] / (1 + float(coin_data['percent_change_7d']) / 100),
                    'market_cap': current_data['market_cap'] / (1 + float(coin_data['percent_change_7d']) / 100),
                    'volume': current_data['volume']  # Pas de changement de volume disponible
                })
            
            # Créer des points supplémentaires par interpolation linéaire
            timestamps = []
            for i in range(days):
                timestamps.append(datetime.now() - timedelta(days=i))
            
            # Créer un DataFrame avec les données connues
            df = pd.DataFrame(data)
            
            # Si nous avons suffisamment de points, utiliser l'interpolation
            if len(df) >= 2:
                # Trier par horodatage
                df = df.sort_values('timestamp')
                
                # Créer un nouveau DataFrame avec tous les timestamps
                full_df = pd.DataFrame({'timestamp': timestamps})
                
                # Fusionner et interpoler
                full_df = pd.merge_asof(full_df.sort_values('timestamp'), 
                                       df.sort_values('timestamp'), 
                                       on='timestamp', 
                                       direction='nearest')
                
                return full_df
            else:
                # Si nous avons trop peu de points, répéter les données actuelles
                full_data = []
                for ts in timestamps:
                    full_data.append({
                        'timestamp': ts,
                        'price': current_data['price'],
                        'market_cap': current_data['market_cap'],
                        'volume': current_data['volume']
                    })
                
                return pd.DataFrame(full_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de marché: {e}")
            return pd.DataFrame()  # Retourner un DataFrame vide en cas d'échec
    
    def get_blockchain_data(self, asset: str = "BTC") -> Dict:
        """
        Récupère des données blockchain générales pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer les données (par défaut "BTC").
            
        Returns:
            Dictionnaire contenant les données blockchain.
        """
        if asset.upper() != "BTC":
            logger.warning(f"Les données blockchain ne sont disponibles que pour BTC, pas pour {asset}")
            return {}
        
        try:
            logger.info(f"Récupération des données blockchain pour {asset}")
            
            # Récupérer les statistiques générales
            endpoint = "stats"
            data = self._make_blockchain_info_request(endpoint)
            
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données blockchain: {e}")
            return {}
    
    def get_utxo_data(self, asset: str = "BTC") -> Dict:
        """
        Récupère des données sur les UTXO pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer les données (par défaut "BTC").
            
        Returns:
            Dictionnaire contenant les données UTXO.
        """
        if asset.upper() != "BTC":
            logger.warning(f"Les données UTXO ne sont disponibles que pour BTC, pas pour {asset}")
            return {}
        
        try:
            logger.info(f"Récupération des données UTXO pour {asset}")
            
            # Utiliser BlockCypher pour les données UTXO
            endpoint = "btc/main"
            data = self._make_blockcypher_request(endpoint)
            
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données UTXO: {e}")
            return {}
    
    def calculate_approximate_realized_cap(self, asset: str, days: int = 365) -> float:
        """
        Calcule une approximation de la capitalisation réalisée pour un actif.
        
        Args:
            asset: Symbole de l'actif (BTC, ETH, etc.).
            days: Nombre de jours de données à récupérer.
            
        Returns:
            Approximation de la capitalisation réalisée.
        """
        try:
            logger.info(f"Calcul de la capitalisation réalisée approximative pour {asset}")
            
            # Récupérer les données de marché historiques
            market_data = self.get_market_data(asset=asset, days=days*4)
            
            if market_data.empty:
                logger.warning(f"Aucune donnée de marché pour {asset}")
                return 0.0
            
            # Prendre les données des derniers 'days' jours
            market_data = market_data.iloc[-days:]
            
            # Calculer une approximation de la capitalisation réalisée
            # Nous utilisons une moyenne pondérée des prix antérieurs comme approximation
            window = min(365, len(market_data) // 2)
            weights = np.array([max(0.1, 1 - i/window) for i in range(window)])
            weights = weights / weights.sum()
            
            # Calculer la capitalisation réalisée approximative
            market_data['realized_price'] = market_data['price'].rolling(window=window).apply(
                lambda x: np.average(x, weights=weights[-len(x):] / np.sum(weights[-len(x):])), 
                raw=True
            ).fillna(market_data['price'])
            
            # Calculer la capitalisation réalisée à partir du prix réalisé
            supply_ratio = market_data['market_cap'] / market_data['price']
            market_data['realized_cap'] = market_data['realized_price'] * supply_ratio
            
            # Calculer le ratio MVRV approximatif
            market_data['mvrv_ratio'] = market_data['market_cap'] / market_data['realized_cap']
            
            return market_data['realized_cap'].iloc[-1]
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la capitalisation réalisée: {e}")
            return 0.0
    
    def get_approximate_mvrv_ratio(self, asset: str = "BTC", days: int = 365) -> pd.DataFrame:
        """
        Calcule une approximation du ratio MVRV.
        
        Args:
            asset: Actif pour lequel calculer le ratio MVRV (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            
        Returns:
            DataFrame contenant le ratio MVRV approximatif.
        """
        try:
            logger.info(f"Calcul du ratio MVRV approximatif pour {asset}")
            
            # Récupérer les données de marché
            market_data = self.get_market_data(asset=asset, days=days)
            
            if market_data.empty:
                logger.warning(f"Aucune donnée de marché pour {asset}")
                return pd.DataFrame()
            
            # Créer un DataFrame avec les colonnes nécessaires
            df_mvrv = pd.DataFrame()
            df_mvrv['timestamp'] = market_data['timestamp']
            df_mvrv['price'] = market_data['price']
            df_mvrv['market_cap'] = market_data['market_cap']
            
            # Prendre la capitalisation boursière actuelle
            current_market_cap = market_data['market_cap'].iloc[-1]
            
            # Calculer l'approximation de la capitalisation réalisée
            realized_cap = self.calculate_approximate_realized_cap(asset=asset, days=days)
            
            if realized_cap <= 0:
                logger.warning(f"Capitalisation réalisée nulle ou négative pour {asset}")
                return pd.DataFrame()
            
            # Estimer l'évolution de la capitalisation réalisée
            # Elle change généralement plus lentement que la capitalisation boursière
            df_mvrv['realized_cap'] = realized_cap
            
            # Pour les données historiques, nous supposons que la cap réalisée
            # a augmenté progressivement jusqu'à sa valeur actuelle
            # C'est une approximation simplifiée
            realized_cap_change_factor = 0.2  # La cap réalisée change à ~20% du taux de la cap boursière
            
            market_cap_ratio = market_data['market_cap'] / current_market_cap
            realized_cap_ratio = market_cap_ratio ** realized_cap_change_factor
            
            df_mvrv['realized_cap'] = realized_cap * realized_cap_ratio
            
            # Calculer le ratio MVRV
            df_mvrv['mvrv_ratio'] = df_mvrv['market_cap'] / df_mvrv['realized_cap']
            
            return df_mvrv
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du ratio MVRV: {e}")
            return pd.DataFrame()
    
    def get_nupl_data(self, asset: str = "BTC", days: int = 365) -> pd.DataFrame:
        """
        Calcule une approximation du NUPL (Net Unrealized Profit/Loss).
        
        Le NUPL est un indicateur qui montre la différence entre la valeur de marché
        et la valeur réalisée en pourcentage de la valeur de marché.
        Il peut être utilisé comme alternative au MVRV.
        
        NUPL = (Market Cap - Realized Cap) / Market Cap
        
        Args:
            asset: Actif pour lequel calculer le NUPL (par défaut "BTC").
            days: Nombre de jours de données à récupérer.
            
        Returns:
            DataFrame contenant le NUPL approximatif.
        """
        try:
            logger.info(f"Calcul du NUPL approximatif pour {asset}")
            
            # Récupérer les données MVRV car NUPL utilise les mêmes données sous-jacentes
            mvrv_data = self.get_approximate_mvrv_ratio(asset=asset, days=days)
            
            if mvrv_data.empty:
                logger.warning(f"Aucune donnée MVRV disponible pour calculer le NUPL pour {asset}")
                return pd.DataFrame()
            
            # Créer un DataFrame pour le NUPL
            df_nupl = pd.DataFrame()
            df_nupl['timestamp'] = mvrv_data['timestamp']
            df_nupl['price'] = mvrv_data['price']
            df_nupl['market_cap'] = mvrv_data['market_cap']
            df_nupl['realized_cap'] = mvrv_data['realized_cap']
            
            # Calculer le NUPL
            df_nupl['nupl'] = (df_nupl['market_cap'] - df_nupl['realized_cap']) / df_nupl['market_cap']
            
            # Classifier le NUPL en catégories
            df_nupl['nupl_category'] = pd.cut(
                df_nupl['nupl'],
                bins=[-1, -0.25, 0, 0.25, 0.5, 0.75, 1],
                labels=['Capitulation', 'Peur', 'Espoir', 'Optimisme', 'Euphorie', 'Avidité']
            )
            
            return df_nupl
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du NUPL: {e}")
            return pd.DataFrame()
