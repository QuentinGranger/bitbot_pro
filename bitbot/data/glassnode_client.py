"""
Module pour interagir avec l'API Glassnode.

Ce module fournit des fonctionnalités pour récupérer des données on-chain
depuis l'API Glassnode, notamment des métriques comme le ratio MVRV.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union, Any
import logging
from dotenv import load_dotenv

from bitbot.utils.logger import logger

class GlassnodeClient:
    """
    Client pour interagir avec l'API Glassnode.
    
    Cette classe permet de récupérer des données on-chain depuis l'API Glassnode,
    notamment des métriques comme le ratio MVRV, les flux d'échanges, etc.
    """
    
    BASE_URL = "https://api.glassnode.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client Glassnode.
        
        Args:
            api_key: Clé API Glassnode. Si None, tente de la récupérer depuis les variables d'environnement.
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        # Récupérer la clé API
        self.api_key = api_key or os.getenv("GLASSNODE_API_KEY")
        
        if not self.api_key:
            logger.warning("Aucune clé API Glassnode trouvée. Certaines fonctionnalités seront limitées.")
        
        logger.info("Client Glassnode initialisé")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Effectue une requête à l'API Glassnode.
        
        Args:
            endpoint: Point de terminaison de l'API.
            params: Paramètres de la requête.
            
        Returns:
            Réponse de l'API sous forme de dictionnaire.
            
        Raises:
            Exception: Si la requête échoue.
        """
        # Paramètres par défaut
        default_params = {
            "api_key": self.api_key,
            "format": "json"
        }
        
        # Fusionner les paramètres
        if params:
            default_params.update(params)
        
        # Construire l'URL
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            # Effectuer la requête
            response = requests.get(url, params=default_params)
            response.raise_for_status()
            
            # Retourner les données
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API Glassnode: {e}")
            if response.status_code == 401:
                logger.error("Clé API Glassnode invalide ou non fournie.")
            elif response.status_code == 429:
                logger.error("Limite de requêtes Glassnode atteinte.")
            raise
    
    def get_mvrv_ratio(self, 
                      asset: str = "BTC", 
                      since: Optional[Union[str, datetime]] = None,
                      until: Optional[Union[str, datetime]] = None,
                      interval: str = "24h") -> pd.DataFrame:
        """
        Récupère le ratio MVRV (Market Value to Realized Value) pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer le ratio MVRV (par défaut "BTC").
            since: Date de début (format ISO ou objet datetime).
            until: Date de fin (format ISO ou objet datetime).
            interval: Intervalle des données (par défaut "24h").
            
        Returns:
            DataFrame contenant les données du ratio MVRV.
        """
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.error("Clé API Glassnode requise pour récupérer le ratio MVRV.")
            return pd.DataFrame()
        
        # Préparer les paramètres
        params = {
            "a": asset,
            "i": interval
        }
        
        # Ajouter les dates si spécifiées
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%d")
            params["s"] = since
        
        if until:
            if isinstance(until, datetime):
                until = until.strftime("%Y-%m-%d")
            params["u"] = until
        
        try:
            # Effectuer la requête
            logger.info(f"Récupération du ratio MVRV pour {asset}")
            data = self._make_request("metrics/market/mvrv", params)
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Convertir les timestamps en datetime
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df.set_index("timestamp", inplace=True)
                df.drop("t", axis=1, inplace=True)
            
            # Renommer les colonnes
            df.rename(columns={"v": "mvrv_ratio"}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ratio MVRV: {e}")
            return pd.DataFrame()
    
    def get_realized_price(self, 
                          asset: str = "BTC", 
                          since: Optional[Union[str, datetime]] = None,
                          until: Optional[Union[str, datetime]] = None,
                          interval: str = "24h") -> pd.DataFrame:
        """
        Récupère le prix réalisé (Realized Price) pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer le prix réalisé (par défaut "BTC").
            since: Date de début (format ISO ou objet datetime).
            until: Date de fin (format ISO ou objet datetime).
            interval: Intervalle des données (par défaut "24h").
            
        Returns:
            DataFrame contenant les données du prix réalisé.
        """
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.error("Clé API Glassnode requise pour récupérer le prix réalisé.")
            return pd.DataFrame()
        
        # Préparer les paramètres
        params = {
            "a": asset,
            "i": interval
        }
        
        # Ajouter les dates si spécifiées
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%d")
            params["s"] = since
        
        if until:
            if isinstance(until, datetime):
                until = until.strftime("%Y-%m-%d")
            params["u"] = until
        
        try:
            # Effectuer la requête
            logger.info(f"Récupération du prix réalisé pour {asset}")
            data = self._make_request("metrics/market/price_realized_usd", params)
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Convertir les timestamps en datetime
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df.set_index("timestamp", inplace=True)
                df.drop("t", axis=1, inplace=True)
            
            # Renommer les colonnes
            df.rename(columns={"v": "realized_price"}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix réalisé: {e}")
            return pd.DataFrame()
    
    def get_market_cap(self, 
                      asset: str = "BTC", 
                      since: Optional[Union[str, datetime]] = None,
                      until: Optional[Union[str, datetime]] = None,
                      interval: str = "24h") -> pd.DataFrame:
        """
        Récupère la capitalisation boursière pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer la capitalisation (par défaut "BTC").
            since: Date de début (format ISO ou objet datetime).
            until: Date de fin (format ISO ou objet datetime).
            interval: Intervalle des données (par défaut "24h").
            
        Returns:
            DataFrame contenant les données de capitalisation boursière.
        """
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.error("Clé API Glassnode requise pour récupérer la capitalisation boursière.")
            return pd.DataFrame()
        
        # Préparer les paramètres
        params = {
            "a": asset,
            "i": interval
        }
        
        # Ajouter les dates si spécifiées
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%d")
            params["s"] = since
        
        if until:
            if isinstance(until, datetime):
                until = until.strftime("%Y-%m-%d")
            params["u"] = until
        
        try:
            # Effectuer la requête
            logger.info(f"Récupération de la capitalisation boursière pour {asset}")
            data = self._make_request("metrics/market/marketcap_usd", params)
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Convertir les timestamps en datetime
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df.set_index("timestamp", inplace=True)
                df.drop("t", axis=1, inplace=True)
            
            # Renommer les colonnes
            df.rename(columns={"v": "market_cap"}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la capitalisation boursière: {e}")
            return pd.DataFrame()
    
    def get_realized_cap(self, 
                        asset: str = "BTC", 
                        since: Optional[Union[str, datetime]] = None,
                        until: Optional[Union[str, datetime]] = None,
                        interval: str = "24h") -> pd.DataFrame:
        """
        Récupère la capitalisation réalisée (Realized Cap) pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer la capitalisation réalisée (par défaut "BTC").
            since: Date de début (format ISO ou objet datetime).
            until: Date de fin (format ISO ou objet datetime).
            interval: Intervalle des données (par défaut "24h").
            
        Returns:
            DataFrame contenant les données de capitalisation réalisée.
        """
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.error("Clé API Glassnode requise pour récupérer la capitalisation réalisée.")
            return pd.DataFrame()
        
        # Préparer les paramètres
        params = {
            "a": asset,
            "i": interval
        }
        
        # Ajouter les dates si spécifiées
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%d")
            params["s"] = since
        
        if until:
            if isinstance(until, datetime):
                until = until.strftime("%Y-%m-%d")
            params["u"] = until
        
        try:
            # Effectuer la requête
            logger.info(f"Récupération de la capitalisation réalisée pour {asset}")
            data = self._make_request("metrics/market/marketcap_realized_usd", params)
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Convertir les timestamps en datetime
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df.set_index("timestamp", inplace=True)
                df.drop("t", axis=1, inplace=True)
            
            # Renommer les colonnes
            df.rename(columns={"v": "realized_cap"}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la capitalisation réalisée: {e}")
            return pd.DataFrame()
    
    def get_sopr(self, 
                asset: str = "BTC", 
                since: Optional[Union[str, datetime]] = None,
                until: Optional[Union[str, datetime]] = None,
                interval: str = "24h") -> pd.DataFrame:
        """
        Récupère le SOPR (Spent Output Profit Ratio) pour un actif.
        
        Args:
            asset: Actif pour lequel récupérer le SOPR (par défaut "BTC").
            since: Date de début (format ISO ou objet datetime).
            until: Date de fin (format ISO ou objet datetime).
            interval: Intervalle des données (par défaut "24h").
            
        Returns:
            DataFrame contenant les données du SOPR.
        """
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.error("Clé API Glassnode requise pour récupérer le SOPR.")
            return pd.DataFrame()
        
        # Préparer les paramètres
        params = {
            "a": asset,
            "i": interval
        }
        
        # Ajouter les dates si spécifiées
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%d")
            params["s"] = since
        
        if until:
            if isinstance(until, datetime):
                until = until.strftime("%Y-%m-%d")
            params["u"] = until
        
        try:
            # Effectuer la requête
            logger.info(f"Récupération du SOPR pour {asset}")
            data = self._make_request("metrics/indicators/sopr", params)
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Convertir les timestamps en datetime
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df.set_index("timestamp", inplace=True)
                df.drop("t", axis=1, inplace=True)
            
            # Renommer les colonnes
            df.rename(columns={"v": "sopr"}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du SOPR: {e}")
            return pd.DataFrame()
