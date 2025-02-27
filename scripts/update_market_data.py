"""
Script de mise à jour quotidienne des données de marché.
À exécuter via cron tous les jours à 00:05 UTC.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import sys
import json
import gzip
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

from bitbot.data.market_data import MarketDataProvider, MarketDataConfig
from bitbot.utils.logger import logger

class MarketDataUpdater:
    """Gestionnaire de mise à jour des données de marché."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Args:
            symbol: Symbole à mettre à jour
        """
        self.symbol = symbol
        
        # Configuration avec la clé API de CoinMarketCap
        self.config = MarketDataConfig(
            coinmarketcap_api_key=os.getenv("COINMARKETCAP_API_KEY"),
            verify_ssl=False,
            data_dir="data"  # Répertoire racine des données
        )
        
        # Client avec redondance
        self.provider = MarketDataProvider(self.config)
        
        # Répertoire de données
        self.data_dir = Path(self.config.data_dir)
        
        # Intervalles à mettre à jour avec leur lookback
        self.intervals = {
            # Données granulaires pour le microtrading
            "1m": timedelta(hours=1),    # Dernière heure
            "5m": timedelta(hours=2),    # 2 dernières heures
            "15m": timedelta(hours=4),   # 4 dernières heures
            # Données intermédiaires
            "1h": timedelta(hours=24),   # 24 dernières heures
            "4h": timedelta(hours=24),   # 24 dernières heures
            # Données journalières avec redondance
            "1d": timedelta(days=5),     # 5 derniers jours pour assurer la continuité
        }
        
        # Profondeur du carnet d'ordres à capturer
        self.order_book_levels = [5, 10, 20, 50, 100, 500, 1000]
        
        # S'assurer que les répertoires existent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "binance_history/order_book").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "coingecko_history/klines").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "coinmarketcap_history/klines").mkdir(parents=True, exist_ok=True)
    
    async def update_interval(self, interval: str, lookback: timedelta):
        """
        Met à jour un intervalle spécifique.
        
        Args:
            interval: Intervalle (1m, 5m, 15m, 1h, 4h, 1d)
            lookback: Période de mise à jour
        """
        logger.info(f"Mise à jour {self.symbol} {interval}")
        
        # Télécharger les nouvelles données avec redondance
        end_time = datetime.now(pytz.UTC)
        
        # Pour les données journalières, on télécharge 10 ans d'historique
        if interval == "1d":
            start_year = 2015  # 10 ans d'historique
            start_time = datetime(start_year, 1, 1, tzinfo=pytz.UTC)
        else:
            start_time = end_time - lookback
        
        try:
            df_new, source = await self.provider.get_klines(
                symbol=self.symbol,
                interval=interval,
                start_time=start_time.replace(tzinfo=None),  # Les APIs n'aiment pas les tz
                end_time=end_time.replace(tzinfo=None)
            )
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données pour {self.symbol} {interval}: {str(e)}")
            return
        
        if df_new.empty:
            logger.info(f"Pas de nouvelles données pour {self.symbol} {interval}")
            return
        
        # Déterminer le répertoire de sauvegarde en fonction de la source
        if source == "binance":
            data_dir = self.data_dir / "binance_history/klines"
        elif source == "coingecko":
            data_dir = self.data_dir / "coingecko_history/klines"
        elif source == "coinmarketcap":
            data_dir = self.data_dir / "coinmarketcap_history/klines"
        elif source.endswith("_cache"):  # Ne pas sauvegarder les données du cache
            logger.warning(f"Données récupérées depuis le cache, pas de sauvegarde")
            return
        else:
            logger.error(f"Source inconnue: {source}")
            return
        
        # Créer le répertoire si nécessaire
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichier de données
        full_file = data_dir / f"{self.symbol}_{interval}_full.csv.gz"
        
        # Charger les données existantes
        if full_file.exists():
            try:
                df_existing = pd.read_csv(
                    full_file,
                    compression='gzip',
                    parse_dates=['timestamp'],
                    index_col='timestamp'
                )
            except pd.errors.EmptyDataError:
                logger.warning(f"Fichier {full_file} vide, initialisation avec les dernières données")
                df_existing = pd.DataFrame()
        else:
            logger.info(f"Création du fichier {full_file}")
            df_existing = pd.DataFrame()
        
        # Fusionner avec les données existantes
        df_merged = pd.concat([df_existing, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged.sort_index(inplace=True)
        
        # Sauvegarder
        df_merged.to_csv(full_file, compression='gzip')
        
        logger.info(
            f"Données {self.symbol} {interval} mises à jour depuis {source}: "
            f"{len(df_new)} nouvelles bougies, "
            f"{len(df_merged)} bougies au total"
        )
    
    async def update_order_book(self):
        """Met à jour les données du carnet d'ordres."""
        logger.info(f"Mise à jour du carnet d'ordres {self.symbol}")
        
        # Récupérer le carnet d'ordres pour chaque profondeur
        for depth in self.order_book_levels:
            try:
                order_book = await self.provider.get_order_book(
                    symbol=self.symbol,
                    limit=depth
                )
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du carnet d'ordres pour {self.symbol} {depth} niveaux: {str(e)}")
                continue
            
            if not order_book:
                logger.warning(f"Pas de données de carnet d'ordres pour {depth} niveaux")
                continue
            
            # Ajouter le timestamp
            order_book['timestamp'] = datetime.now(pytz.UTC).isoformat()
            
            # Sauvegarder dans un fichier
            filename = self.data_dir / "binance_history/order_book" / f"{self.symbol}_depth_{depth}_{datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')}.json.gz"
            with gzip.open(filename, 'wt') as f:
                json.dump(order_book, f)
            
            logger.info(f"Carnet d'ordres {depth} niveaux sauvegardé: {filename}")
            
            # Attendre un peu pour ne pas surcharger l'API
            await asyncio.sleep(1)
    
    async def update_all(self):
        """Met à jour toutes les données."""
        try:
            # Mettre à jour les données OHLCV
            for interval, lookback in self.intervals.items():
                await self.update_interval(interval, lookback)
            
            # Mettre à jour le carnet d'ordres
            await self.update_order_book()
        finally:
            await self.provider.close()

async def main():
    """Point d'entrée principal."""
    try:
        updater = MarketDataUpdater()
        await updater.update_all()
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
