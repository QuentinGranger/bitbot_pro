"""
Script de mise à jour quotidienne des données de marché.
À exécuter via cron tous les jours à 00:05 UTC.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import sys

from bitbot.data.binance_client import BinanceClient
from bitbot.utils.logger import logger

class MarketDataUpdater:
    """Gestionnaire de mise à jour des données de marché."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Args:
            symbol: Symbole à mettre à jour
        """
        self.symbol = symbol
        self.client = BinanceClient(verify_ssl=False)
        self.data_dir = Path("data/binance_history")
        
        # Intervalles à mettre à jour
        self.intervals = {
            "1h": timedelta(hours=24),  # 24 dernières heures
            "4h": timedelta(hours=24),  # 24 dernières heures
            "1d": timedelta(days=2),    # 2 derniers jours
        }
        
        # S'assurer que le répertoire existe
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def update_interval(self, interval: str, lookback: timedelta):
        """
        Met à jour un intervalle spécifique.
        
        Args:
            interval: Intervalle (1h, 4h, 1d)
            lookback: Période de mise à jour
        """
        logger.info(f"Mise à jour {self.symbol} {interval}")
        
        # Fichiers
        interval_dir = self.data_dir / interval
        interval_dir.mkdir(exist_ok=True)
        full_file = self.data_dir / f"{self.symbol}_{interval}_full.csv.gz"
        
        # Charger les données existantes
        if full_file.exists():
            df_existing = pd.read_csv(
                full_file,
                compression='gzip',
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
            last_timestamp = df_existing.index[-1]
        else:
            logger.warning(f"Fichier {full_file} non trouvé, téléchargement complet requis")
            return
        
        # Calculer la période de mise à jour
        end_time = datetime.now(pytz.UTC)
        start_time = max(
            last_timestamp - lookback,  # Chevauchement pour éviter les trous
            end_time - lookback
        )
        
        # Télécharger les nouvelles données
        df_new = await self.client.get_klines(
            symbol=self.symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df_new.empty:
            logger.info(f"Pas de nouvelles données pour {self.symbol} {interval}")
            return
        
        # Fusionner avec les données existantes
        df_merged = pd.concat([df_existing, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged.sort_index(inplace=True)
        
        # Sauvegarder
        df_merged.to_csv(full_file, compression='gzip')
        
        logger.info(
            f"Données {self.symbol} {interval} mises à jour: "
            f"{len(df_new)} nouvelles bougies, "
            f"{len(df_merged)} bougies au total"
        )
    
    async def update_all(self):
        """Met à jour toutes les données."""
        try:
            for interval, lookback in self.intervals.items():
                await self.update_interval(interval, lookback)
        finally:
            await self.client.close()

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
