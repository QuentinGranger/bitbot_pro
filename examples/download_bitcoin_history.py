"""
Script pour télécharger 10 ans de données Bitcoin depuis Binance.
"""

import asyncio
from pathlib import Path
from datetime import datetime

from bitbot.data.binance_client import BinanceClient
from bitbot.utils.logger import logger

async def download_bitcoin_history():
    """Télécharge l'historique complet du Bitcoin."""
    
    # Configuration
    symbol = "BTCUSDT"
    intervals = ["1h", "4h", "1d"]  # Intervalles à télécharger
    start_year = 2015  # Binance a commencé en 2015
    end_year = datetime.now().year
    
    # Répertoire de sortie
    output_dir = Path("data/binance_history")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Client Binance
    client = BinanceClient()
    
    try:
        # Télécharger pour chaque intervalle
        for interval in intervals:
            logger.info(f"Téléchargement {symbol} {interval}")
            
            # Créer un sous-répertoire pour l'intervalle
            interval_dir = output_dir / interval
            interval_dir.mkdir(exist_ok=True)
            
            # Télécharger les données
            await client.download_historical_data(
                symbol=symbol,
                interval=interval,
                start_year=start_year,
                end_year=end_year,
                output_dir=interval_dir
            )
            
            # Fusionner en un seul fichier
            logger.info(f"Fusion des données {symbol} {interval}")
            merged_file = output_dir / f"{symbol}_{interval}_full.csv.gz"
            
            df = client.merge_yearly_files(
                symbol=symbol,
                interval=interval,
                data_dir=interval_dir,
                output_file=merged_file
            )
            
            logger.info(
                f"Données {symbol} {interval} complètes: "
                f"{len(df)} bougies de {df.index[0]} à {df.index[-1]}"
            )
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(download_bitcoin_history())
