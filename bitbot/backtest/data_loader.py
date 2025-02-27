"""
Chargeur de données historiques pour le backtest.
Supporte plusieurs formats et sources de données.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from pathlib import Path
import json
import gzip
import logging
from datetime import datetime, timedelta

from bitbot.utils.logger import logger

class DataLoader:
    """Chargeur de données avec support de différents formats et sources."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Args:
            data_dir: Chemin vers le répertoire des données
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Le répertoire {data_dir} n'existe pas")
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "binance"
    ) -> pd.DataFrame:
        """
        Charge les données OHLCV depuis un fichier.
        
        Args:
            symbol: Symbole (ex: BTC/USDT)
            timeframe: Intervalle (ex: 1m, 1h, 1d)
            start_date: Date de début
            end_date: Date de fin
            source: Source des données
        
        Returns:
            DataFrame avec les données OHLCV
        """
        # Normaliser le symbole pour le nom de fichier
        symbol_normalized = symbol.replace('/', '').lower()
        
        # Construire le chemin du fichier
        file_pattern = f"{symbol_normalized}_{timeframe}_{source}"
        
        # Chercher les fichiers correspondants
        files = list(self.data_dir.glob(f"{file_pattern}*.csv*"))
        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé pour {file_pattern}")
        
        # Charger et concaténer les données
        dfs = []
        for file in sorted(files):
            try:
                if file.suffix == '.gz':
                    with gzip.open(file, 'rt') as f:
                        df = pd.read_csv(f)
                else:
                    df = pd.read_csv(file)
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("Aucune donnée n'a pu être chargée")
        
        # Concaténer et nettoyer les données
        df = pd.concat(dfs, ignore_index=True)
        
        # Convertir la colonne timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df.index, unit='ms')
        
        # Définir l'index
        df.set_index('timestamp', inplace=True)
        
        # Trier et supprimer les doublons
        df = df.sort_index().drop_duplicates()
        
        # Filtrer par date si spécifié
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        # Vérifier et renommer les colonnes si nécessaire
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Mapping des noms de colonnes possibles
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Vérifier que toutes les colonnes requises sont présentes
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
        
        # Convertir en float
        for col in required_columns:
            df[col] = df[col].astype(float)
        
        return df
    
    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str = "binance",
        compress: bool = True
    ):
        """
        Sauvegarde les données OHLCV dans un fichier.
        
        Args:
            df: DataFrame avec les données
            symbol: Symbole
            timeframe: Intervalle
            source: Source des données
            compress: Si True, compresse en gzip
        """
        symbol_normalized = symbol.replace('/', '').lower()
        
        # Créer le nom de fichier avec la plage de dates
        start_date = df.index.min().strftime('%Y%m%d')
        end_date = df.index.max().strftime('%Y%m%d')
        filename = f"{symbol_normalized}_{timeframe}_{source}_{start_date}_{end_date}"
        
        if compress:
            filename += '.csv.gz'
            with gzip.open(self.data_dir / filename, 'wt') as f:
                df.to_csv(f)
        else:
            filename += '.csv'
            df.to_csv(self.data_dir / filename)
        
        logger.info(f"Données sauvegardées dans {filename}")
    
    def merge_files(
        self,
        symbol: str,
        timeframe: str,
        source: str = "binance",
        compress: bool = True
    ):
        """
        Fusionne plusieurs fichiers de données en un seul.
        
        Args:
            symbol: Symbole
            timeframe: Intervalle
            source: Source des données
            compress: Si True, compresse le fichier résultant
        """
        symbol_normalized = symbol.replace('/', '').lower()
        pattern = f"{symbol_normalized}_{timeframe}_{source}_*.csv*"
        
        # Charger tous les fichiers
        df = self.load_ohlcv(symbol, timeframe, source=source)
        
        # Sauvegarder le fichier fusionné
        self.save_ohlcv(df, symbol, timeframe, source, compress)
        
        logger.info(f"Fichiers fusionnés pour {symbol} {timeframe}")
        
        return df
