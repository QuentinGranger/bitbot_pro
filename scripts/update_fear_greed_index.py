#!/usr/bin/env python3
"""
Script de mise à jour automatique de l'indice de peur et d'avidité.
Ce script peut être exécuté périodiquement via un planificateur comme cron.
"""

import asyncio
import sys
import os
import argparse
import pandas as pd
from pathlib import Path
import datetime
import json

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitbot.data.market_data import MarketDataProvider, MarketDataConfig
from bitbot.utils.logger import logger


async def update_fear_greed_index(output_dir, format="csv", days=90):
    """
    Met à jour les données de l'indice de peur et d'avidité.
    
    Args:
        output_dir: Répertoire de sortie
        format: Format de sortie ('csv' ou 'json')
        days: Nombre de jours d'historique à récupérer
    """
    # Assurez-vous que le répertoire de sortie existe
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialiser le MarketDataProvider
    config = MarketDataConfig(
        data_dir="data",
        enable_fear_greed_index=True,
        verify_ssl=False
    )
    provider = MarketDataProvider(config)
    
    try:
        # Récupérer l'indice actuel
        latest = await provider.get_latest_fear_greed_index()
        if latest:
            logger.info(f"Indice actuel: {latest.value} ({latest.classification})")
            
            # Enregistrer dans un fichier timestamp.json
            latest_data = {
                "value": latest.value,
                "classification": latest.classification,
                "timestamp": latest.timestamp.isoformat(),
                "sentiment_score": latest.get_sentiment_score(),
                "time_until_update": latest.time_until_update
            }
            
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            latest_file = output_path / f"fear_greed_latest_{current_date}.json"
            with open(latest_file, "w") as f:
                json.dump(latest_data, f, indent=2)
            logger.info(f"Dernier indice enregistré dans {latest_file}")
            
            # Créer un lien symbolique vers le fichier le plus récent
            latest_link = output_path / "fear_greed_latest.json"
            try:
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(latest_file.name)
            except Exception as e:
                logger.warning(f"Impossible de créer le lien symbolique: {str(e)}")
        
        # Récupérer l'historique
        df = await provider.get_fear_greed_history(days=days)
        if not df.empty:
            logger.info(f"Historique sur {days} jours: {len(df)} entrées")
            
            # Enregistrer dans le format demandé
            if format.lower() == "csv":
                output_file = output_path / "fear_greed_history.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Historique enregistré dans {output_file}")
            elif format.lower() == "json":
                output_file = output_path / "fear_greed_history.json"
                df.to_json(output_file, orient="records", date_format="iso")
                logger.info(f"Historique enregistré dans {output_file}")
            else:
                logger.error(f"Format de sortie non pris en charge: {format}")
        else:
            logger.error("Impossible de récupérer l'historique de l'indice")
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'indice: {str(e)}")
    
    finally:
        await provider.close()


async def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Mise à jour de l'indice de peur et d'avidité")
    parser.add_argument("--output", "-o", default="data/fear_greed", help="Répertoire de sortie")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", help="Format de sortie")
    parser.add_argument("--days", "-d", type=int, default=90, help="Nombre de jours d'historique")
    args = parser.parse_args()
    
    await update_fear_greed_index(args.output, args.format, args.days)


if __name__ == "__main__":
    asyncio.run(main())
