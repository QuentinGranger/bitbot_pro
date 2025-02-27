import logging
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime
import pytz
from collections import namedtuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Import des modules du projet BitBotPro
from bitbot.trader import Trader
from bitbot.config import Config
from bitbot.data.binance_client import BinanceClient
from bitbot.models.market_data import MarketData

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='BitBot Pro - Robot de trading crypto')
    
    # Arguments pour le mode de fonctionnement
    parser.add_argument('--backtest', action='store_true', help='Exécuter en mode backtest')
    parser.add_argument('--live', action='store_true', help='Exécuter en mode trading live')
    parser.add_argument('--simulate', action='store_true', help='Exécuter en mode simulation')
    
    # Arguments pour la configuration
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbole de trading (par défaut: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Intervalle de temps (par défaut: 1h)')
    parser.add_argument('--days', type=int, default=30, help='Nombre de jours pour le backtest (par défaut: 30)')
    
    return parser.parse_args()

def main():
    """Fonction principale de BitBot Pro."""
    args = parse_arguments()
    
    try:
        # Initialisation de la configuration
        config = Config()
        
        # Initialisation du trader
        trader = Trader(config)
        
        if args.backtest:
            logger.info(f"Lancement du backtest sur {args.symbol} (timeframe: {args.timeframe}, jours: {args.days})")
            
            # Obtenir les données historiques
            klines_data = trader.binance_client.get_historical_klines(
                symbol=args.symbol,
                interval=args.timeframe,
                start_str=f"{args.days} days ago UTC"
            )
            
            # Convertir les données au format attendu par MarketData
            Kline = namedtuple('Kline', ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            klines = []
            
            for kline_data in klines_data:
                timestamp = datetime.fromtimestamp(kline_data[0] / 1000, tz=pytz.UTC)
                kline = Kline(
                    timestamp=timestamp,
                    open=kline_data[1],
                    high=kline_data[2],
                    low=kline_data[3],
                    close=kline_data[4],
                    volume=kline_data[5]
                )
                klines.append(kline)
            
            # Créer un objet MarketData
            market_data = MarketData(symbol=args.symbol, timeframe=args.timeframe)
            market_data.update_from_klines(klines)
            
            # Exécuter le backtest
            results = trader.run_backtest(market_data)
            
            # Afficher les résultats
            logger.info(f"Résultats du backtest: {results}")
            
        elif args.live:
            logger.info(f"Lancement du trading live sur {args.symbol} (timeframe: {args.timeframe})")
            trader.run_live(args.symbol, args.timeframe)
            
        elif args.simulate:
            logger.info(f"Lancement de la simulation sur {args.symbol} (timeframe: {args.timeframe})")
            trader.run_simulation(args.symbol, args.timeframe)
            
        else:
            logger.info("Veuillez spécifier un mode de fonctionnement: --backtest, --live ou --simulate")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise

if __name__ == "__main__":
    main()
