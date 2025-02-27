"""
Classe Trader pour BitBotPro.

Ce module contient la classe principale du trader qui utilise différentes sources
de données pour générer des signaux de trading et exécuter des stratégies.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from bitbot.data.binance_client import BinanceClient
from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.data.cryptopanic_client import CryptoPanicClient
from bitbot.strategies.strategy_factory import StrategyFactory
from bitbot.models.market_data import MarketData, Kline, Signal
from bitbot.models.portfolio import Portfolio
from bitbot.config import Config
from bitbot.utils.logger import logger
from bitbot.utils.notifications import notification_manager, NotificationType, NotificationPriority
from bitbot.utils.data_cleaner import clean_market_data, CleaningMethod

# Configuration du logger
logger = logging.getLogger(__name__)

class Trader:
    """
    Classe principale du trader BitBotPro.
    
    Cette classe coordonne les différentes sources de données et stratégies
    pour générer des signaux de trading et exécuter des ordres.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialise le trader avec une configuration.
        
        Args:
            config: Configuration du trader
        """
        self.config = config or Config()
        self.binance_client = BinanceClient(
            verify_ssl=self.config.verify_ssl,
            data_dir=self.config.data_dir
        )
        self.google_trends_client = GoogleTrendsClient()
        self.cryptopanic_client = CryptoPanicClient(
            api_key=self.config.cryptopanic_api_key
        )
        self.strategy = StrategyFactory.create_strategy(
            self.config.strategy_name,
            self.config.strategy_params
        )
        self.portfolio = Portfolio()
        
        # Créer le répertoire pour les rapports
        os.makedirs("reports", exist_ok=True)
    
    def update_market_data(self, symbol: str = "BTCUSDT", timeframe: str = "1h", limit: int = 100) -> MarketData:
        """
        Met à jour les données de marché pour un symbole donné.
        
        Args:
            symbol: Symbole à mettre à jour (par défaut: BTCUSDT)
            timeframe: Intervalle de temps (par défaut: 1h)
            limit: Nombre de bougies à récupérer
            
        Returns:
            Données de marché mises à jour
        """
        # Récupérer les données de marché depuis Binance
        klines_data = self.binance_client.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        # Convertir les données brutes en objets Kline
        from datetime import datetime
        from decimal import Decimal
        
        klines = []
        for kline_data in klines_data:
            kline = Kline(
                timestamp=datetime.fromtimestamp(kline_data[0] / 1000),
                open=Decimal(str(kline_data[1])),
                high=Decimal(str(kline_data[2])),
                low=Decimal(str(kline_data[3])),
                close=Decimal(str(kline_data[4])),
                volume=Decimal(str(kline_data[5])),
                close_time=datetime.fromtimestamp(kline_data[6] / 1000),
                quote_volume=Decimal(str(kline_data[7])),
                trades=kline_data[8],
                taker_buy_volume=Decimal(str(kline_data[9])),
                taker_buy_quote_volume=Decimal(str(kline_data[10])),
                interval=timeframe
            )
            klines.append(kline)
        
        # Créer un objet MarketData
        market_data = MarketData(symbol=symbol, timeframe=timeframe)
        market_data.update_from_klines(klines)
        
        # Nettoyer les données pour détecter et corriger les valeurs aberrantes
        try:
            # Déterminer le cas d'utilisation en fonction de la stratégie active
            use_case = "general"
            if hasattr(self, 'strategy') and self.strategy:
                strategy_name = self.strategy.__class__.__name__.lower()
                if "trend" in strategy_name:
                    use_case = "trend_following"
                elif "reversion" in strategy_name or "mean" in strategy_name:
                    use_case = "mean_reversion"
                elif "breakout" in strategy_name:
                    use_case = "breakout"
                elif "volatility" in strategy_name:
                    use_case = "volatility"
            
            # Appliquer le nettoyage avec auto-sélection du filtre optimal
            market_data = clean_market_data(
                market_data,
                use_case=use_case
            )
            
            # Stocker les informations sur le nettoyage dans les métadonnées
            market_data.metadata['cleaned'] = True
            market_data.metadata['cleaning_use_case'] = use_case
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage des données: {e}")
        
        # Ajouter les données Google Trends - utilisation synchrone
        # (Note: idéalement, cette méthode devrait être async et appeler await sur get_interest_over_time)
        # trends_data = self.google_trends_client.get_interest_over_time(
        #     keyword=currency,
        #     timeframe='today 7-d'
        # )
        # 
        # if trends_data is not None:
        #     market_data.add_indicator("google_trends", trends_data)
        currency = symbol[:3]  # Extraction de la devise à partir du symbole
        try:
            # Puisque nous ne pouvons pas utiliser await dans une fonction synchrone, 
            # nous désactivons temporairement l'appel à Google Trends
            # trends_data = self.google_trends_client.get_interest_over_time(
            #     keyword=currency,
            #     timeframe='today 7-d'
            # )
            # 
            # if trends_data is not None:
            #     market_data.add_indicator("google_trends", trends_data)
            pass
        except Exception as e:
            logger.warning(f"Impossible de récupérer les données Google Trends: {e}")
        
        # Ajouter les données de sentiment depuis CryptoPanic
        try:
            sentiment_data = self.cryptopanic_client.get_sentiment_analysis(
                currency=currency,
                days=7
            )
            
            if sentiment_data and "sentiment_score" in sentiment_data:
                market_data.add_indicator("sentiment_score", sentiment_data["sentiment_score"])
                market_data.add_indicator("sentiment_positive_ratio", sentiment_data.get("positive_ratio", 0))
                market_data.add_indicator("sentiment_negative_ratio", sentiment_data.get("negative_ratio", 0))
        except Exception as e:
            logger.warning(f"Impossible de récupérer les données de sentiment: {e}")
        
        return market_data
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les données de marché.
        
        Args:
            market_data: Données de marché à analyser
            
        Returns:
            Liste des signaux générés
        """
        # Générer un signal basé sur les tendances Google
        google_signal = None
        if "google_trends" in market_data.indicators:
            trends = market_data.indicators["google_trends"]
            if len(trends) >= 2:
                last_value = trends[-1]
                prev_value = trends[-2]
                
                # Signal positif si la tendance augmente de plus de 20%
                if last_value > prev_value * 1.2:
                    google_signal = Signal(
                        symbol=market_data.symbol,
                        timestamp=datetime.now(),
                        signal_type="GOOGLE_TRENDS",
                        direction=1.0,
                        strength=min(1.0, (last_value - prev_value) / prev_value),
                        source="google_trends"
                    )
                # Signal négatif si la tendance diminue de plus de 20%
                elif last_value < prev_value * 0.8:
                    google_signal = Signal(
                        symbol=market_data.symbol,
                        timestamp=datetime.now(),
                        signal_type="GOOGLE_TRENDS",
                        direction=-1.0,
                        strength=min(1.0, (prev_value - last_value) / prev_value),
                        source="google_trends"
                    )
        
        # Générer un signal basé sur le sentiment des actualités
        news_signal = None
        if "sentiment_score" in market_data.indicators:
            sentiment_score = market_data.indicators["sentiment_score"]
            sentiment_strength = abs(sentiment_score)
            
            if sentiment_score > 0.2:  # Seuil positif
                news_signal = Signal(
                    symbol=market_data.symbol,
                    timestamp=datetime.now(),
                    signal_type="NEWS_SENTIMENT",
                    direction=1.0,
                    strength=sentiment_strength,
                    source="cryptopanic"
                )
            elif sentiment_score < -0.2:  # Seuil négatif
                news_signal = Signal(
                    symbol=market_data.symbol,
                    timestamp=datetime.now(),
                    signal_type="NEWS_SENTIMENT",
                    direction=-1.0,
                    strength=sentiment_strength,
                    source="cryptopanic"
                )
        
        # Combiner avec les signaux de la stratégie
        strategy_signals = self.strategy.generate_signals(market_data)
        
        # Fusionner tous les signaux
        all_signals = []
        if google_signal:
            all_signals.append(google_signal)
        if news_signal:
            all_signals.append(news_signal)
        all_signals.extend(strategy_signals)
        
        return all_signals
    
    def execute_signals(self, signals: List[Signal]) -> bool:
        """
        Exécute les signaux générés.
        
        Args:
            signals: Liste des signaux à exécuter
            
        Returns:
            True si les signaux ont été exécutés correctement
        """
        if not signals:
            logger.info("Aucun signal à exécuter")
            return True
        
        try:
            # Envoyer notification pour chaque signal
            import asyncio
            from bitbot.utils.notifications import notification_manager, NotificationPriority, NotificationType
            
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            for signal in signals:
                # Log du signal
                logger.info(f"Exécution du signal: {signal}")
                
                # Déterminer le type de notification et la priorité
                if signal.direction > 0:
                    action = "ACHAT"
                    notification_type = NotificationType.BUY_SIGNAL
                    emoji = "🟢"
                else:
                    action = "VENTE"
                    notification_type = NotificationType.SELL_SIGNAL
                    emoji = "🔴"
                
                # Force du signal détermine la priorité
                priority = NotificationPriority.MEDIUM
                if abs(signal.strength) > 0.8:
                    priority = NotificationPriority.HIGH
                
                # Envoyer la notification
                async def send_signal_notification(signal, action, emoji, notification_type, priority):
                    await notification_manager.notify(
                        message=f"{emoji} Signal de {action} détecté pour {signal.symbol}",
                        title=f"BitBotPro - Signal de {action}",
                        notification_type=notification_type,
                        priority=priority,
                        details={
                            "Symbole": signal.symbol,
                            "Force": f"{signal.strength:.2f}",
                            "Source": signal.source,
                            "Type": signal.signal_type,
                            "Date": signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )
                
                # Exécuter la notification
                loop.run_until_complete(send_signal_notification(
                    signal, action, emoji, notification_type, priority
                ))
                
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution des signaux: {e}")
            return False
    
    def visualize_sentiment(self, currency: str = "BTC", days: int = 7) -> str:
        """
        Visualise le sentiment des actualités pour une cryptomonnaie.
        
        Args:
            currency: Code de la cryptomonnaie
            days: Nombre de jours à analyser
            
        Returns:
            Chemin vers le fichier d'image généré
        """
        # Récupérer les actualités pour la période spécifiée
        news = []
        
        # Récupérer les actualités page par page
        for page in range(1, 5):  # Limiter à 4 pages
            news_page = self.cryptopanic_client.get_news(
                currencies=currency,
                page=page,
                force_refresh=True
            )
            news.extend(news_page)
            
            # Vérifier si nous avons assez d'actualités 
            # ou si nous avons atteint la fin des résultats
            if not news_page or len(news) >= 50:
                break
        
        # Filtrer les actualités par date
        cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days)
        filtered_news = []
        for item in news:
            if item.published_at:
                # Normaliser la date sans fuseau horaire
                item_date = item.published_at
                if item_date.tzinfo is not None:
                    item_date = item_date.replace(tzinfo=None)
                
                if item_date >= cutoff_date:
                    filtered_news.append(item)
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        
        # Préparer les données
        dates = [item.published_at.replace(tzinfo=None) if item.published_at.tzinfo else item.published_at for item in filtered_news]
        sentiments = [item.get_sentiment_score() for item in filtered_news]
        
        # Couleurs basées sur le sentiment
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments]
        
        # Tracer les points
        plt.scatter(dates, sentiments, c=colors, alpha=0.7, s=50)
        
        # Ajouter une ligne de tendance
        if dates and sentiments:
            z = np.polyfit(
                [d.timestamp() for d in dates], 
                sentiments, 
                1
            )
            p = np.poly1d(z)
            
            date_range = [min(dates), max(dates)]
            timestamp_range = [d.timestamp() for d in date_range]
            trend_line = p(timestamp_range)
            
            plt.plot(
                date_range, 
                trend_line, 
                'b--', 
                linewidth=2, 
                label=f'Tendance: {"Positive" if trend_line[1] > trend_line[0] else "Négative"}'
            )
        
        # Ajouter l'axe horizontal à zéro
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajouter des étiquettes et un titre
        plt.title(f'Analyse de sentiment pour {currency} (derniers {days} jours)')
        plt.xlabel('Date de publication')
        plt.ylabel('Score de sentiment (-1 à 1)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Formater l'axe des dates
        plt.gcf().autofmt_xdate()
        
        # Sauvegarder le graphique
        output_file = f"reports/{currency.lower()}_sentiment_analysis.png"
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def run(self, symbol: str = "BTCUSDT", timeframe: str = "1h", limit: int = 100):
        """
        Exécute une itération complète du trader.
        
        Args:
            symbol: Symbole à trader
            timeframe: Intervalle de temps
            limit: Nombre de bougies à récupérer
        """
        # Mettre à jour les données de marché
        market_data = self.update_market_data(symbol, timeframe, limit)
        
        # Générer des signaux
        signals = self.generate_signals(market_data)
        
        # Exécuter les signaux
        self.execute_signals(signals)
        
        # Générer des visualisations
        currency = symbol[:3]
        self.visualize_sentiment(currency)
        
        return True
        
    def run_backtest(self, market_data: MarketData) -> Dict:
        """
        Exécute un backtest sur des données historiques.
        
        Args:
            market_data: Données de marché à analyser
            
        Returns:
            Résultats du backtest
        """
        logger.info(f"Démarrage du backtest sur {market_data.symbol} ({market_data.timeframe})")
        
        # Initialiser les résultats
        results = {
            "initial_balance": 10000.0,  # USD
            "final_balance": 10000.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "max_drawdown": 0.0,
            "trades": []
        }
        
        # Vérifier si nous avons des données
        if market_data.ohlcv.empty:
            logger.warning(f"Pas de données pour {market_data.symbol} ({market_data.timeframe})")
            return results
        
        # Diviser les données pour simuler le trading jour par jour
        data_length = len(market_data.ohlcv)
        window_size = 24  # 24 bougies pour 1 jour avec timeframe 1h
        
        # Simuler le trading sur différentes périodes
        current_balance = results["initial_balance"]
        max_balance = current_balance
        
        for i in range(window_size, data_length):
            # Créer un sous-ensemble de données pour l'analyse
            window_data = market_data.ohlcv.iloc[:i].copy()
            window_market_data = MarketData(symbol=market_data.symbol, timeframe=market_data.timeframe)
            window_market_data.ohlcv = window_data
            
            # Générer des signaux sur la fenêtre actuelle
            signals = self.generate_signals(window_market_data)
            
            # Simuler l'exécution des signaux
            if signals:
                last_price = float(window_data.iloc[-1]["close"])
                
                for signal in signals:
                    trade = {
                        "timestamp": window_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": signal.symbol,
                        "side": signal.side,
                        "price": last_price,
                        "quantity": 0.1,  # Simuler un achat/vente de 0.1 BTC
                        "profit_loss": 0.0
                    }
                    
                    # Simuler le résultat du trade (simplifié)
                    if signal.side == "BUY":
                        # Simuler un gain/perte aléatoire entre -2% et +5%
                        profit_percentage = np.random.uniform(-0.02, 0.05)
                    else:  # SELL
                        # Simuler un gain/perte aléatoire entre -2% et +5% (inversé pour SELL)
                        profit_percentage = np.random.uniform(-0.02, 0.05) * -1
                    
                    trade_value = last_price * trade["quantity"]
                    profit_loss = trade_value * profit_percentage
                    trade["profit_loss"] = profit_loss
                    
                    # Mettre à jour le solde
                    current_balance += profit_loss
                    max_balance = max(max_balance, current_balance)
                    
                    # Mettre à jour les statistiques
                    results["total_trades"] += 1
                    if profit_loss > 0:
                        results["winning_trades"] += 1
                    else:
                        results["losing_trades"] += 1
                    
                    results["trades"].append(trade)
        
        # Calculer les résultats finaux
        results["final_balance"] = current_balance
        results["profit_loss"] = current_balance - results["initial_balance"]
        
        if results["total_trades"] > 0:
            results["win_rate"] = results["winning_trades"] / results["total_trades"] * 100
        
        results["max_drawdown"] = (max_balance - min(max_balance, current_balance)) / max_balance * 100
        
        # Générer des visualisations
        currency = market_data.symbol[:3]
        self.visualize_sentiment(currency)
        
        logger.info(f"Backtest terminé: Balance finale: {current_balance:.2f} USD, P&L: {results['profit_loss']:.2f} USD")
        
        return results
    
    def run_live(self, symbol: str = "BTCUSDT", timeframe: str = "1h"):
        """
        Exécute le trading en mode live.
        
        Args:
            symbol: Symbole à trader
            timeframe: Intervalle de temps
        
        Returns:
            True si l'exécution a réussi, False sinon
        """
        logger.info(f"Démarrage du trading live sur {symbol} ({timeframe})")
        
        try:
            # Envoyer une notification de démarrage via Telegram
            import asyncio
            from bitbot.utils.notifications import notification_manager
            
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Fonction asynchrone pour envoyer la notification
            async def send_startup_notification():
                await notification_manager.notify(
                    message=f"BitBotPro démarré en mode live pour {symbol} ({timeframe})",
                    title="BitBotPro - Démarrage",
                    details={
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": "Live Trading",
                        "symbole": symbol,
                        "intervalle": timeframe
                    }
                )
            
            # Exécuter la notification de manière asynchrone
            loop.run_until_complete(send_startup_notification())
            
            # Mettre à jour les données de marché
            market_data = self.update_market_data(symbol, timeframe)
            
            # Générer des signaux
            signals = self.generate_signals(market_data)
            
            # Exécuter les signaux
            self.execute_signals(signals)
            
            # Générer des visualisations
            currency = symbol[:3]
            self.visualize_sentiment(currency)
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du trading live: {e}")
            return False
    
    def run_simulation(self, symbol: str = "BTCUSDT", timeframe: str = "1h"):
        """
        Exécute le trading en mode simulation (paper trading).
        
        Args:
            symbol: Symbole à trader
            timeframe: Intervalle de temps
        
        Returns:
            True si l'exécution a réussi, False sinon
        """
        logger.info(f"Démarrage de la simulation sur {symbol} ({timeframe})")
        
        try:
            # Mettre à jour les données de marché
            market_data = self.update_market_data(symbol, timeframe)
            
            # Générer des signaux
            signals = self.generate_signals(market_data)
            
            # Simuler l'exécution des signaux
            if signals:
                for signal in signals:
                    logger.info(f"Signal simulé: {signal.symbol} {signal.side} à {signal.price}")
            
            # Générer des visualisations
            currency = symbol[:3]
            self.visualize_sentiment(currency)
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la simulation: {e}")
            return False
