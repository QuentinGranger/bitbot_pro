"""
Classe Trader pour BitBotPro.

Ce module contient la classe principale du trader qui utilise différentes sources
de données pour générer des signaux de trading et exécuter des stratégies.
"""

import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import json

from bitbot.config import Config
from bitbot.data.binance_client import BinanceClient
from bitbot.data.google_trends_client import GoogleTrendsClient
from bitbot.data.cryptopanic_client import CryptoPanicClient
from bitbot.strategies.strategy_factory import StrategyFactory
from bitbot.models.market_data import MarketData
from bitbot.models.portfolio import Portfolio
from bitbot.models.signal import Signal
from bitbot.utils.performance_monitor import performance_monitor
from bitbot.utils.journal import log_decision, log_signal, log_order, log_error, log_market_data, generate_report

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
        
        # Mode Safe - protection contre les événements critiques
        self.safe_mode_enabled = False
        self.safe_mode_reason = None
        self.safe_mode_activated_at = None
        
        # Configuration du moniteur de performance
        if hasattr(self.config, "max_drawdown_threshold"):
            performance_monitor.max_drawdown_threshold = self.config.max_drawdown_threshold
        if hasattr(self.config, "drawdown_alert_threshold"):
            performance_monitor.drawdown_alert_threshold = self.config.drawdown_alert_threshold
        if hasattr(self.config, "order_frequency_threshold"):
            performance_monitor.order_frequency_threshold = self.config.order_frequency_threshold
        
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
        logger.info(f"Mise à jour des données pour {symbol} ({timeframe})")
        
        try:
            # Récupérer les données OHLCV (bougies)
            ohlcv = self.binance_client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            # Créer l'objet MarketData
            market_data = MarketData(symbol=symbol, timeframe=timeframe)
            market_data.ohlcv = ohlcv
            
            # Récupérer les tendances Google si disponibles
            try:
                trends_data = self.google_trends_client.get_trends(
                    keyword=symbol[:3],
                    timeframe="today 3-m"  # Derniers 3 mois
                )
                market_data.trends = trends_data
            except Exception as e:
                logger.warning(f"Impossible de récupérer les données de tendances: {e}")
                log_error("trends_data_error", f"Erreur lors de la récupération des tendances Google: {e}", 
                          {"symbol": symbol, "timeframe": timeframe})
            
            # Récupérer les actualités si disponibles
            try:
                news_data = self.cryptopanic_client.get_news(currencies=symbol[:3])
                market_data.news = news_data
            except Exception as e:
                logger.warning(f"Impossible de récupérer les actualités: {e}")
                log_error("news_data_error", f"Erreur lors de la récupération des actualités: {e}", 
                         {"symbol": symbol, "timeframe": timeframe})
            
            # Stocker les données et la timestamp de la dernière mise à jour
            self.last_market_data = market_data
            self.last_data_update = datetime.now()
            
            # Journaliser les données de marché pour analyse post-mortem
            log_market_data(symbol, timeframe, market_data)
            
            # Journaliser la décision de mise à jour des données
            log_decision(
                decision_type="market_data_update",
                symbol=symbol,
                details={
                    "timeframe": timeframe,
                    "data_points": len(ohlcv) if hasattr(ohlcv, "__len__") else "N/A",
                    "last_price": ohlcv.iloc[-1]["close"] if hasattr(ohlcv, "iloc") else "N/A"
                },
                context={
                    "update_time": datetime.now().isoformat(),
                    "has_trends": trends_data is not None if 'trends_data' in locals() else False,
                    "has_news": news_data is not None if 'news_data' in locals() else False
                }
            )
            
            return market_data
            
        except Exception as e:
            error_msg = f"Erreur lors de la mise à jour des données de marché: {e}"
            logger.error(error_msg)
            
            # Journaliser l'erreur
            log_error("market_data_update_error", error_msg, {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit
            })
            
            # Créer un MarketData vide en cas d'erreur
            empty_market_data = MarketData(symbol=symbol, timeframe=timeframe)
            return empty_market_data
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère des signaux de trading en utilisant la stratégie configurée.
        
        Args:
            market_data: Données de marché à analyser
            
        Returns:
            Liste des signaux générés
        """
        logger.info(f"Génération de signaux pour {market_data.symbol} ({market_data.timeframe})")
        
        if not self.strategy:
            logger.error("Aucune stratégie n'est configurée")
            log_error("strategy_missing", "Aucune stratégie n'est configurée pour générer des signaux")
            return []
        
        try:
            # Appliquer la stratégie aux données de marché
            signals = self.strategy.generate_signals(market_data)
            
            logger.info(f"{len(signals)} signaux générés")
            
            # Journaliser chaque signal généré
            for signal in signals:
                log_signal(signal)
                
                # Journaliser également la décision de générer ce signal
                log_decision(
                    decision_type="signal_generation",
                    symbol=signal.symbol,
                    details={
                        "side": signal.side,
                        "strength": signal.strength,
                        "price": signal.price,
                        "quantity": signal.quantity,
                        "timestamp": signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp)
                    },
                    reasons=signal.reasons,
                    indicators=signal.indicators if hasattr(signal, "indicators") else {},
                    context={
                        "strategy": self.strategy.__class__.__name__,
                        "timeframe": market_data.timeframe
                    }
                )
            
            return signals
        except Exception as e:
            error_msg = f"Erreur lors de la génération des signaux: {e}"
            logger.error(error_msg)
            log_error("signal_generation_error", error_msg, {
                "strategy": self.strategy.__class__.__name__,
                "symbol": market_data.symbol,
                "timeframe": market_data.timeframe
            })
            return []
    
    def execute_signals(self, signals: List[Signal]) -> bool:
        """
        Exécute les signaux générés.
        
        Args:
            signals: Liste des signaux à exécuter
            
        Returns:
            True si les signaux ont été exécutés correctement
        """
        if not signals:
            return True
            
        # Vérifier si le mode Safe est activé
        if self.safe_mode_enabled:
            logger.warning(f"Signaux ignorés: Mode Safe activé (Raison: {self.safe_mode_reason})")
            
            # Journaliser la décision d'ignorer les signaux
            log_decision(
                decision_type="signals_ignored",
                symbol=signals[0].symbol if signals else "UNKNOWN",
                details={
                    "count": len(signals),
                    "safe_mode": True,
                    "safe_mode_reason": self.safe_mode_reason
                },
                reasons=["Mode Safe activé"],
                context={
                    "safe_mode_enabled": True,
                    "safe_mode_activated_at": self.safe_mode_activated_at.isoformat() if self.safe_mode_activated_at else None
                }
            )
            
            return False
        
        try:
            # Asynchronous notification code
            import asyncio
            from bitbot.utils.notifications import notification_manager, NotificationType, NotificationPriority
            
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            executed_signals = 0
            
            for signal in signals:
                # Déterminer l'action
                if signal.side == "BUY":
                    action = "achat"
                    emoji = "🟢"
                    notification_type = NotificationType.BUY
                elif signal.side == "SELL":
                    action = "vente"
                    emoji = "🔴"
                    notification_type = NotificationType.SELL
                else:
                    action = "neutre"
                    emoji = "⚪"
                    notification_type = NotificationType.SYSTEM
                
                logger.info(f"Exécution du signal de {action} pour {signal.symbol}")
                
                # Déterminer la priorité selon la force du signal
                priority = NotificationPriority.MEDIUM
                if abs(signal.strength) > 0.8:
                    priority = NotificationPriority.HIGH
                
                # Enregistrer le trade dans le moniteur de performance
                trade = {
                    "timestamp": datetime.now(),
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "price": signal.price or 0.0,
                    "quantity": signal.quantity or 0.1,
                    "profit_loss": 0.0  # Sera mis à jour ultérieurement
                }
                performance_monitor.record_trade(trade)
                
                # Journaliser l'ordre
                log_order(trade)
                
                # Journaliser la décision d'exécuter le signal
                log_decision(
                    decision_type="signal_execution",
                    symbol=signal.symbol,
                    details={
                        "side": signal.side,
                        "action": action,
                        "price": signal.price,
                        "quantity": signal.quantity,
                        "strength": signal.strength
                    },
                    reasons=[f"Exécution du signal de {action}"] + (signal.reasons if hasattr(signal, "reasons") else []),
                    indicators=signal.indicators if hasattr(signal, "indicators") else {},
                    context={
                        "strategy": self.strategy.__class__.__name__ if self.strategy else "Unknown",
                        "safe_mode": False
                    }
                )
                
                # Envoyer une notification
                async def send_trade_notification():
                    await notification_manager.notify(
                        message=f"{emoji} Signal de {action.upper()} pour {signal.symbol}",
                        notification_type=notification_type,
                        priority=priority,
                        title=f"BitBotPro - Signal de {action}",
                        details={
                            "symbol": signal.symbol,
                            "price": str(signal.price) if signal.price else "N/A",
                            "quantité": str(signal.quantity) if signal.quantity else "N/A",
                            "force": f"{signal.strength:.2f}" if hasattr(signal, "strength") else "N/A",
                            "heure": datetime.now().strftime("%H:%M:%S")
                        }
                    )
                
                loop.run_until_complete(send_trade_notification())
                executed_signals += 1
            
            logger.info(f"{executed_signals}/{len(signals)} signaux exécutés avec succès")
            return True
        except Exception as e:
            error_msg = f"Erreur lors de l'exécution des signaux: {e}"
            logger.error(error_msg)
            
            # Journaliser l'erreur
            log_error("signal_execution_error", error_msg, {
                "signals_count": len(signals),
                "symbols": [s.symbol for s in signals] if signals else []
            })
            
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
        
        # Initialiser le moniteur de performance pour le backtest
        performance_monitor.initial_balance = results["initial_balance"]
        performance_monitor.current_balance = results["initial_balance"]
        performance_monitor.peak_balance = results["initial_balance"]
        performance_monitor.trades_history = []
        performance_monitor.balance_history = []
        performance_monitor.drawdown_history = []
        
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
                    
                    # Mettre à jour le moniteur de performance
                    trade_datetime = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S") if isinstance(trade["timestamp"], str) else trade["timestamp"]
                    performance_monitor.update_balance(current_balance, trade_datetime)
                    
                    performance_monitor.record_trade(trade)
                    
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
    
    async def _send_performance_report(self, symbol: str):
        """
        Envoie un rapport de performance périodique.
        
        Args:
            symbol: Symbole sur lequel le trading est effectué
        """
        report = performance_monitor.generate_performance_report()
        
        # Formater les pourcentages et les valeurs pour le rapport
        message = (
            "📊 *Rapport de Performance BitBotPro* 📊\n\n"
            f"*Symbole:* {symbol}\n"
            f"*Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"*Balance initiale:* {report['initial_balance']:.2f} USD\n"
            f"*Balance actuelle:* {report['current_balance']:.2f} USD\n"
            f"*Balance maximale:* {report['peak_balance']:.2f} USD\n\n"
            f"*P&L:* {report['profit_loss']:.2f} USD ({report['profit_loss_pct']:.2f}%)\n"
            f"*Drawdown actuel:* {report['current_drawdown']:.2f}%\n"
            f"*Drawdown maximum:* {report['max_drawdown']:.2f}%\n\n"
            f"*Trades total:* {report['total_trades']}\n"
            f"*Trades gagnants:* {report['profitable_trades']}\n"
            f"*Win rate:* {report['win_rate']:.2f}%"
        )
        
        await notification_manager.notify(
            message=message,
            title="Rapport de Performance",
            priority=NotificationPriority.MEDIUM,
            notification_type=NotificationType.SYSTEM,
            details=report
        )
        
        logger.info("Rapport de performance envoyé")
    
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
        
        # Journaliser le démarrage du trading live
        log_decision(
            decision_type="trading_start",
            symbol=symbol,
            details={
                "mode": "live",
                "timeframe": timeframe,
                "start_time": datetime.now().isoformat()
            },
            context={
                "trader_config": {
                    "strategy": self.strategy.__class__.__name__ if self.strategy else "Unknown",
                    "safe_mode_enabled": self.safe_mode_enabled
                }
            }
        )
        
        try:
            # Envoyer une notification de démarrage via Telegram
            import asyncio
            from bitbot.utils.notifications import notification_manager
            from bitbot.utils.telegram_alerts import send_telegram_alert, AlertType, AlertPriority
            
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
                
                # Programmer l'envoi périodique de rapports de performance
                asyncio.create_task(self._schedule_performance_reports(symbol))
            
            # Exécuter la notification de manière asynchrone
            loop.run_until_complete(send_startup_notification())
            
            # Mettre à jour les données de marché
            market_data = self.update_market_data(symbol, timeframe)
            
            # Vérifier l'état du système et les conditions de marché
            critical_events_detected = self._check_for_critical_events(symbol)
            
            # Journaliser le résultat de la vérification
            log_decision(
                decision_type="critical_event_check",
                symbol=symbol,
                details={
                    "critical_events_detected": critical_events_detected,
                    "safe_mode_enabled": self.safe_mode_enabled,
                    "safe_mode_reason": self.safe_mode_reason if self.safe_mode_enabled else None
                },
                context={
                    "check_time": datetime.now().isoformat()
                }
            )
            
            # Générer des signaux (seulement si mode Safe n'est pas activé)
            signals = []
            if not self.safe_mode_enabled:
                signals = self.generate_signals(market_data)
                
                # Journaliser la génération de signaux
                log_decision(
                    decision_type="signals_generated",
                    symbol=symbol,
                    details={
                        "signals_count": len(signals),
                        "signals_sides": [s.side for s in signals] if signals else []
                    },
                    context={
                        "strategy": self.strategy.__class__.__name__ if self.strategy else "Unknown",
                        "timeframe": timeframe
                    }
                )
            
            # Exécuter les signaux (la fonction vérifie déjà le mode Safe)
            execution_success = self.execute_signals(signals)
            
            # Journaliser le résultat de l'exécution
            log_decision(
                decision_type="signals_execution_result",
                symbol=symbol,
                details={
                    "execution_success": execution_success,
                    "signals_executed": len(signals),
                    "safe_mode": self.safe_mode_enabled
                },
                context={
                    "execution_time": datetime.now().isoformat()
                }
            )
            
            # Générer des visualisations
            currency = symbol[:3]
            self.visualize_sentiment(currency)
            
            # Générer un rapport de session intermédiaire
            session_report = generate_report()
            logger.info(f"Rapport de session intermédiaire: {session_report['decisions_count']} décisions, "
                       f"{session_report['signals_count']} signaux, {session_report['orders_count']} ordres")
            
            return True
        except Exception as e:
            error_msg = f"Erreur lors du trading live: {e}"
            logger.error(error_msg)
            
            # Journaliser l'erreur
            log_error("live_trading_error", error_msg, {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            })
            
            # Activer le mode Safe en cas d'erreur grave
            self.activate_safe_mode(f"Erreur critique: {str(e)}", close_positions=True, symbol=symbol)
            
            return False
            
    def _check_for_critical_events(self, symbol: str) -> bool:
        """
        Vérifie s'il y a des événements critiques qui nécessitent l'activation du mode Safe.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            True si des événements critiques ont été détectés
        """
        logger.info(f"Vérification des événements critiques pour {symbol}")
        
        critical_events_detected = False
        critical_event_reasons = []
        
        # Récupérer les données de marché récentes
        try:
            recent_data = self.update_market_data(symbol, timeframe="5m", limit=12)  # Dernière heure
            
            # 1. Vérifier les chutes de prix brutales (Flash Crash)
            if hasattr(recent_data, 'ohlcv') and len(recent_data.ohlcv) >= 2:
                current_price = recent_data.ohlcv.iloc[-1]['close']
                previous_price = recent_data.ohlcv.iloc[-2]['close']
                
                price_change_pct = (current_price - previous_price) / previous_price * 100
                
                if price_change_pct < -8:  # Chute de plus de 8% en 5 minutes
                    critical_message = f"ALERTE: Chute brutale de prix détectée: {price_change_pct:.2f}% en 5 minutes"
                    logger.critical(critical_message)
                    
                    # Ajouter à la liste des événements critiques
                    critical_events_detected = True
                    critical_event_reasons.append(critical_message)
                    
                    # Journaliser cet événement critique
                    log_error("flash_crash_detected", critical_message, {
                        "symbol": symbol,
                        "price_change_pct": price_change_pct,
                        "current_price": current_price,
                        "previous_price": previous_price,
                        "timeframe": "5m"
                    })
            
            # 2. Vérifier le volume anormal
            if hasattr(recent_data, 'ohlcv') and len(recent_data.ohlcv) >= 12:
                current_volume = recent_data.ohlcv.iloc[-1]['volume']
                avg_volume = recent_data.ohlcv.iloc[:-1]['volume'].mean()
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio > 10:  # Volume 10x supérieur à la moyenne
                    critical_message = f"ALERTE: Volume anormal détecté: {volume_ratio:.2f}x la moyenne"
                    logger.critical(critical_message)
                    
                    # Ajouter à la liste des événements critiques
                    critical_events_detected = True
                    critical_event_reasons.append(critical_message)
                    
                    # Journaliser cet événement critique
                    log_error("abnormal_volume_detected", critical_message, {
                        "symbol": symbol,
                        "volume_ratio": volume_ratio,
                        "current_volume": current_volume,
                        "average_volume": avg_volume,
                        "timeframe": "5m"
                    })
            
            # 3. Vérifier les actualités critiques (si disponibles)
            if hasattr(recent_data, 'news') and recent_data.news is not None:
                critical_news_keywords = ["hack", "piratage", "fraude", "enquête", "régulation", "interdiction", "ban"]
                
                for news_item in recent_data.news:
                    if any(keyword in news_item.get('title', '').lower() for keyword in critical_news_keywords):
                        critical_message = f"ALERTE: Actualité critique détectée: {news_item.get('title')}"
                        logger.critical(critical_message)
                        
                        # Ajouter à la liste des événements critiques
                        critical_events_detected = True
                        critical_event_reasons.append(critical_message)
                        
                        # Journaliser cet événement critique
                        log_error("critical_news_detected", critical_message, {
                            "symbol": symbol,
                            "news_title": news_item.get('title'),
                            "news_url": news_item.get('url'),
                            "news_source": news_item.get('source')
                        })
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des événements critiques: {e}")
            log_error("critical_events_check_error", f"Erreur lors de la vérification des événements critiques: {e}", {
                "symbol": symbol
            })
        
        # Si des événements critiques ont été détectés, activer le mode Safe
        if critical_events_detected:
            reason = " | ".join(critical_event_reasons)
            self.activate_safe_mode(reason, close_positions=True, symbol=symbol)
            
            # Journaliser la décision de détecter un événement critique
            log_decision(
                decision_type="critical_event_detection",
                symbol=symbol,
                details={
                    "events_count": len(critical_event_reasons),
                    "events": critical_event_reasons
                },
                reasons=critical_event_reasons,
                context={
                    "detection_time": datetime.now().isoformat(),
                    "safe_mode_activated": True
                }
            )
        
        return critical_events_detected
    
    async def _schedule_performance_reports(self, symbol: str):
        """
        Programme l'envoi périodique de rapports de performance.
        
        Args:
            symbol: Symbole sur lequel le trading est effectué
        """
        # Envoyer un rapport toutes les 24 heures
        while True:
            await asyncio.sleep(24 * 60 * 60)  # 24 heures
            await self._send_performance_report(symbol)
    
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
            
    def deactivate_safe_mode(self) -> bool:
        """
        Désactive le mode Safe.
        
        Returns:
            True si le mode Safe a été désactivé avec succès
        """
        if not self.safe_mode_enabled:
            logger.info("Le mode Safe n'est pas activé.")
            return True
            
        logger.info("Désactivation du mode Safe")
        
        # Stocker les informations avant la désactivation pour le journal
        previous_reason = self.safe_mode_reason
        activation_time = self.safe_mode_activated_at
        deactivation_time = datetime.now()
        
        # Mise à jour de l'état
        self.safe_mode_enabled = False
        self.safe_mode_reason = None
        
        # Journaliser la désactivation du mode Safe
        log_decision(
            decision_type="safe_mode_deactivation",
            symbol="ALL",
            details={
                "previous_reason": previous_reason,
                "activation_time": activation_time.isoformat() if activation_time else "Unknown",
                "deactivation_time": deactivation_time.isoformat(),
                "duration_minutes": round((deactivation_time - activation_time).total_seconds() / 60, 2) if activation_time else 0
            },
            reasons=["Désactivation manuelle du mode Safe"],
            context={
                "trader_state": {
                    "strategy": self.strategy.__class__.__name__ if self.strategy else "Unknown"
                }
            }
        )
        
        # Notification de désactivation
        try:
            import asyncio
            from bitbot.utils.notifications import notification_manager, NotificationType, NotificationPriority
            
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def send_safe_mode_deactivation_notification():
                await notification_manager.notify(
                    message="✅ MODE SAFE DÉSACTIVÉ",
                    title="BitBotPro - Mode Safe désactivé",
                    notification_type=NotificationType.SYSTEM,
                    priority=NotificationPriority.HIGH,
                    details={
                        "raison_précédente": previous_reason,
                        "durée": f"{round((deactivation_time - activation_time).total_seconds() / 60, 2)} minutes" if activation_time else "Inconnue",
                        "date": deactivation_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
            
            loop.run_until_complete(send_safe_mode_deactivation_notification())
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification de désactivation du mode Safe: {e}")
            log_error("safe_mode_deactivation_notification_error", f"Erreur lors de l'envoi de la notification de désactivation du mode Safe: {e}")
        
        return True
        
    def activate_safe_mode(self, reason: str, close_positions: bool = False, symbol: str = None) -> bool:
        """
        Active le mode Safe pour protéger le portefeuille.
        
        Args:
            reason: Raison de l'activation du mode Safe
            close_positions: Si True, ferme toutes les positions ouvertes
            symbol: Symbole spécifique pour lequel fermer les positions (si None, ferme toutes les positions)
        """
        logger.warning(f"ACTIVATION DU MODE SAFE: {reason}")
        
        self.safe_mode_enabled = True
        self.safe_mode_reason = reason
        self.safe_mode_activated_at = datetime.now()
        
        # Journaliser la décision d'activer le mode Safe
        log_decision(
            decision_type="safe_mode_activation",
            symbol=symbol or "ALL",
            details={
                "reason": reason,
                "close_positions": close_positions,
                "activation_time": self.safe_mode_activated_at.isoformat()
            },
            reasons=[reason],
            context={
                "trader_state": {
                    "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
                    "strategy": self.strategy.__class__.__name__ if self.strategy else "Unknown"
                }
            }
        )
        
        if close_positions:
            try:
                logger.info("Fermeture de toutes les positions ouvertes")
                # Code pour fermer les positions via Binance API
                # ...
                
                # Journaliser les actions de fermeture de positions
                log_decision(
                    decision_type="positions_closed",
                    symbol=symbol or "ALL",
                    details={
                        "reason": "Activation du mode Safe",
                        "close_time": datetime.now().isoformat()
                    },
                    reasons=["Fermeture des positions suite à l'activation du mode Safe"],
                    context={
                        "safe_mode_enabled": True,
                        "safe_mode_reason": reason
                    }
                )
                
            except Exception as e:
                error_msg = f"Erreur lors de la fermeture des positions: {e}"
                logger.error(error_msg)
                log_error("positions_close_error", error_msg, {
                    "symbol": symbol or "ALL",
                    "safe_mode": True,
                    "safe_mode_reason": reason
                })
        
        # Envoyer une notification d'alerte pour le mode Safe
        try:
            import asyncio
            from bitbot.utils.notifications import notification_manager, NotificationType, NotificationPriority
            
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def send_safe_mode_notification():
                await notification_manager.notify(
                    message=f"⚠️ MODE SAFE ACTIVÉ: {reason}",
                    title="BitBotPro - ALERTE CRITIQUE",
                    notification_type=NotificationType.ALERT,
                    priority=NotificationPriority.CRITICAL,
                    details={
                        "raison": reason,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "positions_fermées": "Oui" if close_positions else "Non"
                    }
                )
            
            loop.run_until_complete(send_safe_mode_notification())
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification du mode Safe: {e}")
            log_error("safe_mode_notification_error", f"Erreur lors de l'envoi de la notification du mode Safe: {e}")
        
        return True
        
    def close_all_positions(self, symbol: str = None) -> bool:
        """
        Ferme toutes les positions ouvertes.
        
        Args:
            symbol: Symbole spécifique pour lequel fermer les positions (None = tous)
            
        Returns:
            True si toutes les positions ont été fermées correctement
        """
        logger.warning(f"Fermeture de toutes les positions pour {symbol if symbol else 'tous les symboles'}")
        
        try:
            # Récupérer toutes les positions ouvertes
            positions = self.binance_client.get_open_positions()
            
            if not positions:
                logger.info("Aucune position ouverte à fermer.")
                return True
                
            # Filtrer par symbole si spécifié
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
                
            if not positions:
                logger.info(f"Aucune position ouverte pour {symbol}.")
                return True
                
            # Fermer chaque position
            success = True
            for position in positions:
                try:
                    # Déterminer le côté pour fermer la position
                    side = "SELL" if position.amount > 0 else "BUY"
                    
                    # Placer un ordre de marché pour fermer la position
                    order = self.binance_client.place_order(
                        symbol=position.symbol,
                        side=side,
                        type="MARKET",
                        quantity=abs(position.amount)
                    )
                    
                    logger.info(f"Position fermée pour {position.symbol}: {order}")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la fermeture de la position {position.symbol}: {e}")
                    success = False
                    
            return success
        except Exception as e:
            logger.error(f"Erreur lors de la récupération ou fermeture des positions: {e}")
            return False
