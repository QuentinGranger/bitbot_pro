import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from strategies.deepseek_integration import DeepSeekAnalyzer
from binance.client import Client
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Gestionnaire de risque avancé avec analyse AI pour le trading de Bitcoin
    """
    
    def __init__(self,
                 max_position_risk: float = 0.05,      # 5% risque max par position
                 max_drawdown: float = 0.25,           # 25% drawdown maximum
                 volatility_lookback: int = 30,        # 30 périodes pour la volatilité
                 min_risk_reward_ratio: float = 3.0,   # Ratio risque/rendement minimum (1:3)
                 max_capital_per_trade: float = 0.01,  # 1% du capital max par trade
                 drawdown_thresholds: Dict[float, float] = None,  # Seuils de drawdown et leurs réductions
                 ai_enabled: bool = True):
        """
        Initialise le gestionnaire de risque
        
        Args:
            max_position_risk: Risque maximum par position individuelle
            max_drawdown: Drawdown maximum autorisé
            volatility_lookback: Nombre de périodes pour le calcul de la volatilité
            min_risk_reward_ratio: Ratio risque/rendement minimum requis (ex: 3.0 = 1:3)
            max_capital_per_trade: Pourcentage maximum du capital à risquer par trade
            drawdown_thresholds: Dict des seuils de drawdown et leurs réductions {seuil: réduction}
            ai_enabled: Activer l'analyse AI du risque
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        # Initialiser le client Binance
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            tld='com'
        )
        
        self.max_position_risk = max_position_risk
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.max_capital_per_trade = max_capital_per_trade
        # Seuils de drawdown par défaut
        self.drawdown_thresholds = drawdown_thresholds or {
            0.05: 0.20,  # -5% → réduction de 20%
            0.10: 0.50,  # -10% → réduction de 50%
            0.15: 0.75,  # -15% → réduction de 75%
            0.20: 1.00   # -20% → arrêt total (réduction de 100%)
        }
        self._initial_capital = None
        self._peak_capital = None
        self._consecutive_losses = 0  # Compteur de pertes consécutives
        self.ai_enabled = ai_enabled
        
        # Initialiser l'analyseur AI
        self.ai_analyzer = DeepSeekAnalyzer() if ai_enabled else None
        
        # État de la position BTC
        self._position = None
        self.portfolio_history = []  # Historique du portefeuille
        
    def get_current_btc_data(self, interval='1h', limit=100) -> pd.DataFrame:
        """
        Récupère les données BTC depuis Binance
        
        Args:
            interval: Intervalle des bougies
            limit: Nombre de bougies à récupérer
            
        Returns:
            DataFrame avec les données
        """
        try:
            # Récupérer les klines
            klines = self.client.get_klines(
                symbol='BTCUSDT',
                interval=interval,
                limit=limit
            )
            
            # Convertir en DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                'buy_quote_volume', 'ignore'
            ])
            
            # Convertir les types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données BTC: {str(e)}")
            return None
        
    def calculate_position_risk(self,
                              position_size: float,
                              entry_price: float,
                              stop_loss: float,
                              capital: float) -> float:
        """
        Calcule le risque d'une position
        
        Args:
            position_size: Taille de la position en BTC
            entry_price: Prix d'entrée
            stop_loss: Prix du stop loss
            capital: Capital total
            
        Returns:
            Risque en pourcentage du capital
        """
        if capital <= 0:
            return float('inf')
            
        if stop_loss >= entry_price:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
            
        total_risk = abs(risk_per_unit * position_size)
        return total_risk / capital
        
    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calcule le drawdown actuel
        
        Args:
            equity_curve: Série temporelle de l'équité
            
        Returns:
            Drawdown actuel en pourcentage
        """
        if len(equity_curve) == 0:
            return 0.0
            
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return abs(drawdown.min())
        
    def calculate_drawdown(self, current_capital: float) -> float:
        """
        Calcule le drawdown actuel
        
        Args:
            current_capital: Capital actuel
            
        Returns:
            Drawdown en pourcentage
        """
        # Initialiser le capital initial si nécessaire
        if self._initial_capital is None:
            self._initial_capital = current_capital
            self._peak_capital = current_capital
        
        # Mettre à jour le pic de capital
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital
        
        # Calculer le drawdown
        if self._peak_capital > 0:
            return (self._peak_capital - current_capital) / self._peak_capital
        return 0.0
        
    def get_drawdown_status(self, drawdown: float) -> Tuple[float, str, str]:
        """
        Obtient le statut du drawdown actuel
        
        Args:
            drawdown: Drawdown actuel en pourcentage
            
        Returns:
            Tuple[multiplicateur, message d'alerte, niveau de risque]
        """
        # Trouver le seuil applicable
        reduction = 0.0
        message = "Trading normal"
        risk_level = "normal"
        
        for threshold, red in sorted(self.drawdown_thresholds.items()):
            if drawdown >= threshold:
                reduction = red
                if reduction >= 1.0:
                    message = f"⛔ TRADING ARRÊTÉ - Drawdown critique de {drawdown:.1%}"
                    risk_level = "critical"
                else:
                    message = f"⚠️ Positions réduites de {reduction:.0%} - Drawdown de {drawdown:.1%}"
                    risk_level = "high" if reduction > 0.5 else "warning"
        
        return (1.0 - reduction), message, risk_level

    def get_drawdown_multiplier(self, current_capital: float) -> float:
        """
        Calcule le multiplicateur de position basé sur le drawdown
        
        Args:
            current_capital: Capital actuel
            
        Returns:
            Multiplicateur pour la taille de position (0 à 1)
        """
        drawdown = self.calculate_drawdown(current_capital)
        multiplier, _, _ = self.get_drawdown_status(drawdown)
        
        # Réduire davantage si pertes consécutives
        if self._consecutive_losses >= 3:
            multiplier *= 0.5
            
        return multiplier
        
    def calculate_risk_reward_ratio(self,
                                   entry_price: float,
                                   stop_loss: float,
                                   take_profit: float) -> float:
        """
        Calcule le ratio risque/rendement d'une position
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Prix du stop loss
            take_profit: Prix du take profit
            
        Returns:
            Ratio risque/rendement (>1 = favorable)
        """
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return 0.0
            
        reward = abs(take_profit - entry_price)
        return reward / risk
        
    def calculate_optimal_take_profit(self,
                                    entry_price: float,
                                    stop_loss: float,
                                    min_ratio: float = None) -> float:
        """
        Calcule un take profit optimal basé sur le ratio risque/rendement minimum
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Prix du stop loss
            min_ratio: Ratio minimum souhaité (utilise self.min_risk_reward_ratio si None)
            
        Returns:
            Prix du take profit optimal
        """
        if min_ratio is None:
            min_ratio = self.min_risk_reward_ratio
            
        risk = abs(entry_price - stop_loss)
        required_reward = risk * min_ratio
        
        # Position longue ou courte
        if stop_loss < entry_price:  # Long
            return entry_price + required_reward
        else:  # Short
            return entry_price - required_reward
        
    def validate_trade_setup(self,
                           entry_price: float,
                           stop_loss: float,
                           take_profit: float) -> Tuple[bool, str]:
        """
        Valide si un setup de trade respecte nos critères de risque
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Prix du stop loss
            take_profit: Prix du take profit
            
        Returns:
            Tuple[setup valide?, raison]
        """
        # Calculer le ratio R/R
        rr_ratio = self.calculate_risk_reward_ratio(
            entry_price, stop_loss, take_profit
        )
        
        # Vérifier le ratio minimum
        if rr_ratio < self.min_risk_reward_ratio:
            return False, f"Ratio R/R ({rr_ratio:.2f}) inférieur au minimum requis ({self.min_risk_reward_ratio:.2f})"
            
        return True, f"Setup valide avec ratio R/R de {rr_ratio:.2f}"

    def adjust_position_size(self,
                           base_size: float,
                           entry_price: float,
                           stop_loss: float,
                           take_profit: float,
                           capital: float) -> Tuple[float, Dict]:
        """
        Ajuste la taille de position en fonction des contraintes de risque
        
        Args:
            base_size: Taille de position suggérée en BTC
            entry_price: Prix d'entrée
            stop_loss: Stop loss
            take_profit: Take profit
            capital: Capital total
            
        Returns:
            Tuple[taille ajustée, dict des métriques]
        """
        # Récupérer les données récentes
        market_data = self.get_current_btc_data(interval='1h', limit=100)
        if market_data is None:
            raise Exception("Impossible de récupérer les données de marché")
        
        # Calculer le multiplicateur de drawdown
        drawdown_multiplier = self.get_drawdown_multiplier(capital)
        drawdown = self.calculate_drawdown(capital)
        _, drawdown_message, risk_level = self.get_drawdown_status(drawdown)
        
        # Si le drawdown est trop important, on arrête le trading
        if drawdown_multiplier == 0:
            logger.error(drawdown_message)
            return 0.0, {
                'initial_risk': 0,
                'adjusted_risk': 0,
                'volatility': 0,
                'risk_reward_ratio': 0,
                'max_position_size': 0,
                'position_value': 0,
                'capital_at_risk': 0,
                'drawdown': self.calculate_drawdown(capital),
                'trading_stopped': True,
                'risk_level': risk_level,
                'drawdown_message': drawdown_message,
                'ai_analysis': None
            }
        
        # Calculer la taille maximale basée sur le risque par trade
        max_size = self.calculate_max_position_size(
            entry_price, stop_loss, capital
        )
        
        # S'assurer que la taille de base ne dépasse pas le maximum
        if base_size > max_size:
            base_size = max_size
            
        # Calculer le risque initial
        initial_risk = self.calculate_position_risk(
            base_size, entry_price, stop_loss, capital
        )
        
        # Calculer le ratio R/R
        rr_ratio = self.calculate_risk_reward_ratio(
            entry_price, stop_loss, take_profit
        )
        
        # Ajuster la taille selon le ratio R/R
        adjusted_size = base_size
        if rr_ratio > self.min_risk_reward_ratio * 1.5:  # Excellent ratio
            adjusted_size *= 1.2  # Augmenter la position de 20%
        elif rr_ratio < self.min_risk_reward_ratio:  # Ratio insuffisant
            adjusted_size *= 0.8  # Réduire la position de 20%
            
        # Appliquer le multiplicateur de drawdown
        adjusted_size *= drawdown_multiplier
        
        # S'assurer que la taille ajustée ne dépasse pas le maximum
        if adjusted_size > max_size:
            adjusted_size = max_size
            
        # Ajuster si le risque est trop élevé
        if initial_risk > self.max_position_risk:
            adjusted_size *= self.max_position_risk / initial_risk
            
        # Calculer la volatilité
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualisé
        
        # Ajuster en fonction de la volatilité
        if volatility > 0.5:  # Haute volatilité
            adjusted_size *= 0.8  # Réduire la position
        elif volatility < 0.2:  # Basse volatilité
            adjusted_size *= 1.2  # Augmenter la position
            
        # Vérifier une dernière fois que nous ne dépassons pas le maximum
        if adjusted_size > max_size:
            adjusted_size = max_size
            
        # Obtenir l'analyse AI si activée
        ai_analysis = None
        if self.ai_enabled and self.ai_analyzer:
            try:
                market_context = f"""
                Price: ${entry_price:.2f}
                Stop Loss: ${stop_loss:.2f}
                Take Profit: ${take_profit:.2f}
                Position Size: {adjusted_size:.4f} BTC
                Volatility: {volatility:.2%}
                Risk: {initial_risk:.2%}
                """
                
                ai_analysis = self.ai_analyzer.analyze_market_sentiment([market_context])
                
                # Ajuster la taille en fonction du sentiment
                sentiment_score = ai_analysis.get('sentiment_score', 0)
                if abs(sentiment_score) > 0.7:  # Fort sentiment
                    adjusted_size *= (1 + 0.2 * np.sign(sentiment_score))
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse AI: {str(e)}")
                
        return adjusted_size, {
            'initial_risk': initial_risk,
            'adjusted_risk': self.calculate_position_risk(
                adjusted_size, entry_price, stop_loss, capital
            ),
            'volatility': volatility,
            'risk_reward_ratio': rr_ratio,
            'max_position_size': max_size,
            'position_value': self.calculate_position_value(adjusted_size, entry_price),
            'capital_at_risk': (self.calculate_position_value(adjusted_size, entry_price) / capital) * 100,
            'drawdown': self.calculate_drawdown(capital),
            'drawdown_multiplier': drawdown_multiplier,
            'risk_level': risk_level,
            'drawdown_message': drawdown_message,
            'consecutive_losses': self._consecutive_losses,
            'trading_stopped': False,
            'ai_analysis': ai_analysis
        }
        
    def calculate_max_position_size(self,
                                   entry_price: float,
                                   stop_loss: float,
                                   capital: float) -> float:
        """
        Calcule la taille maximale de position basée sur le risque par trade
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Stop loss
            capital: Capital total
            
        Returns:
            Taille maximale de la position en BTC
        """
        # Calculer le montant maximum à risquer
        max_risk_amount = capital * self.max_capital_per_trade
        
        # Calculer la perte par BTC
        risk_per_btc = abs(entry_price - stop_loss)
        
        # Calculer la taille maximale
        if risk_per_btc > 0:
            return max_risk_amount / risk_per_btc
        return 0.0
        
    def calculate_position_value(self,
                               position_size: float,
                               current_price: float) -> float:
        """
        Calcule la valeur d'une position
        
        Args:
            position_size: Taille de la position en BTC
            current_price: Prix actuel
            
        Returns:
            Valeur de la position en USD
        """
        return position_size * current_price
        
    def update_position(self,
                       position_size: float,
                       entry_price: float,
                       current_price: float,
                       stop_loss: float,
                       take_profit: float,
                       capital: float):
        """
        Met à jour la position BTC
        
        Args:
            position_size: Taille de la position en BTC
            entry_price: Prix d'entrée
            current_price: Prix actuel
            stop_loss: Stop loss
            take_profit: Take profit
            capital: Capital total
        """
        self._position = {
            'size': position_size,
            'entry_price': entry_price,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'value': position_size * current_price,
            'pnl': position_size * (current_price - entry_price),
            'risk': self.calculate_position_risk(
                position_size, entry_price, stop_loss, capital
            ),
            'risk_reward_ratio': self.calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profit
            )
        }
        
        # Mettre à jour l'historique
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': self._position['value']
        })
        
    def get_position_metrics(self) -> Dict:
        """
        Obtient les métriques actuelles de la position
        
        Returns:
            Dict des métriques
        """
        if not self._position:
            return {
                'position_value': 0.0,
                'pnl': 0.0,
                'risk': 0.0,
                'drawdown': 0.0
            }
            
        # Calculer le drawdown
        equity_curve = pd.Series([h['value'] for h in self.portfolio_history])
        if len(equity_curve) > 0:
            drawdown = self.calculate_historical_drawdown(equity_curve)
        else:
            drawdown = 0.0
        
        return {
            'position_value': self._position['value'],
            'pnl': self._position['pnl'],
            'risk': self._position['risk'],
            'drawdown': drawdown
        }
        
    def calculate_historical_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calcule le drawdown historique maximum
        
        Args:
            equity_curve: Série des valeurs du portefeuille
            
        Returns:
            Drawdown maximum en pourcentage
        """
        if len(equity_curve) > 0:
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            return abs(float(drawdown.min()))
        return 0.0
        
    def update_consecutive_losses(self, pnl: float):
        """
        Met à jour le compteur de pertes consécutives
        
        Args:
            pnl: Profit/perte de la dernière position
        """
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
            
        # Réduire les positions après 3 pertes consécutives
        if self._consecutive_losses >= 3:
            logger.warning(f"⚠️ {self._consecutive_losses} pertes consécutives - Positions réduites de 50%")

    def should_close_position(self,
                            current_price: float) -> Tuple[bool, str]:
        """
        Détermine si une position doit être fermée
        
        Args:
            current_price: Prix actuel
            
        Returns:
            Tuple[bool, str]: (fermer la position, raison)
        """
        if self._position is None:
            return False, "Pas de position ouverte"
            
        # Vérifier le stop loss
        if float(current_price) <= float(self._position['stop_loss']):
            return True, "Stop loss atteint"
            
        # Vérifier le take profit
        if float(current_price) >= float(self._position['take_profit']):
            return True, "Take profit atteint"
            
        # Vérifier le drawdown maximum
        position_metrics = self.get_position_metrics()
        if float(position_metrics['drawdown']) >= self.max_drawdown:
            return True, f"Drawdown maximum ({self.max_drawdown:.0%}) atteint"
            
        # Vérifier le sentiment AI si activé
        if self.ai_enabled and self.ai_analyzer:
            try:
                market_context = f"""
                Price: ${current_price:.2f}
                Stop Loss: ${self._position['stop_loss']:.2f}
                Take Profit: ${self._position['take_profit']:.2f}
                Position Size: {self._position['size']:.4f} BTC
                Drawdown: {position_metrics['drawdown']:.2%}
                """
                
                ai_analysis = self.ai_analyzer.analyze_market_sentiment(
                    [market_context]
                )
                 
                # Fermer si le sentiment est très négatif
                if ai_analysis.get('sentiment_score', 0) < -0.8:
                    return True, "Sentiment de marché très négatif"
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse AI: {str(e)}")
        
        return False, "Position maintenue"
