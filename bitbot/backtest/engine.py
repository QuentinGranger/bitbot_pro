"""
Moteur de backtest pour BitBot Pro.
Simule le trading avec conditions réelles (latence, slippage) et analyse les performances.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bitbot.models.position import Position
from bitbot.models.signal import Signal, SignalType
from bitbot.trading.strategy import BaseStrategy
from bitbot.utils.logger import logger

@dataclass
class BacktestConfig:
    """Configuration du backtest."""
    initial_balance: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    latency: int = 500  # ms
    leverage: float = 1.0
    position_size: float = 0.1  # % du capital par position
    max_positions: int = 3
    stop_loss: float = 0.02  # 2%
    take_profit: float = 0.06  # 6%

class BacktestResult:
    """Résultats du backtest avec métriques et analyses."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: pd.Series = None
        self.positions: List[Position] = []
        self.metrics: Dict = {}
    
    def calculate_metrics(self):
        """Calcule les métriques de performance."""
        if not self.trades:
            return
        
        # Convertir en DataFrame pour faciliter l'analyse
        df = pd.DataFrame(self.trades)
        
        # Métriques de base
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # Calcul des ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calcul du drawdown
        cumulative_returns = (1 + df['pnl'].cumsum())
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
        
        # Calcul du Sharpe Ratio (supposant rendements quotidiens)
        daily_returns = df.groupby(df['exit_time'].dt.date)['pnl'].sum()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 else 0
        
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': (cumulative_returns.iloc[-1] - 1) if not cumulative_returns.empty else 0,
            'avg_trade_duration': (df['exit_time'] - df['entry_time']).mean().total_seconds() / 60  # en minutes
        }
    
    def plot(self, filename: Optional[str] = None):
        """
        Génère une visualisation interactive des résultats avec Plotly.
        
        Args:
            filename: Si spécifié, sauvegarde le graphique en HTML
        """
        # Créer le graphique avec sous-graphiques
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Prix et Positions', 'Equity Curve', 'Drawdown'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Graphique des prix avec positions
        df_trades = pd.DataFrame(self.trades)
        
        fig.add_trace(
            go.Candlestick(
                x=self.equity_curve.index,
                open=self.equity_curve['open'],
                high=self.equity_curve['high'],
                low=self.equity_curve['low'],
                close=self.equity_curve['close'],
                name='Prix'
            ),
            row=1, col=1
        )
        
        # Ajouter les entrées/sorties
        for _, trade in df_trades.iterrows():
            # Entrée
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade['side'] == 'LONG' else 'triangle-down',
                        size=12,
                        color='green' if trade['side'] == 'LONG' else 'red'
                    ),
                    name=f"{trade['side']} Entry"
                ),
                row=1, col=1
            )
            
            # Sortie
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_time']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='red' if trade['side'] == 'LONG' else 'green'
                    ),
                    name=f"{trade['side']} Exit"
                ),
                row=1, col=1
            )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve['equity'],
                name='Equity',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # Drawdown
        drawdown = (self.equity_curve['equity'] - self.equity_curve['equity'].expanding().max()) / self.equity_curve['equity'].expanding().max()
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # Mise en forme
        fig.update_layout(
            title='Résultats du Backtest',
            xaxis_title='Date',
            yaxis_title='Prix',
            yaxis2_title='Equity',
            yaxis3_title='Drawdown',
            height=1200
        )
        
        # Ajouter les métriques dans une annotation
        metrics_text = (
            f"Win Rate: {self.metrics['win_rate']:.2%}<br>"
            f"Profit Factor: {self.metrics['profit_factor']:.2f}<br>"
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}<br>"
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}<br>"
            f"Total Return: {self.metrics['total_return']:.2%}"
        )
        
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        if filename:
            fig.write_html(filename)
        
        return fig

class BacktestEngine:
    """Moteur de backtest avec simulation des conditions réelles."""
    
    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: Configuration du backtest
        """
        self.config = config
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.positions: List[Position] = []
        self.results = BacktestResult()
    
    def _apply_slippage(self, price: float, side: SignalType) -> float:
        """
        Applique le slippage au prix.
        
        Args:
            price: Prix théorique
            side: Type d'ordre (LONG/SHORT)
        
        Returns:
            Prix avec slippage
        """
        slippage_factor = 1 + (self.config.slippage * (1 if side == SignalType.LONG else -1))
        return price * slippage_factor
    
    def _simulate_latency(self, timestamp: datetime) -> datetime:
        """
        Simule la latence réseau.
        
        Args:
            timestamp: Horodatage original
        
        Returns:
            Horodatage avec latence
        """
        return timestamp + timedelta(milliseconds=self.config.latency)
    
    def _calculate_position_size(self, price: float) -> float:
        """
        Calcule la taille de position optimale.
        
        Args:
            price: Prix d'entrée
        
        Returns:
            Taille de la position en unités
        """
        position_value = self.balance * self.config.position_size
        return position_value / price
    
    async def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """
        Exécute le backtest.
        
        Args:
            data: Données historiques (OHLCV)
            strategy: Stratégie à tester
        
        Returns:
            Résultats du backtest
        """
        equity_history = []
        
        for i in range(len(data)):
            timestamp = data.index[i]
            current_bar = data.iloc[i]
            
            # Mise à jour de l'equity
            self.equity = self.balance + sum(
                pos.amount * current_bar['close'] * (1 if pos.side == SignalType.LONG else -1)
                for pos in self.positions
            )
            equity_history.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close']
            })
            
            # Vérifier les positions existantes
            for position in self.positions[:]:
                # Simuler la latence pour la vérification
                check_time = self._simulate_latency(timestamp)
                
                # Vérifier stop loss et take profit
                if position.side == SignalType.LONG:
                    if current_bar['low'] <= position.stop_loss:
                        self._close_position(position, position.stop_loss, check_time)
                    elif current_bar['high'] >= position.take_profit:
                        self._close_position(position, position.take_profit, check_time)
                else:  # SHORT
                    if current_bar['high'] >= position.stop_loss:
                        self._close_position(position, position.stop_loss, check_time)
                    elif current_bar['low'] <= position.take_profit:
                        self._close_position(position, position.take_profit, check_time)
            
            # Analyser pour nouveaux signaux
            lookback = data.iloc[max(0, i-100):i+1]  # Utiliser 100 barres pour l'analyse
            signal = await strategy.analyze(lookback)
            
            if signal and len(self.positions) < self.config.max_positions:
                # Simuler la latence pour l'entrée
                entry_time = self._simulate_latency(timestamp)
                entry_price = self._apply_slippage(current_bar['close'], signal.signal_type)
                
                # Calculer la taille de position
                size = self._calculate_position_size(entry_price)
                
                # Créer la position
                position = Position(
                    symbol=signal.symbol,
                    entry_price=entry_price,
                    amount=size,
                    side=signal.signal_type,
                    entry_time=entry_time,
                    stop_loss=entry_price * (1 - self.config.stop_loss if signal.signal_type == SignalType.LONG else 1 + self.config.stop_loss),
                    take_profit=entry_price * (1 + self.config.take_profit if signal.signal_type == SignalType.LONG else 1 - self.config.take_profit)
                )
                
                self.positions.append(position)
        
        # Fermer les positions restantes
        for position in self.positions[:]:
            self._close_position(position, data.iloc[-1]['close'], data.index[-1])
        
        # Préparer les résultats
        self.results.equity_curve = pd.DataFrame(equity_history).set_index('timestamp')
        self.results.calculate_metrics()
        
        return self.results
    
    def _close_position(self, position: Position, price: float, timestamp: datetime):
        """
        Ferme une position et enregistre le trade.
        
        Args:
            position: Position à fermer
            price: Prix de sortie
            timestamp: Horodatage de sortie
        """
        exit_price = self._apply_slippage(price, SignalType.SHORT if position.side == SignalType.LONG else SignalType.LONG)
        
        # Calculer le P&L
        if position.side == SignalType.LONG:
            pnl = (exit_price - position.entry_price) * position.amount
        else:
            pnl = (position.entry_price - exit_price) * position.amount
        
        # Soustraire les commissions
        commission = (position.entry_price + exit_price) * position.amount * self.config.commission
        pnl -= commission
        
        # Mettre à jour le solde
        self.balance += pnl
        
        # Enregistrer le trade
        self.results.trades.append({
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'symbol': position.symbol,
            'side': position.side.name,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'amount': position.amount,
            'pnl': pnl,
            'commission': commission,
            'duration': (timestamp - position.entry_time).total_seconds() / 60  # en minutes
        })
        
        # Retirer la position
        self.positions.remove(position)
