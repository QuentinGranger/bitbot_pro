"""
Package d'agrégation des signaux pour la plateforme BitBotPro.
"""

from .signal_aggregator import SignalAggregator, Signal, AggregatedSignal, SignalCategory, SignalStrength

__all__ = ['SignalAggregator', 'Signal', 'AggregatedSignal', 'SignalCategory', 'SignalStrength']
