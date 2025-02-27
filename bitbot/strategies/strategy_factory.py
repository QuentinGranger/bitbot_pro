"""
Factory pour créer des stratégies de trading.
"""

from typing import Dict, Any, List
from bitbot.models.market_data import MarketData, Signal

class BaseStrategy:
    """Classe de base pour toutes les stratégies de trading."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialise la stratégie avec des paramètres.
        
        Args:
            params: Paramètres de la stratégie
        """
        self.params = params or {}
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les données de marché.
        
        Args:
            market_data: Données de marché à analyser
            
        Returns:
            Liste des signaux générés
        """
        # À implémenter dans les sous-classes
        return []

class DummyStrategy(BaseStrategy):
    """Stratégie factice pour les tests."""
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère des signaux de trading factices pour les tests.
        
        Args:
            market_data: Données de marché à analyser
            
        Returns:
            Liste des signaux générés
        """
        # Pour les tests, on retourne une liste vide
        return []

class StrategyFactory:
    """Factory pour créer des stratégies de trading."""
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Dict[str, Any] = None) -> BaseStrategy:
        """
        Crée une instance de stratégie basée sur le nom.
        
        Args:
            strategy_name: Nom de la stratégie à créer
            params: Paramètres de la stratégie
            
        Returns:
            Instance de la stratégie
        """
        # Pour l'instant, on ne retourne qu'une stratégie factice
        return DummyStrategy(params)
