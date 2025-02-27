"""
Module de gestion du portefeuille pour BitBotPro.
"""

from typing import Dict, Optional, List
from datetime import datetime

class Portfolio:
    """
    Classe représentant un portefeuille de crypto-monnaies.
    """
    
    def __init__(self):
        """Initialise un nouveau portefeuille vide."""
        self.positions = {}  # type: Dict[str, float]
        self.transactions = []  # type: List[Dict]
    
    def add_position(self, symbol: str, quantity: float, price: Optional[float] = None) -> bool:
        """
        Ajoute une position au portefeuille.
        
        Args:
            symbol: Symbole de la crypto-monnaie
            quantity: Quantité à ajouter
            price: Prix d'achat (optionnel)
            
        Returns:
            True si la position a été ajoutée, False sinon
        """
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        
        # Enregistrer la transaction
        transaction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'type': 'BUY',
            'quantity': quantity,
            'price': price
        }
        self.transactions.append(transaction)
        
        return True
    
    def remove_position(self, symbol: str, quantity: Optional[float] = None, price: Optional[float] = None) -> bool:
        """
        Supprime une position du portefeuille.
        
        Args:
            symbol: Symbole de la crypto-monnaie
            quantity: Quantité à retirer (si None, retire toute la position)
            price: Prix de vente (optionnel)
            
        Returns:
            True si la position a été retirée, False sinon
        """
        if symbol not in self.positions:
            return False
        
        current_quantity = self.positions[symbol]
        
        if quantity is None or quantity >= current_quantity:
            # Retirer toute la position
            quantity_to_remove = current_quantity
            del self.positions[symbol]
        else:
            # Retirer une partie de la position
            quantity_to_remove = quantity
            self.positions[symbol] -= quantity
        
        # Enregistrer la transaction
        transaction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'type': 'SELL',
            'quantity': quantity_to_remove,
            'price': price
        }
        self.transactions.append(transaction)
        
        return True
    
    def get_position(self, symbol: str) -> float:
        """
        Récupère la quantité d'une position.
        
        Args:
            symbol: Symbole de la crypto-monnaie
            
        Returns:
            Quantité de la position (0 si la position n'existe pas)
        """
        return self.positions.get(symbol, 0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """
        Récupère toutes les positions du portefeuille.
        
        Returns:
            Dictionnaire des positions
        """
        return self.positions.copy()
    
    def get_transactions(self) -> List[Dict]:
        """
        Récupère l'historique des transactions.
        
        Returns:
            Liste des transactions
        """
        return self.transactions.copy()
