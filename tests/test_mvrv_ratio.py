"""
Tests pour l'indicateur MVRV (Market Value to Realized Value).

Ces tests vérifient le bon fonctionnement de l'indicateur MVRV, notamment:
- Calcul du ratio MVRV
- Génération de signaux
- Mécanisme de fallback vers NUPL
- Ajustement de la pondération en fonction de la volatilité
- Analyse des carnets d'ordres
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from bitbot.strategie.base.MVRVRatio import MVRVIndicator, MVRVSignal
from bitbot.data.onchain_client import OnChainClient

class TestMVRVRatio(unittest.TestCase):
    """Tests pour l'indicateur MVRV (Market Value to Realized Value)."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        self.mvrv = MVRVIndicator(
            ema_period=14,
            undervalued_threshold=1.0,
            strong_undervalued_threshold=0.75,
            overvalued_threshold=2.5,
            strong_overvalued_threshold=3.5,
            use_fallback=True,
            volatility_threshold=0.02,
            consider_orderbook=True
        )
        
        # Création de données de test
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        # Données MVRV
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'price': np.linspace(10000, 20000, 30),  # Prix croissant
            'market_cap': np.linspace(200e9, 400e9, 30),  # Cap de marché croissante
            'realized_cap': np.linspace(150e9, 250e9, 30),  # Cap réalisée croissante
            'mvrv_ratio': np.linspace(0.8, 3.0, 30)  # MVRV croissant
        })
        
        # Ajouter l'EMA
        self.test_data['mvrv_ema'] = self.test_data['mvrv_ratio'].ewm(span=14, adjust=False).mean()
        
    def test_get_signal(self):
        """Test de la génération des signaux MVRV."""
        # Tester tous les signaux possibles
        
        # Signal STRONG_UNDERVALUED
        df_undervalued = self.test_data.copy()
        df_undervalued['mvrv_ratio'] = 0.5  # Sous le seuil de forte sous-évaluation
        signal = self.mvrv.get_signal(df_undervalued)
        self.assertEqual(signal, MVRVSignal.STRONG_UNDERVALUED)
        
        # Signal UNDERVALUED
        df_undervalued = self.test_data.copy()
        df_undervalued['mvrv_ratio'] = 0.8  # Entre les seuils de sous-évaluation
        signal = self.mvrv.get_signal(df_undervalued)
        self.assertEqual(signal, MVRVSignal.UNDERVALUED)
        
        # Signal NEUTRAL
        df_neutral = self.test_data.copy()
        df_neutral['mvrv_ratio'] = 1.5  # Entre les seuils
        signal = self.mvrv.get_signal(df_neutral)
        self.assertEqual(signal, MVRVSignal.NEUTRAL)
        
        # Signal OVERVALUED
        df_overvalued = self.test_data.copy()
        df_overvalued['mvrv_ratio'] = 3.0  # Entre les seuils de surévaluation
        signal = self.mvrv.get_signal(df_overvalued)
        self.assertEqual(signal, MVRVSignal.OVERVALUED)
        
        # Signal STRONG_OVERVALUED
        df_overvalued = self.test_data.copy()
        df_overvalued['mvrv_ratio'] = 4.0  # Au-dessus du seuil de forte surévaluation
        signal = self.mvrv.get_signal(df_overvalued)
        self.assertEqual(signal, MVRVSignal.STRONG_OVERVALUED)
    
    def test_is_undervalued(self):
        """Test de la détection de sous-évaluation."""
        # Cas sous-évalué
        df_undervalued = self.test_data.copy()
        df_undervalued['mvrv_ratio'] = 0.9  # Sous le seuil de sous-évaluation
        self.assertTrue(self.mvrv.is_undervalued(df_undervalued))
        
        # Cas non sous-évalué
        df_not_undervalued = self.test_data.copy()
        df_not_undervalued['mvrv_ratio'] = 1.5  # Au-dessus du seuil de sous-évaluation
        self.assertFalse(self.mvrv.is_undervalued(df_not_undervalued))
    
    def test_is_overvalued(self):
        """Test de la détection de surévaluation."""
        # Cas surévalué
        df_overvalued = self.test_data.copy()
        df_overvalued['mvrv_ratio'] = 3.0  # Au-dessus du seuil de surévaluation
        self.assertTrue(self.mvrv.is_overvalued(df_overvalued))
        
        # Cas non surévalué
        df_not_overvalued = self.test_data.copy()
        df_not_overvalued['mvrv_ratio'] = 2.0  # Sous le seuil de surévaluation
        self.assertFalse(self.mvrv.is_overvalued(df_not_overvalued))
    
    @patch('bitbot.data.onchain_client.OnChainClient.get_approximate_mvrv_ratio')
    @patch('bitbot.data.onchain_client.OnChainClient.get_nupl_data')
    def test_fallback_to_nupl(self, mock_get_nupl_data, mock_get_approximate_mvrv_ratio):
        """Test du fallback vers NUPL quand MVRV échoue."""
        # Simuler un échec de MVRV
        mock_get_approximate_mvrv_ratio.return_value = pd.DataFrame()
        
        # Créer des données de test pour NUPL
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        nupl_data = pd.DataFrame({
            'timestamp': dates,
            'price': np.linspace(10000, 20000, 30),
            'market_cap': np.linspace(200e9, 400e9, 30),
            'realized_cap': np.linspace(150e9, 250e9, 30),
            'nupl': np.linspace(-0.2, 0.6, 30)
        })
        
        # Catégoriser NUPL
        nupl_data['nupl_category'] = pd.cut(
            nupl_data['nupl'],
            bins=[-1, -0.25, 0, 0.25, 0.5, 0.75, 1],
            labels=['Capitulation', 'Peur', 'Espoir', 'Optimisme', 'Euphorie', 'Avidité']
        )
        
        # Simuler des données NUPL réussies
        mock_get_nupl_data.return_value = nupl_data
        
        # Appeler la méthode qui devrait utiliser le fallback
        result = self.mvrv.get_mvrv_data(asset="BTC", days=30)
        
        # Vérifier que get_nupl_data a été appelé
        mock_get_nupl_data.assert_called_once()
        
        # Vérifier que les données NUPL ont été correctement converties en format MVRV
        self.assertFalse(result.empty)
        self.assertIn('mvrv_ratio', result.columns)
        self.assertIn('nupl', result.columns)
        self.assertIn('nupl_category', result.columns)
    
    def test_adjust_volatility_weight(self):
        """Test de l'ajustement de pondération en fonction de la volatilité."""
        # Cas de haute volatilité (pas de réduction de poids)
        df_high_volatility = self.test_data.copy()
        # Créer des variations de prix qui donnent une forte volatilité
        high_volatility_prices = [10000]
        for i in range(1, 30):
            # Générer des variations de prix avec des fluctuations importantes
            change = np.random.uniform(-0.05, 0.05)  # +/- 5% de fluctuation
            high_volatility_prices.append(high_volatility_prices[-1] * (1 + change))
        
        df_high_volatility['price'] = high_volatility_prices
        
        # Ajuster le poids selon la volatilité
        self.mvrv._adjust_volatility_weight(df_high_volatility)
        
        # Vérifier que pour une forte volatilité, le poids est conservé à 1.0
        self.assertAlmostEqual(self.mvrv.volatility_weight, 1.0, places=1)
        
        # Cas de faible volatilité (réduction de poids)
        df_low_volatility = self.test_data.copy()
        # Créer des variations de prix qui donnent une faible volatilité
        low_volatility_prices = [10000]
        for i in range(1, 30):
            # Générer des variations de prix avec des fluctuations minimes
            change = np.random.uniform(-0.001, 0.001)  # +/- 0.1% de fluctuation
            low_volatility_prices.append(low_volatility_prices[-1] * (1 + change))
        
        df_low_volatility['price'] = low_volatility_prices
        
        # Ajuster le poids selon la volatilité
        self.mvrv._adjust_volatility_weight(df_low_volatility)
        
        # Vérifier que pour une faible volatilité, le poids est réduit
        self.assertLess(self.mvrv.volatility_weight, 1.0)
    
    def test_analyze_orderbooks(self):
        """Test de l'analyse des carnets d'ordres."""
        result = self.mvrv._analyze_orderbooks(asset="BTC")
        
        # Vérifier que l'analyse des carnets d'ordres retourne bien un dictionnaire
        self.assertIsInstance(result, dict)
        
        # Vérifier la présence des clés attendues
        self.assertIn('orderbook_analyzed', result)
        self.assertIn('buy_walls_detected', result)
        self.assertIn('sell_walls_detected', result)
        self.assertIn('orderbook_note', result)
        
        # Vérifier que le flag d'analyse est à True
        self.assertTrue(result['orderbook_analyzed'])

if __name__ == '__main__':
    unittest.main()
