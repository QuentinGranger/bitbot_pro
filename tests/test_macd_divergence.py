"""
Tests pour la stratégie de divergence MACD.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bitbot.strategie.indicators.macd_divergence_strategy import MACDDivergenceStrategy, DivergenceType
from bitbot.strategie.base.MACD import MACDSignalType

class TestMACDDivergenceStrategy(unittest.TestCase):
    """Tests de la stratégie de divergence MACD."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer un jeu de données de test
        self.dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        
        # Données de base
        self.test_data = pd.DataFrame({
            'open': np.ones(100),
            'high': np.ones(100) * 101,
            'low': np.ones(100) * 99,
            'close': np.ones(100) * 100,
            'volume': np.ones(100) * 1000
        }, index=self.dates)
        
        # Stratégie avec paramètres par défaut
        self.strategy = MACDDivergenceStrategy()
        
        # Stratégie sans filtre de volatilité
        self.strategy_no_filter = MACDDivergenceStrategy(use_volatility_filter=False)
    
    def test_divergence_detection_bullish_regular(self):
        """Test de détection d'une divergence haussière régulière."""
        # Créer un prix qui fait des creux de plus en plus bas
        prices = np.array([100, 105, 95, 100, 90, 95])
        
        # Créer une copie locale des données pour ce test
        test_data = self.test_data.copy()
        
        # Mettre à jour les 6 dernières valeurs des prix
        for i in range(6):
            test_data.iloc[-6+i, test_data.columns.get_indexer(['close'])[0]] = prices[i]
        
        # Mais un MACD qui fait des creux de plus en plus hauts
        # Simuler cela en modifiant directement le DataFrame après calcul
        df = self.strategy.macd_indicator.calculate_macd(test_data)
        
        # Modifier les dernières valeurs du MACD
        macd_values = np.array([-2.0, -1.0, -3.0, -1.5, -2.5, -1.0])
        for i in range(6):
            df.loc[df.index[-6+i], 'macd'] = macd_values[i]
        
        # Tester la détection des sommets et creux
        price_peaks, price_troughs = self.strategy._find_peaks_and_troughs(df['close'].iloc[-30:])
        macd_peaks, macd_troughs = self.strategy._find_peaks_and_troughs(df['macd'].iloc[-30:])
        
        # Vérifier qu'on a bien trouvé des sommets et des creux
        self.assertGreater(len(price_peaks) + len(price_troughs), 0)
        self.assertGreater(len(macd_peaks) + len(macd_troughs), 0)
    
    def test_divergence_detection_bearish_regular(self):
        """Test de détection d'une divergence baissière régulière."""
        # Créer un prix qui fait des sommets de plus en plus hauts
        prices = np.array([100, 95, 105, 100, 110, 105])
        
        # Créer une copie locale des données pour ce test
        test_data = self.test_data.copy()
        
        # Mettre à jour les 6 dernières valeurs des prix
        for i in range(6):
            test_data.iloc[-6+i, test_data.columns.get_indexer(['close'])[0]] = prices[i]
        
        # Mais un MACD qui fait des sommets de plus en plus bas
        # Simuler cela en modifiant directement le DataFrame après calcul
        df = self.strategy.macd_indicator.calculate_macd(test_data)
        
        # Modifier les dernières valeurs du MACD
        macd_values = np.array([2.0, 1.0, 3.0, 1.5, 2.5, 1.0])
        for i in range(6):
            df.loc[df.index[-6+i], 'macd'] = macd_values[i]
        
        # Tester la détection des sommets et creux
        price_peaks, price_troughs = self.strategy._find_peaks_and_troughs(df['close'].iloc[-30:])
        macd_peaks, macd_troughs = self.strategy._find_peaks_and_troughs(df['macd'].iloc[-30:])
        
        # Vérifier qu'on a bien trouvé des sommets et des creux
        self.assertGreater(len(price_peaks) + len(price_troughs), 0)
        self.assertGreater(len(macd_peaks) + len(macd_troughs), 0)
    
    def test_volatility_filter(self):
        """Test du filtre de volatilité."""
        # Créer une série de prix avec une faible volatilité
        test_data_low_vol = self.test_data.copy()
        test_data_low_vol['high'] = test_data_low_vol['close'] * 1.001  # Volatilité très faible
        test_data_low_vol['low'] = test_data_low_vol['close'] * 0.999
        
        # Tester le filtre de volatilité (qui devrait rejeter les signaux)
        strategy_strict = MACDDivergenceStrategy(atr_threshold_pct=1.0)  # Seuil de volatilité élevé
        volatility_check = strategy_strict.check_volatility(test_data_low_vol)
        self.assertFalse(volatility_check['is_valid'])
        
        # Créer une série de prix avec une volatilité élevée
        test_data_high_vol = self.test_data.copy()
        test_data_high_vol['high'] = test_data_high_vol['close'] * 1.05
        test_data_high_vol['low'] = test_data_high_vol['close'] * 0.95
        
        # Tester le filtre de volatilité (qui devrait accepter les signaux)
        volatility_check = self.strategy.check_volatility(test_data_high_vol)
        self.assertTrue(volatility_check['is_valid'])
        
        # Tester sans filtre de volatilité (toujours valide)
        volatility_check = self.strategy_no_filter.check_volatility(test_data_high_vol)
        self.assertTrue(volatility_check['is_valid'])
    
    def test_signal_generation(self):
        """Test de la génération de signaux."""
        # Créer un jeu de données avec des prix qui suivent une tendance puis inversent
        n = 100
        prices = np.zeros(n)
        
        # Créer une tendance haussière puis baissière
        for i in range(n):
            if i < n//3:
                prices[i] = 100 + i * 0.5
            elif i < 2*n//3:
                prices[i] = 100 + (n//3) * 0.5 - (i - n//3) * 0.1
            else:
                prices[i] = 100 + (n//3) * 0.5 - (2*n//3 - n//3) * 0.1 + (i - 2*n//3) * 0.3
        
        # Créer une copie locale des données pour ce test
        test_data = self.test_data.copy()
        test_data['close'] = prices
        test_data['high'] = test_data['close'] * 1.01
        test_data['low'] = test_data['close'] * 0.99
        
        # Calculer les signaux historiques avec la stratégie sans filtre de volatilité
        df = self.strategy_no_filter.calculate_signals_historical(test_data)
        
        # Vérifier que le DataFrame contient les colonnes attendues
        self.assertIn('signal', df.columns)
        self.assertIn('macd', df.columns)
        self.assertIn('atr', df.columns)

if __name__ == '__main__':
    unittest.main()
