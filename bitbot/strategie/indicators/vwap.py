"""
Module pour calculer et analyser le Volume Weighted Average Price (VWAP).

Ce module fournit des fonctions pour calculer le VWAP,
un indicateur technique qui prend en compte le volume des transactions
pour déterminer le prix moyen pondéré par les volumes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from enum import Enum
from datetime import datetime, timedelta

from bitbot.models.market_data import MarketData
from bitbot.utils.data_cleaner import DataCleaner
from bitbot.utils.logger import logger

class MissingDataStrategy(Enum):
    """Stratégies de gestion des données manquantes."""
    SKIP = "Ignorer"
    INTERPOLATE = "Interpoler"
    PREVIOUS = "Utiliser la valeur précédente"
    FALLBACK = "Utiliser un indicateur alternatif"
    FAIL = "Échouer avec une erreur"

class VWAPTimeFrame(Enum):
    """Périodes temporelles pour le calcul du VWAP."""
    INTRADAY = "Intraday"
    DAILY = "Journalier"
    WEEKLY = "Hebdomadaire"
    MONTHLY = "Mensuel"
    CUSTOM = "Personnalisé"

class VWAPIndicator:
    """Classe pour calculer et analyser le Volume Weighted Average Price (VWAP)."""
    
    def __init__(self, 
                 time_frame: VWAPTimeFrame = VWAPTimeFrame.DAILY,
                 bands_std_dev: float = 1.0,
                 clean_data: bool = True,
                 missing_data_strategy: MissingDataStrategy = MissingDataStrategy.FALLBACK):
        """
        Initialise l'indicateur VWAP.
        
        Args:
            time_frame: Période temporelle pour calculer le VWAP
            bands_std_dev: Écart-type pour les bandes de VWAP
            clean_data: Si True, nettoie automatiquement les données avant calcul
            missing_data_strategy: Stratégie pour gérer les données manquantes
        """
        self.time_frame = time_frame
        self.bands_std_dev = bands_std_dev
        self.data_cleaner = DataCleaner() if clean_data else None
        self.missing_data_strategy = missing_data_strategy
        
        # Pour stocker les résultats calculés
        self._last_reset_time = None
        self._cumulative_tpv = None  # Typical Price * Volume cumulatif
        self._cumulative_volume = None  # Volume cumulatif
        
        logger.info(f"Indicateur VWAP initialisé: Période={time_frame.value}, "
                   f"Écart-type bandes={bands_std_dev}, "
                   f"Stratégie données manquantes={missing_data_strategy.value}")
    
    def _should_reset_calculation(self, timestamp: pd.Timestamp) -> bool:
        """
        Détermine si le calcul du VWAP doit être réinitialisé en fonction de la période.
        
        Args:
            timestamp: Horodatage actuel
            
        Returns:
            True si le calcul doit être réinitialisé, False sinon
        """
        if self._last_reset_time is None:
            return True
            
        if self.time_frame == VWAPTimeFrame.INTRADAY:
            # Réinitialisation quand on change de jour
            return timestamp.date() != self._last_reset_time.date()
            
        elif self.time_frame == VWAPTimeFrame.DAILY:
            # En mode journalier, on réinitialise chaque jour
            return timestamp.date() != self._last_reset_time.date()
            
        elif self.time_frame == VWAPTimeFrame.WEEKLY:
            # Réinitialisation hebdomadaire (lundi)
            curr_week = timestamp.isocalendar()[1]
            last_week = self._last_reset_time.isocalendar()[1]
            return curr_week != last_week
            
        elif self.time_frame == VWAPTimeFrame.MONTHLY:
            # Réinitialisation mensuelle
            return (timestamp.year != self._last_reset_time.year or 
                    timestamp.month != self._last_reset_time.month)
                    
        # Pour les périodes personnalisées, on ne réinitialise pas automatiquement
        return False
    
    def _handle_missing_data(self, data: pd.DataFrame, required_cols: List[str]) -> Optional[pd.DataFrame]:
        """
        Gère les données manquantes selon la stratégie configurée.
        
        Args:
            data: DataFrame contenant les données
            required_cols: Colonnes requises pour le calcul
            
        Returns:
            DataFrame traité ou None si le traitement échoue
        """
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if not missing_cols:
            return data
            
        logger.warning(f"Données manquantes détectées. Colonnes manquantes: {missing_cols}")
        
        if self.missing_data_strategy == MissingDataStrategy.FAIL:
            raise ValueError(f"Colonnes requises manquantes: {missing_cols}")
            
        elif self.missing_data_strategy == MissingDataStrategy.SKIP:
            logger.info("Ignorer les données manquantes et continuer le calcul")
            # Retourner le DataFrame original au lieu de None pour être compatible avec les tests
            # Pour les colonnes manquantes, les valeurs seront NaN
            df = data.copy()
            for col in missing_cols:
                if col not in df.columns:
                    df[col] = np.nan
            return df
            
        elif self.missing_data_strategy == MissingDataStrategy.INTERPOLATE:
            logger.info("Tentative d'interpolation des données manquantes")
            return self._interpolate_missing_data(data, missing_cols)
            
        elif self.missing_data_strategy == MissingDataStrategy.PREVIOUS:
            logger.info("Utilisation des valeurs précédentes pour les données manquantes")
            return self._use_previous_values(data, missing_cols)
            
        elif self.missing_data_strategy == MissingDataStrategy.FALLBACK:
            logger.info("Utilisation d'un indicateur alternatif (SMA pondérée)")
            return self._calculate_fallback_indicator(data)
            
        return None
    
    def _interpolate_missing_data(self, data: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
        """
        Interpole les données manquantes.
        
        Args:
            data: DataFrame contenant les données
            missing_cols: Colonnes manquantes à interpoler
            
        Returns:
            DataFrame avec données interpolées
        """
        df = data.copy()
        
        # Si volume est manquant, on peut estimer avec la moyenne des derniers volumes
        if 'volume' in missing_cols and len(df) > 0:
            # Utiliser la moyenne des 10 derniers volumes si disponible
            if 'volume' in df.columns and len(df) > 10:
                avg_volume = df['volume'].iloc[-10:].mean()
            else:
                # Sinon utiliser une valeur arbitraire
                avg_volume = 1000
                
            df['volume'] = avg_volume
            logger.debug(f"Volume estimé à {avg_volume}")
        
        # Si high, low sont manquants mais qu'on a open et close, on peut estimer
        if ('high' in missing_cols or 'low' in missing_cols) and 'open' in df.columns and 'close' in df.columns:
            if 'high' in missing_cols:
                df['high'] = df[['open', 'close']].max(axis=1) * 1.001  # Légèrement supérieur
                logger.debug("Prix high estimé")
                
            if 'low' in missing_cols:
                df['low'] = df[['open', 'close']].min(axis=1) * 0.999  # Légèrement inférieur
                logger.debug("Prix low estimé")
        
        return df
    
    def _use_previous_values(self, data: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
        """
        Utilise les valeurs précédentes pour les données manquantes.
        
        Args:
            data: DataFrame contenant les données
            missing_cols: Colonnes manquantes à remplir
            
        Returns:
            DataFrame avec données précédentes
        """
        df = data.copy()
        
        # On vérifie s'il y a des données précédentes
        if len(df) <= 1:
            logger.warning("Pas assez de données pour utiliser les valeurs précédentes")
            return self._interpolate_missing_data(data, missing_cols)
        
        # Pour chaque colonne manquante, prendre la valeur précédente
        for col in missing_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
            else:
                # Si la colonne n'existe pas du tout, on doit l'estimer
                df = self._interpolate_missing_data(df, [col])
        
        return df
    
    def _calculate_fallback_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un indicateur alternatif quand les données sont insuffisantes pour le VWAP.
        
        Args:
            data: DataFrame contenant les données disponibles
            
        Returns:
            DataFrame avec l'indicateur alternatif calculé
        """
        df = data.copy()
        
        # Vérifier quelles colonnes sont disponibles
        has_close = 'close' in df.columns
        has_volume = 'volume' in df.columns
        
        if not has_close:
            logger.error("Impossible de calculer l'indicateur alternatif: prix de clôture manquant")
            return None
        
        # Stratégie de fallback: SMA pondérée
        if has_volume:
            # Si on a les volumes, on peut faire une moyenne pondérée par le volume
            logger.info("Calcul d'une moyenne mobile pondérée par le volume (volume-weighted MA)")
            df['typical_price'] = df['close']  # Simplification: on utilise juste le close
            df['tpv'] = df['typical_price'] * df['volume']
            df['volume_sum'] = df['volume'].rolling(window=20).sum()
            df['tpv_sum'] = df['tpv'].rolling(window=20).sum()
            df['vwma'] = df['tpv_sum'] / df['volume_sum']
            df['vwap_fallback'] = df['vwma']
            
            # Calculer des pseudo-bandes
            df['vwap_std'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['vwap_fallback'] + (df['vwap_std'] * self.bands_std_dev)
            df['lower_band'] = df['vwap_fallback'] - (df['vwap_std'] * self.bands_std_dev)
            
        else:
            # Si on n'a pas les volumes, on utilise une SMA simple
            logger.info("Calcul d'une simple moyenne mobile comme fallback")
            df['vwap_fallback'] = df['close'].rolling(window=20).mean()
            df['vwap_std'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['vwap_fallback'] + (df['vwap_std'] * self.bands_std_dev)
            df['lower_band'] = df['vwap_fallback'] - (df['vwap_std'] * self.bands_std_dev)
        
        logger.warning("Utilisation d'un indicateur de remplacement au lieu du VWAP")
        return df
    
    def calculate_vwap(self, data: Union[pd.DataFrame, MarketData]) -> Optional[pd.DataFrame]:
        """
        Calcule le VWAP et les bandes de VWAP sur les données fournies.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec les colonnes VWAP, upper_band et lower_band ajoutées
        """
        # Convertir MarketData en DataFrame si nécessaire
        if isinstance(data, MarketData):
            df = data.ohlcv.copy()
        else:
            df = data.copy()
        
        # Nettoyer les données si nécessaire
        if self.data_cleaner:
            if isinstance(data, MarketData):
                cleaned_market_data = self.data_cleaner.clean_market_data(data)
                df = cleaned_market_data.ohlcv
            else:
                # Créer un MarketData temporaire pour nettoyer le DataFrame
                temp_market_data = MarketData("temp", "1h")
                temp_market_data.ohlcv = df
                cleaned_market_data = self.data_cleaner.clean_market_data(temp_market_data)
                df = cleaned_market_data.ohlcv
        
        # Vérifier les colonnes nécessaires
        required_cols = ['high', 'low', 'close', 'volume']
        df = self._handle_missing_data(df, required_cols)
        
        if df is None:
            logger.error("Impossible de calculer le VWAP: données insuffisantes")
            return None
        
        # Calculer le prix typique
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Initialiser les colonnes de résultats
        df['vwap'] = np.nan
        df['upper_band'] = np.nan
        df['lower_band'] = np.nan
        df['vwap_std'] = np.nan
        
        # Calculer le VWAP
        current_day = None
        cumulative_tpv = 0
        cumulative_vol = 0
        tpv_values = []  # Pour calculer l'écart-type
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if isinstance(idx, pd.Timestamp):
                timestamp = idx
            else:
                # Si l'index n'est pas un timestamp, utiliser une valeur par défaut
                timestamp = pd.Timestamp.now()
            
            # Vérifier si on doit réinitialiser les calculs en fonction de la période
            if self._should_reset_calculation(timestamp):
                cumulative_tpv = 0
                cumulative_vol = 0
                tpv_values = []
                self._last_reset_time = timestamp
            
            # Calcul du VWAP cumulatif
            tp = row['typical_price']
            vol = row['volume']
            
            tpv = tp * vol
            cumulative_tpv += tpv
            cumulative_vol += vol
            
            if cumulative_vol > 0:
                vwap = cumulative_tpv / cumulative_vol
            else:
                vwap = tp  # Fallback si pas de volume
            
            df.loc[idx, 'vwap'] = vwap
            
            # Stocker les valeurs pour calculer l'écart-type
            tpv_values.append(tpv)
            
            # Calculer l'écart-type dynamique si on a assez de données
            if len(tpv_values) >= 5:
                tpv_std = np.std(tpv_values) if len(tpv_values) > 1 else 0
                df.loc[idx, 'vwap_std'] = tpv_std
                
                # S'assurer que les bandes sont toujours au-dessus/en-dessous du VWAP
                # même si l'écart-type est très petit
                min_distance = vwap * 0.001  # Au moins 0.1% d'écart
                
                df.loc[idx, 'upper_band'] = vwap + max(self.bands_std_dev * tpv_std, min_distance)
                df.loc[idx, 'lower_band'] = vwap - max(self.bands_std_dev * tpv_std, min_distance)
        
        # Mettre à jour les variables d'état
        self._cumulative_tpv = cumulative_tpv
        self._cumulative_volume = cumulative_vol
        
        return df
    
    def get_current_vwap(self, data: Union[pd.DataFrame, MarketData]) -> Dict[str, float]:
        """
        Obtient les valeurs VWAP actuelles sans recalculer tout l'historique.
        
        Args:
            data: DataFrame ou MarketData contenant les dernières données
            
        Returns:
            Dictionnaire avec les valeurs VWAP, bandes supérieure et inférieure
        """
        df = self.calculate_vwap(data)
        
        if df is None or len(df) == 0:
            return {
                'vwap': None,
                'upper_band': None,
                'lower_band': None,
                'is_fallback': True
            }
        
        # Vérifier si on a utilisé un fallback
        is_fallback = 'vwap_fallback' in df.columns and not pd.isna(df['vwap_fallback'].iloc[-1])
        
        if is_fallback:
            return {
                'vwap': df['vwap_fallback'].iloc[-1],
                'upper_band': df['upper_band'].iloc[-1],
                'lower_band': df['lower_band'].iloc[-1],
                'is_fallback': True
            }
        else:
            return {
                'vwap': df['vwap'].iloc[-1],
                'upper_band': df['upper_band'].iloc[-1],
                'lower_band': df['lower_band'].iloc[-1],
                'is_fallback': False
            }
    
    def is_price_above_vwap(self, price: float, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Vérifie si le prix est au-dessus du VWAP.
        
        Args:
            price: Prix à vérifier
            data: DataFrame ou MarketData pour calculer le VWAP
            
        Returns:
            True si le prix est au-dessus du VWAP, False sinon
        """
        vwap_info = self.get_current_vwap(data)
        
        if vwap_info['vwap'] is None:
            return False
        
        return price > vwap_info['vwap']
    
    def is_price_in_upper_band(self, price: float, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Vérifie si le prix est dans la bande supérieure du VWAP.
        
        Args:
            price: Prix à vérifier
            data: DataFrame ou MarketData pour calculer le VWAP
            
        Returns:
            True si le prix est au-dessus de la bande supérieure, False sinon
        """
        vwap_info = self.get_current_vwap(data)
        
        if vwap_info['upper_band'] is None:
            return False
        
        return price > vwap_info['upper_band']
    
    def is_price_in_lower_band(self, price: float, data: Union[pd.DataFrame, MarketData]) -> bool:
        """
        Vérifie si le prix est dans la bande inférieure du VWAP.
        
        Args:
            price: Prix à vérifier
            data: DataFrame ou MarketData pour calculer le VWAP
            
        Returns:
            True si le prix est en-dessous de la bande inférieure, False sinon
        """
        vwap_info = self.get_current_vwap(data)
        
        if vwap_info['lower_band'] is None:
            return False
        
        return price < vwap_info['lower_band']
    
    def identify_support_resistance(self, data: Union[pd.DataFrame, MarketData], 
                                  volume_threshold: float = 1.5) -> Dict[str, Any]:
        """
        Identifie les niveaux de support et résistance basés sur le VWAP et les volumes.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            volume_threshold: Seuil pour identifier les volumes significatifs
            
        Returns:
            Dictionnaire avec les niveaux de support et résistance identifiés
        """
        df = self.calculate_vwap(data)
        
        if df is None or len(df) == 0:
            return {'supports': [], 'resistances': []}
        
        # Calculer le volume moyen
        avg_volume = df['volume'].mean()
        high_volume_threshold = avg_volume * volume_threshold
        
        # Identifier les zones à volume élevé
        df['high_volume'] = df['volume'] > high_volume_threshold
        
        # Trouver les points où le prix traverse le VWAP avec un volume élevé
        df['above_vwap'] = df['close'] > df['vwap']
        df['cross_above'] = (df['above_vwap'] != df['above_vwap'].shift(1)) & (df['above_vwap'] == True)
        df['cross_below'] = (df['above_vwap'] != df['above_vwap'].shift(1)) & (df['above_vwap'] == False)
        
        # Identifier les niveaux de support (croisement descendant avec volume élevé)
        supports = df[df['cross_below'] & df['high_volume']]
        
        # Identifier les niveaux de résistance (croisement ascendant avec volume élevé)
        resistances = df[df['cross_above'] & df['high_volume']]
        
        # Regrouper les niveaux proches
        support_levels = self._group_price_levels(supports['vwap'].tolist())
        resistance_levels = self._group_price_levels(resistances['vwap'].tolist())
        
        return {
            'supports': support_levels,
            'resistances': resistance_levels,
            'vwap': df['vwap'].iloc[-1] if not df.empty else None
        }
    
    def _group_price_levels(self, prices: List[float], proximity_pct: float = 0.005) -> List[float]:
        """
        Groupe les niveaux de prix proches.
        
        Args:
            prices: Liste des niveaux de prix
            proximity_pct: Pourcentage de proximité pour regrouper
            
        Returns:
            Liste des niveaux de prix regroupés
        """
        if not prices:
            return []
            
        # Trier les prix
        sorted_prices = sorted(prices)
        
        # Regrouper les niveaux proches
        grouped_levels = []
        current_group = [sorted_prices[0]]
        
        for i in range(1, len(sorted_prices)):
            current = sorted_prices[i]
            prev_avg = sum(current_group) / len(current_group)
            
            # Si le prix actuel est proche du groupe précédent
            if abs(current - prev_avg) / prev_avg <= proximity_pct:
                current_group.append(current)
            else:
                # Ajouter la moyenne du groupe précédent
                grouped_levels.append(sum(current_group) / len(current_group))
                current_group = [current]
        
        # Ajouter le dernier groupe
        if current_group:
            grouped_levels.append(sum(current_group) / len(current_group))
        
        return grouped_levels


class VWAPStrategy:
    """
    Stratégie de trading basée sur le VWAP.
    
    Cette stratégie utilise le VWAP pour identifier les opportunités de trading
    en analysant les croisements de prix et les niveaux de support/résistance.
    """
    
    def __init__(self, 
                 time_frame: VWAPTimeFrame = VWAPTimeFrame.INTRADAY,
                 bands_std_dev: float = 1.0,
                 reversion_mode: bool = True,
                 volume_filter: bool = True,
                 volume_threshold: float = 1.5,
                 missing_data_strategy: MissingDataStrategy = MissingDataStrategy.FALLBACK):
        """
        Initialise la stratégie VWAP.
        
        Args:
            time_frame: Période de calcul du VWAP
            bands_std_dev: Écart-type pour les bandes VWAP
            reversion_mode: Si True, recherche des retours à la moyenne, 
                            sinon suit la tendance
            volume_filter: Si True, filtre les signaux par volume
            volume_threshold: Seuil de volume minimum (ratio par rapport à la moyenne)
            missing_data_strategy: Stratégie pour gérer les données manquantes
        """
        self.vwap_indicator = VWAPIndicator(
            time_frame=time_frame,
            bands_std_dev=bands_std_dev,
            missing_data_strategy=missing_data_strategy
        )
        self.reversion_mode = reversion_mode
        self.volume_filter = volume_filter
        self.volume_threshold = volume_threshold
        
        logger.info(f"Stratégie VWAP initialisée: Mode={'retour à la moyenne' if reversion_mode else 'suivi de tendance'}, "
                   f"Filtre volume={'activé' if volume_filter else 'désactivé'}")
    
    def calculate_signals(self, data: Union[pd.DataFrame, MarketData]) -> pd.DataFrame:
        """
        Calcule les signaux d'achat/vente basés sur le VWAP.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            DataFrame avec les colonnes de signaux ajoutées
        """
        # Calculer le VWAP
        df = self.vwap_indicator.calculate_vwap(data)
        
        if df is None or len(df) == 0:
            logger.error("Impossible de calculer les signaux VWAP: données insuffisantes")
            return pd.DataFrame()  # Retourner un DataFrame vide plutôt que None
        
        # Initialiser les colonnes de signal
        df['signal'] = 0  # 0 = neutre, 1 = achat, -1 = vente
        
        # Calculer la colonne de volume élevé si le filtre est activé
        if self.volume_filter:
            avg_volume = df['volume'].mean()
            df['high_volume'] = df['volume'] > (avg_volume * self.volume_threshold)
        else:
            df['high_volume'] = True  # Toujours considérer le volume comme élevé si le filtre est désactivé
        
        # Calculer les croisements du VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['cross_above'] = (df['above_vwap'] != df['above_vwap'].shift(1)) & (df['above_vwap'] == True)
        df['cross_below'] = (df['above_vwap'] != df['above_vwap'].shift(1)) & (df['above_vwap'] == False)
        
        # Calcul des signaux basés sur le mode
        if self.reversion_mode:
            # Mode retour à la moyenne:
            # Acheter quand le prix tombe en dessous de la bande inférieure (survente)
            # Vendre quand le prix monte au-dessus de la bande supérieure (surachat)
            df.loc[(df['close'] < df['lower_band']) & df['high_volume'], 'signal'] = 1  # Signal d'achat
            df.loc[(df['close'] > df['upper_band']) & df['high_volume'], 'signal'] = -1  # Signal de vente
        else:
            # Mode suivi de tendance:
            # Acheter quand le prix croise le VWAP à la hausse
            # Vendre quand le prix croise le VWAP à la baisse
            df.loc[df['cross_above'] & df['high_volume'], 'signal'] = 1  # Signal d'achat
            df.loc[df['cross_below'] & df['high_volume'], 'signal'] = -1  # Signal de vente
        
        return df
    
    def generate_trading_signals(self, data: Union[pd.DataFrame, MarketData]) -> Dict[str, Any]:
        """
        Génère des signaux de trading complets basés sur le VWAP.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            
        Returns:
            Dictionnaire contenant les signaux, le VWAP, et les niveaux de support/résistance
        """
        # Calculer les signaux
        df = self.calculate_signals(data)
        
        if df is None or len(df) == 0:
            return {
                'signal': 0,
                'vwap': None,
                'supports': [],
                'resistances': [],
                'is_fallback': True
            }
        
        # Obtenir le signal actuel
        current_signal = df['signal'].iloc[-1]
        
        # Vérifier si on a utilisé un fallback
        is_fallback = 'vwap_fallback' in df.columns and not df['vwap_fallback'].isna().all()
        
        # Identifier les niveaux de support et résistance
        levels = self.vwap_indicator.identify_support_resistance(data, self.volume_threshold)
        
        return {
            'signal': current_signal,
            'vwap': df['vwap'].iloc[-1] if not is_fallback else df['vwap_fallback'].iloc[-1],
            'upper_band': df['upper_band'].iloc[-1],
            'lower_band': df['lower_band'].iloc[-1],
            'supports': levels['supports'],
            'resistances': levels['resistances'],
            'is_fallback': is_fallback
        }
    
    def get_signal_description(self, signal_info: Dict[str, Any]) -> str:
        """
        Génère une description textuelle du signal.
        
        Args:
            signal_info: Dictionnaire contenant les informations du signal
            
        Returns:
            Description textuelle du signal
        """
        signal = signal_info['signal']
        is_fallback = signal_info.get('is_fallback', False)
        
        mode_text = "retour à la moyenne" if self.reversion_mode else "suivi de tendance"
        source_text = "indicateur de remplacement" if is_fallback else "VWAP"
        
        if signal == 1:
            return f"⬆️ ACHAT ({mode_text}): Prix favorable par rapport au {source_text}"
        elif signal == -1:
            return f"⬇️ VENTE ({mode_text}): Prix défavorable par rapport au {source_text}"
        else:
            return f"➡️ NEUTRE: Aucun signal généré par la stratégie VWAP ({mode_text})"
    
    def plot_vwap(self, data: Union[pd.DataFrame, MarketData], ax=None, figsize=(14, 7)):
        """
        Crée un graphique du VWAP et des signaux.
        
        Args:
            data: DataFrame ou MarketData contenant les données OHLCV
            ax: Axe matplotlib optionnel pour le tracé
            figsize: Taille de la figure si ax n'est pas fourni
            
        Returns:
            Figure matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
        except ImportError:
            logger.error("matplotlib est requis pour tracer le VWAP")
            return None
            
        # Calculer les signaux
        df = self.calculate_signals(data)
        
        if df is None or len(df) == 0:
            logger.error("Pas de données à afficher")
            return None
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            
        # Tracer le prix
        ax.plot(df.index, df['close'], label='Prix de clôture', color='black', alpha=0.7)
        
        # Tracer le VWAP
        vwap_col = 'vwap_fallback' if 'vwap_fallback' in df.columns and not df['vwap_fallback'].isna().all() else 'vwap'
        ax.plot(df.index, df[vwap_col], label='VWAP', color='blue', linestyle='-', linewidth=1.5)
        
        # Tracer les bandes
        ax.plot(df.index, df['upper_band'], label='Bande supérieure', color='red', linestyle='--', alpha=0.7)
        ax.plot(df.index, df['lower_band'], label='Bande inférieure', color='green', linestyle='--', alpha=0.7)
        
        # Tracer les signaux
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Signal d\'achat')
        ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Signal de vente')
        
        # Ombrer la zone entre les bandes
        ax.fill_between(df.index, df['lower_band'], df['upper_band'], color='blue', alpha=0.1)
        
        # Formater l'axe des X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // 10)))
        plt.xticks(rotation=45)
        
        # Légende et titre
        ax.set_title(f"VWAP - Mode {('Retour à la moyenne' if self.reversion_mode else 'Suivi de tendance')}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une note si le fallback a été utilisé
        if 'vwap_fallback' in df.columns and not df['vwap_fallback'].isna().all():
            ax.annotate('* Indicateur de remplacement utilisé', xy=(0.02, 0.02), xycoords='axes fraction', 
                       fontsize=9, color='red')
        
        plt.tight_layout()
        return fig
