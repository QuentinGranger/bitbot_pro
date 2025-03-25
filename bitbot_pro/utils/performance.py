"""
Utilitaires de mesure de performance pour BitBot Pro.
Fournit des décorateurs et fonctions pour mesurer le temps d'exécution 
et optimiser les performances des traitements.
"""
import time
import logging
import functools
from datetime import datetime
from typing import Callable, Any, Optional

from bitbot_pro.utils.logger import logger


def timeit(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func: La fonction à décorer
        
    Returns:
        La fonction décorée avec mesure de performance
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Journalisation simplifiée compatible avec le logger de BitBot Pro
        try:
            logger.debug(f"Exécution de {func.__name__}: {elapsed:.4f} secondes")
        except Exception:
            # En cas d'erreur avec le logger, continuer silencieusement
            pass
        
        return result
    return wrapper


def measure_execution_time(func: Callable) -> Callable:
    """
    Décorateur pour mesurer et afficher le temps d'exécution d'une fonction.
    Plus verbeux que timeit, utile pour les benchmarks.
    
    Args:
        func: La fonction à décorer
        
    Returns:
        La fonction décorée avec mesure de performance détaillée
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_datetime = datetime.now()
        
        logger.info(f"Démarrage de {func.__name__} à {start_datetime.strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"Exécution de {func.__name__} terminée en {elapsed:.4f} secondes")
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Erreur dans {func.__name__} après {elapsed:.4f} secondes: {str(e)}")
            raise
    
    return wrapper


class PerformanceTracker:
    """
    Classe pour suivre les performances des opérations critiques.
    Permet d'analyser les performances globales du système.
    """
    
    def __init__(self, name: str):
        """
        Initialise un tracker de performance.
        
        Args:
            name: Nom du composant à suivre
        """
        self.name = name
        self.execution_times = []
        self.start_time = None
    
    def start(self) -> None:
        """Démarre le chronomètre"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Arrête le chronomètre et enregistre le temps écoulé.
        
        Returns:
            Le temps écoulé en secondes
        """
        if self.start_time is None:
            logger.warning(f"PerformanceTracker {self.name}: stop() appelé sans start() préalable")
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.execution_times.append(elapsed)
        self.start_time = None
        
        return elapsed
    
    def get_stats(self) -> dict:
        """
        Calcule les statistiques de performance.
        
        Returns:
            Dictionnaire contenant les statistiques (min, max, avg, count)
        """
        if not self.execution_times:
            return {
                'min': 0.0,
                'max': 0.0,
                'avg': 0.0,
                'count': 0
            }
        
        return {
            'min': min(self.execution_times),
            'max': max(self.execution_times),
            'avg': sum(self.execution_times) / len(self.execution_times),
            'count': len(self.execution_times)
        }
    
    def reset(self) -> None:
        """Réinitialise les statistiques"""
        self.execution_times = []
        self.start_time = None
    
    def __enter__(self):
        """Support du context manager (with statement)"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager (with statement)"""
        self.stop()
        return False  # Propager les exceptions
