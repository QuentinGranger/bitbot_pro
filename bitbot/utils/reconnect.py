"""
Gestionnaire de reconnexion avec backoff exponentiel pour les connexions API et WebSocket.
"""

import asyncio
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from bitbot.utils.logger import logger

# Types génériques pour les décorateurs
P = ParamSpec('P')
T = TypeVar('T')

def with_exponential_backoff(
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 60,
    exceptions: tuple = (ConnectionError, TimeoutError)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Décorateur pour réessayer une fonction avec backoff exponentiel.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        min_wait: Temps d'attente minimum en secondes
        max_wait: Temps d'attente maximum en secondes
        exceptions: Tuple d'exceptions à intercepter
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, log_level="WARNING"),
        after=after_log(logger, log_level="INFO"),
        reraise=True
    )

class WebSocketReconnector:
    """Gestionnaire de reconnexion WebSocket avec backoff exponentiel."""
    
    def __init__(
        self,
        min_delay: float = 1.0,
        max_delay: float = 60.0,
        factor: float = 2.0,
        jitter: float = 0.1
    ):
        """
        Args:
            min_delay: Délai minimum entre les tentatives
            max_delay: Délai maximum entre les tentatives
            factor: Facteur multiplicatif pour le backoff
            jitter: Facteur de variation aléatoire (0-1)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self.current_delay = min_delay
        self.attempt = 0
    
    async def wait(self) -> float:
        """
        Attend le délai approprié avant la prochaine tentative.
        
        Returns:
            Délai effectif d'attente en secondes
        """
        self.attempt += 1
        
        # Calcul du délai avec backoff exponentiel
        delay = min(
            self.max_delay,
            self.min_delay * (self.factor ** (self.attempt - 1))
        )
        
        # Ajout d'un jitter pour éviter les reconnexions synchronisées
        jitter_range = delay * self.jitter
        effective_delay = delay + random.uniform(-jitter_range, jitter_range)
        
        logger.warning(f"Tentative de reconnexion dans {effective_delay:.2f} secondes (tentative {self.attempt})")
        await asyncio.sleep(effective_delay)
        
        return effective_delay
    
    def reset(self):
        """Réinitialise le compteur de tentatives."""
        self.attempt = 0
        self.current_delay = self.min_delay

class ConnectionManager:
    """
    Gestionnaire de connexion avec surveillance de l'état et reconnexion automatique.
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Nom de la connexion pour les logs
        """
        self.name = name
        self.is_connected = False
        self.reconnector = WebSocketReconnector()
        self._connection_lock = asyncio.Lock()
    
    async def connect(self, connect_func: Callable, *args, **kwargs) -> bool:
        """
        Établit une connexion avec gestion des erreurs et reconnexion.
        
        Args:
            connect_func: Fonction de connexion à appeler
            *args, **kwargs: Arguments pour la fonction de connexion
        
        Returns:
            True si la connexion est établie, False sinon
        """
        async with self._connection_lock:
            try:
                if not self.is_connected:
                    await connect_func(*args, **kwargs)
                    self.is_connected = True
                    self.reconnector.reset()
                    logger.info(f"Connexion établie : {self.name}")
                return True
            
            except Exception as e:
                logger.error(f"Erreur de connexion ({self.name}): {str(e)}")
                self.is_connected = False
                return False
    
    async def disconnect(self, disconnect_func: Callable, *args, **kwargs):
        """
        Ferme proprement la connexion.
        
        Args:
            disconnect_func: Fonction de déconnexion à appeler
            *args, **kwargs: Arguments pour la fonction de déconnexion
        """
        async with self._connection_lock:
            if self.is_connected:
                try:
                    await disconnect_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Erreur lors de la déconnexion ({self.name}): {str(e)}")
                finally:
                    self.is_connected = False
                    logger.info(f"Déconnexion : {self.name}")
    
    async def ensure_connected(self, connect_func: Callable, *args, **kwargs):
        """
        S'assure que la connexion est établie, avec tentatives de reconnexion.
        
        Args:
            connect_func: Fonction de connexion à appeler
            *args, **kwargs: Arguments pour la fonction de connexion
        """
        while not self.is_connected:
            if not await self.connect(connect_func, *args, **kwargs):
                await self.reconnector.wait()
            
# Exemple d'utilisation
if __name__ == "__main__":
    from bitbot.utils.logger import setup_logger
    
    async def example():
        setup_logger()
        
        # Exemple avec le décorateur
        @with_exponential_backoff(max_attempts=3)
        async def fetch_data():
            # Simuler une erreur de connexion
            raise ConnectionError("Erreur de connexion simulée")
        
        # Exemple avec le gestionnaire de connexion
        async def mock_connect():
            # Simuler une connexion
            await asyncio.sleep(1)
            
        manager = ConnectionManager("ExchangeAPI")
        
        try:
            # Tester le décorateur
            await fetch_data()
        except Exception as e:
            logger.error(f"Échec final : {str(e)}")
        
        # Tester le gestionnaire de connexion
        await manager.ensure_connected(mock_connect)
    
    asyncio.run(example())
