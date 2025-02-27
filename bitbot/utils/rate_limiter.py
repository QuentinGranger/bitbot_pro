"""
Rate limiter pour gérer les limites d'API.
"""

import time
import asyncio
from typing import Dict, Optional
from collections import deque

class RateLimiter:
    """
    Rate limiter avec fenêtre glissante.
    Permet de contrôler le nombre de requêtes par période.
    """
    
    def __init__(self, rate: int, per: float):
        """
        Args:
            rate: Nombre maximum de requêtes
            per: Période en secondes
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Tente d'acquérir une autorisation.
        
        Returns:
            True si autorisé, False sinon
        """
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            # Ajouter les jetons accumulés
            self.allowance += time_passed * (self.rate / self.per)
            
            # Plafonner à rate
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            # Vérifier si on peut consommer
            if self.allowance < 1:
                return False
            
            self.allowance -= 1
            return True
    
    async def wait(self):
        """Attend jusqu'à ce qu'une autorisation soit disponible."""
        while not await self.acquire():
            await asyncio.sleep(self.per / self.rate)

class MultiRateLimiter:
    """
    Rate limiter pour plusieurs ressources avec différentes limites.
    Utile pour gérer plusieurs endpoints d'API avec des limites différentes.
    """
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.lock = asyncio.Lock()
    
    def add_limiter(self, name: str, rate: int, per: float):
        """
        Ajoute un nouveau rate limiter.
        
        Args:
            name: Nom du limiter
            rate: Nombre maximum de requêtes
            per: Période en secondes
        """
        self.limiters[name] = RateLimiter(rate, per)
    
    async def acquire(self, name: str) -> bool:
        """
        Tente d'acquérir une autorisation pour un limiter spécifique.
        
        Args:
            name: Nom du limiter
        
        Returns:
            True si autorisé, False sinon
        """
        if name not in self.limiters:
            raise KeyError(f"Rate limiter '{name}' non trouvé")
        
        return await self.limiters[name].acquire()
    
    async def wait(self, name: str):
        """
        Attend jusqu'à ce qu'une autorisation soit disponible.
        
        Args:
            name: Nom du limiter
        """
        if name not in self.limiters:
            raise KeyError(f"Rate limiter '{name}' non trouvé")
        
        await self.limiters[name].wait()

class BurstRateLimiter:
    """
    Rate limiter avec support des bursts.
    Permet des pics d'activité tout en maintenant une moyenne.
    """
    
    def __init__(self, rate: int, per: float, burst: int):
        """
        Args:
            rate: Nombre maximum de requêtes
            per: Période en secondes
            burst: Taille maximale du burst
        """
        self.rate = rate
        self.per = per
        self.burst = burst
        self.tokens = burst
        self.last_check = time.time()
        self.lock = asyncio.Lock()
        
        # File pour suivre les requêtes
        self.requests = deque(maxlen=burst)
    
    async def acquire(self) -> bool:
        """
        Tente d'acquérir une autorisation avec support des bursts.
        
        Returns:
            True si autorisé, False sinon
        """
        async with self.lock:
            current = time.time()
            
            # Nettoyer les anciennes requêtes
            while self.requests and current - self.requests[0] > self.per:
                self.requests.popleft()
            
            # Vérifier si on peut faire un burst
            if len(self.requests) >= self.burst:
                return False
            
            # Vérifier le taux moyen
            if len(self.requests) >= self.rate:
                oldest = self.requests[0]
                if current - oldest < self.per:
                    return False
            
            self.requests.append(current)
            return True
    
    async def wait(self):
        """Attend jusqu'à ce qu'une autorisation soit disponible."""
        while not await self.acquire():
            await asyncio.sleep(self.per / self.rate)
