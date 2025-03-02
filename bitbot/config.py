"""
Configuration pour BitBot Pro.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration pour BitBot Pro."""
    
    # API Keys
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    coingecko_api_key: Optional[str] = None
    coinmarketcap_api_key: Optional[str] = None
    google_trends_api_key: Optional[str] = None
    cryptopanic_api_key: Optional[str] = None
    
    # Répertoires
    data_dir: Path = Path("data")
    
    # Options SSL
    verify_ssl: bool = True
    
    # Options de cache
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 heure
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_max: int = 10  # requêtes par seconde
    
    # Options de stratégie
    strategy_name: str = "dummy"
    strategy_params: dict = None
    
    # Options de trading
    signal_threshold: float = 0.5
    order_quantity: float = 0.01
    leverage: float = 1.0
    
    # Configuration du mode Safe (protection automatique)
    safe_mode_enabled: bool = True  # Mode Safe activé par défaut
    safe_mode_max_drawdown_threshold: float = 15.0  # Seuil de drawdown critique en pourcentage
    safe_mode_max_volatility_threshold: float = 5.0  # Seuil de volatilité critique en pourcentage
    safe_mode_max_data_age_minutes: int = 10  # Âge maximal des données en minutes avant d'activer le mode Safe
    safe_mode_auto_deactivate_after_hours: int = 0  # 0 = désactivation manuelle uniquement, sinon nombre d'heures
    safe_mode_notify_admin: bool = True  # Envoyer une notification à l'administrateur
    
    def __post_init__(self):
        """Initialise les valeurs par défaut après la création de l'objet."""
        # S'assurer que data_dir est un Path
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
            
        # Créer le répertoire de données s'il n'existe pas
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser strategy_params si None
        if self.strategy_params is None:
            self.strategy_params = {}
