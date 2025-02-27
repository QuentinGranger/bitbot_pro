import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from app.core.signal_aggregator import SignalAggregator
from app.core.risk_manager import RiskManager
from app.exchanges.exchange_manager import ExchangeManager
from app.api.routes import router as api_router
from app.config.settings import Settings
from app.utils.security import setup_security

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'application FastAPI
app = FastAPI(title="BitBot Pro", version="1.0.0")

# Configuration des composants principaux
settings = Settings()
exchange_manager = ExchangeManager()
signal_aggregator = SignalAggregator()
risk_manager = RiskManager()

# Configuration des routes API
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialisation des composants au démarrage"""
    try:
        # Configuration de la sécurité
        setup_security()
        
        # Initialisation des connexions aux exchanges
        await exchange_manager.initialize()
        
        # Démarrage du signal aggregator
        await signal_aggregator.start()
        
        logger.info("BitBot Pro démarré avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage des ressources à l'arrêt"""
    try:
        await exchange_manager.cleanup()
        await signal_aggregator.stop()
        logger.info("BitBot Pro arrêté avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
