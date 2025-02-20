import pandas as pd
import numpy as np
import logging
from risk_management.risk_manager import RiskManager
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_normal_conditions():
    """Test en conditions normales"""
    risk_manager = RiskManager(
        max_position_risk=0.05,     # 5% max par position
        max_drawdown=0.25,          # 25% drawdown max
        volatility_lookback=30,     # 30 périodes
        min_risk_reward_ratio=3.0,  # Ratio R/R minimum de 1:3
        max_capital_per_trade=0.01, # 1% du capital max par trade
        drawdown_thresholds={      # Seuils de drawdown personnalisés
            0.05: 0.20,  # -5% → réduction de 20%
            0.10: 0.50,  # -10% → réduction de 50%
            0.15: 0.75,  # -15% → réduction de 75%
            0.20: 1.00   # -20% → arrêt total
        },
        ai_enabled=True
    )
    
    # Capital initial
    capital = 100000  # $100,000
    
    # Récupérer les données BTC
    btc_data = risk_manager.get_current_btc_data(interval='1h', limit=100)
    if btc_data is None:
        logger.error("Impossible de récupérer les données BTC")
        return
        
    # Prix actuel
    current_price = btc_data['close'].iloc[-1]
    
    # Simuler une position longue
    base_size = 0.5  # 0.5 BTC
    entry_price = current_price
    stop_loss = entry_price * 0.95  # -5%
    
    # Calculer le take profit optimal (ratio 1:3)
    take_profit = risk_manager.calculate_optimal_take_profit(
        entry_price, stop_loss
    )
    
    # Valider le setup
    is_valid, reason = risk_manager.validate_trade_setup(
        entry_price, stop_loss, take_profit
    )
    
    logger.info("\nValidation du setup:")
    logger.info(f"Setup valide: {is_valid}")
    logger.info(f"Raison: {reason}")
    
    if not is_valid:
        logger.warning("Le setup ne respecte pas nos critères de risque")
        return
        
    # Calculer la taille maximale de position
    max_size = risk_manager.calculate_max_position_size(
        entry_price, stop_loss, capital
    )
    
    logger.info("\nAnalyse du position sizing:")
    logger.info(f"Taille maximale autorisée: {max_size:.4f} BTC")
    logger.info(f"Valeur maximale: ${(max_size * entry_price):,.2f}")
    logger.info(f"Capital à risquer: ${(capital * risk_manager.max_capital_per_trade):,.2f}")
    
    # Ajuster la taille de position
    adjusted_size, metrics = risk_manager.adjust_position_size(
        base_size,
        entry_price,
        stop_loss,
        take_profit,
        capital
    )
    
    # Mettre à jour la position
    risk_manager.update_position(
        adjusted_size,
        entry_price,
        current_price,
        stop_loss,
        take_profit,
        capital
    )
    
    # Afficher les résultats
    logger.info("\nAnalyse du risque pour BTC:")
    logger.info(f"Prix d'entrée: ${entry_price:,.2f}")
    logger.info(f"Stop Loss: ${stop_loss:,.2f}")
    logger.info(f"Take Profit: ${take_profit:,.2f}")
    logger.info(f"Taille initiale: {base_size:.4f} BTC")
    logger.info(f"Taille ajustée: {adjusted_size:.4f} BTC")
    logger.info(f"Risque initial: {metrics['initial_risk']:.2%}")
    logger.info(f"Risque ajusté: {metrics['adjusted_risk']:.2%}")
    logger.info(f"Ratio Risque/Rendement: {metrics['risk_reward_ratio']:.2f}")
    logger.info(f"Volatilité: {metrics['volatility']:.2%}")
    logger.info(f"Valeur de la position: ${metrics['position_value']:,.2f}")
    logger.info(f"Capital à risque: {metrics['capital_at_risk']:.2%}")
    logger.info(f"Drawdown actuel: {metrics['drawdown']:.2%}")
    logger.info(f"Niveau de risque: {metrics['risk_level']}")
    
    if metrics.get('drawdown_message'):
        logger.warning(metrics['drawdown_message'])
    
    if metrics['ai_analysis']:
        logger.info("\nAnalyse AI pour BTC:")
        logger.info(metrics['ai_analysis']['analysis'])
    
    # Obtenir les métriques de la position
    position_metrics = risk_manager.get_position_metrics()
    
    logger.info("\nMétriques de la position:")
    logger.info(f"Valeur: ${position_metrics['position_value']:,.2f}")
    logger.info(f"P&L: ${position_metrics['pnl']:,.2f}")
    logger.info(f"Risque: {position_metrics['risk']:.2%}")
    logger.info(f"Drawdown: {position_metrics['drawdown']:.2%}")
    
    # Vérifier si on doit fermer la position
    should_close, reason = risk_manager.should_close_position(current_price)
    
    logger.info("\nAnalyse de clôture:")
    logger.info(f"Fermer la position: {should_close}")
    logger.info(f"Raison: {reason}")

def test_drawdown_levels():
    """Test des différents niveaux de drawdown"""
    risk_manager = RiskManager(
        max_position_risk=0.05,
        max_drawdown=0.25,
        volatility_lookback=30,
        min_risk_reward_ratio=3.0,
        max_capital_per_trade=0.01,
        drawdown_thresholds={
            0.05: 0.20,  # -5% → réduction de 20%
            0.10: 0.50,  # -10% → réduction de 50%
            0.15: 0.75,  # -15% → réduction de 75%
            0.20: 1.00   # -20% → arrêt total
        },
        ai_enabled=True
    )
    
    # Test de différents niveaux de drawdown
    drawdown_levels = [0.03, 0.07, 0.12, 0.17, 0.22]
    initial_capital = 100000
    
    logger.info("\nTest des niveaux de drawdown:")
    for dd in drawdown_levels:
        current_capital = initial_capital * (1 - dd)
        multiplier, message, risk_level = risk_manager.get_drawdown_status(dd)
        
        logger.info(f"\nTest avec drawdown de {dd:.1%}:")
        logger.info(f"Capital initial: ${initial_capital:,.2f}")
        logger.info(f"Capital actuel: ${current_capital:,.2f}")
        logger.info(f"Multiplicateur: {multiplier:.2f}")
        logger.info(f"Niveau de risque: {risk_level}")
        logger.warning(message)
        
    # Test des pertes consécutives
    logger.info("\nTest des pertes consécutives:")
    for i in range(4):
        risk_manager.update_consecutive_losses(-1000)  # Simuler une perte
        logger.info(f"Perte #{i+1}")
        multiplier = risk_manager.get_drawdown_multiplier(current_capital)
        logger.info(f"Multiplicateur après {i+1} pertes: {multiplier:.2f}")

def main():
    """Test du gestionnaire de risque"""
    try:
        # Test en conditions normales
        test_normal_conditions()
        
        # Test des différents niveaux de drawdown
        test_drawdown_levels()
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {str(e)}")

if __name__ == "__main__":
    main()
