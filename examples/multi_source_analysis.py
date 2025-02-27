"""
Exemple d'analyse combinant données de marché et données on-chain.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bitbot.data.market_data import MarketDataConfig, CoinGeckoClient, CoinMarketCapClient
from bitbot.data.onchain import GlassnodeClient
from bitbot.utils.logger import logger

async def analyze_bitcoin_market():
    """Analyse complète du marché Bitcoin."""
    
    # Configuration
    config = MarketDataConfig(
        coingecko_api_key="YOUR_COINGECKO_KEY",  # Optionnel
        coinmarketcap_api_key="YOUR_CMC_KEY",
        glassnode_api_key="YOUR_GLASSNODE_KEY"
    )
    
    # Initialiser les clients
    coingecko = CoinGeckoClient(config)
    cmc = CoinMarketCapClient(config)
    glassnode = GlassnodeClient(config)
    
    try:
        # Période d'analyse
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # 1. Données de prix (CoinGecko)
        logger.info("Récupération des données de prix...")
        price_data = await coingecko.get_price_history(
            coin_id="bitcoin",
            days=90,
            interval="daily"
        )
        
        # 2. Données de marché (CoinMarketCap)
        logger.info("Récupération des données de marché...")
        market_data = await cmc.get_latest_quotes("BTC")
        
        # 3. Données on-chain (Glassnode)
        logger.info("Récupération des données on-chain...")
        mvrv_data = await glassnode.get_mvrv(
            asset="BTC",
            since=start_date,
            until=end_date
        )
        
        netflow_data = await glassnode.get_exchange_netflow(
            asset="BTC",
            since=start_date,
            until=end_date
        )
        
        sopr_data = await glassnode.get_sopr(
            asset="BTC",
            since=start_date,
            until=end_date
        )
        
        # Analyse combinée
        logger.info("Analyse des données...")
        
        # Créer le graphique
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Prix BTC/USD',
                'MVRV Ratio',
                'Exchange Net Flow',
                'SOPR'
            )
        )
        
        # Prix
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='BTC/USD'
            ),
            row=1, col=1
        )
        
        # MVRV
        fig.add_trace(
            go.Scatter(
                x=mvrv_data.index,
                y=mvrv_data['v'],
                name='MVRV',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # Exchange Net Flow
        fig.add_trace(
            go.Bar(
                x=netflow_data.index,
                y=netflow_data['v'],
                name='Net Flow',
                marker_color='green'
            ),
            row=3, col=1
        )
        
        # SOPR
        fig.add_trace(
            go.Scatter(
                x=sopr_data.index,
                y=sopr_data['v'],
                name='SOPR',
                line=dict(color='purple')
            ),
            row=4, col=1
        )
        
        # Mise en forme
        fig.update_layout(
            title='Analyse Bitcoin - Marché et On-Chain',
            height=1200,
            showlegend=True
        )
        
        # Sauvegarder le graphique
        fig.write_html("bitcoin_analysis.html")
        
        # Analyse des signaux
        logger.info("\nAnalyse des signaux :")
        
        # 1. Tendance de prix
        price_change = (
            price_data['close'][-1] - price_data['close'][0]
        ) / price_data['close'][0] * 100
        
        logger.info(f"Variation de prix sur 90 jours : {price_change:.2f}%")
        
        # 2. Analyse MVRV
        current_mvrv = mvrv_data['v'].iloc[-1]
        logger.info(f"MVRV actuel : {current_mvrv:.2f}")
        if current_mvrv > 3.0:
            logger.warning("MVRV élevé : possible surévaluation")
        elif current_mvrv < 1.0:
            logger.warning("MVRV bas : possible sous-évaluation")
        
        # 3. Analyse des flux
        recent_netflow = netflow_data['v'].tail(7).sum()
        logger.info(
            f"Flux net des exchanges (7j) : {recent_netflow:.2f} BTC"
            f"({'sortie' if recent_netflow < 0 else 'entrée'})"
        )
        
        # 4. Analyse SOPR
        current_sopr = sopr_data['v'].iloc[-1]
        logger.info(f"SOPR actuel : {current_sopr:.2f}")
        if current_sopr < 1:
            logger.warning("SOPR < 1 : les vendeurs sont en perte en moyenne")
        
        logger.info("\nAnalyse sauvegardée dans bitcoin_analysis.html")
        
    finally:
        # Fermer les connexions
        await coingecko.close()
        await cmc.close()
        await glassnode.close()

if __name__ == "__main__":
    asyncio.run(analyze_bitcoin_market())
