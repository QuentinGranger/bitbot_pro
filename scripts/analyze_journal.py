#!/usr/bin/env python
"""
Script d'analyse post-mortem des journaux de trading.

Ce script permet d'analyser en détail les journaux créés par le bot 
de trading pour identifier les causes des problèmes et comprendre
les décisions prises par le bot.

Exemple d'utilisation:
    python scripts/analyze_journal.py --session 20250227_123045 --symbol BTCUSDT
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import logging
import re

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitbot.utils.journal import TradingJournal
from bitbot.utils.visualization import plot_candlestick_with_signals

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Analyseur post-mortem des journaux de trading")
    
    parser.add_argument("--session", type=str, help="ID de la session à analyser (format: YYYYMMDD_HHMMSS)")
    parser.add_argument("--symbol", type=str, help="Symbole à analyser (ex: BTCUSDT)")
    parser.add_argument("--decision-type", type=str, help="Type de décision à analyser")
    parser.add_argument("--start-date", type=str, help="Date de début (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Date de fin (format: YYYY-MM-DD)")
    parser.add_argument("--error-analysis", action="store_true", help="Analyser les erreurs")
    parser.add_argument("--correlation-analysis", action="store_true", help="Analyser les corrélations")
    parser.add_argument("--timeline", action="store_true", help="Générer une chronologie des décisions")
    parser.add_argument("--output-dir", type=str, default="analysis_reports", help="Répertoire de sortie pour les rapports")
    parser.add_argument("--latest", action="store_true", help="Analyser la session la plus récente")
    parser.add_argument("--plot", action="store_true", help="Générer des graphiques")
    
    return parser.parse_args()

def find_latest_session(journal_dir="journals"):
    """Trouve la session la plus récente en se basant sur les noms de fichiers."""
    journal_dir = Path(journal_dir)
    decisions_dir = journal_dir / "decisions"
    
    if not decisions_dir.exists():
        logger.error(f"Répertoire de décisions non trouvé: {decisions_dir}")
        return None
    
    # Trouver tous les fichiers de décisions
    decision_files = list(decisions_dir.glob("decisions_*.jsonl"))
    
    if not decision_files:
        logger.error(f"Aucun fichier de décisions trouvé dans {decisions_dir}")
        return None
    
    # Extraire les IDs de session et trouver le plus récent
    session_ids = []
    for file in decision_files:
        match = re.search(r'decisions_(\d{8}_\d{6})\.jsonl', file.name)
        if match:
            session_ids.append(match.group(1))
    
    if not session_ids:
        logger.error("Impossible d'extraire les IDs de session")
        return None
    
    # Trier les IDs par ordre décroissant (le plus récent en premier)
    session_ids.sort(reverse=True)
    
    return session_ids[0]

def analyze_session(session_id, symbol=None, decision_type=None, start_date=None, end_date=None,
                   error_analysis=False, correlation_analysis=False, timeline=False,
                   output_dir="analysis_reports", plot=False):
    """
    Analyse une session de trading spécifique.
    
    Args:
        session_id: ID de la session à analyser
        symbol: Filtrer par symbole
        decision_type: Filtrer par type de décision
        start_date: Date de début
        end_date: Date de fin
        error_analysis: Si True, analyse les erreurs
        correlation_analysis: Si True, analyse les corrélations
        timeline: Si True, génère une chronologie des décisions
        output_dir: Répertoire de sortie pour les rapports
        plot: Si True, génère des graphiques
    """
    # Créer le répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Créer un journal temporaire pointant vers les fichiers de la session spécifiée
    journal = TradingJournal()
    journal.session_id = session_id
    journal.decisions_file = journal.decisions_dir / f"decisions_{session_id}.jsonl"
    journal.signals_file = journal.signals_dir / f"signals_{session_id}.jsonl"
    journal.orders_file = journal.orders_dir / f"orders_{session_id}.jsonl"
    journal.errors_file = journal.errors_dir / f"errors_{session_id}.jsonl"
    
    # Vérifier que les fichiers existent
    if not journal.decisions_file.exists():
        logger.error(f"Fichier de décisions non trouvé: {journal.decisions_file}")
        return
    
    logger.info(f"Analysing session: {session_id}")
    
    # Générer un rapport complet
    report = {
        "session_id": session_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "filters": {
            "symbol": symbol,
            "decision_type": decision_type,
            "start_date": start_date,
            "end_date": end_date
        }
    }
    
    # Analyser les décisions
    decisions_df = journal.analyze_decisions(symbol, decision_type, start_date, end_date)
    
    if decisions_df.empty:
        logger.warning("Aucune décision trouvée avec les filtres spécifiés")
        report["decisions"] = {"message": "Aucune décision trouvée"}
    else:
        # Statistiques sur les décisions
        decision_types = decisions_df["decision_type"].value_counts().to_dict()
        symbols = decisions_df["symbol"].value_counts().to_dict()
        
        # Ajouter au rapport
        report["decisions"] = {
            "total_count": len(decisions_df),
            "decision_types": decision_types,
            "symbols": symbols
        }
        
        # Sauvegarder les décisions filtrées
        decisions_output = output_path / f"decisions_{session_id}.csv"
        decisions_df.to_csv(decisions_output, index=False)
        logger.info(f"Décisions sauvegardées dans {decisions_output}")
    
    # Analyser les erreurs si demandé
    if error_analysis:
        if not journal.errors_file.exists():
            logger.warning(f"Fichier d'erreurs non trouvé: {journal.errors_file}")
            report["errors"] = {"message": "Aucun fichier d'erreurs trouvé"}
        else:
            error_patterns = journal.find_error_patterns()
            report["errors"] = error_patterns
            
            # Générer un graphique des erreurs si demandé
            if plot and "hourly_distribution" in error_patterns:
                plt.figure(figsize=(12, 6))
                hours = list(error_patterns["hourly_distribution"].keys())
                counts = list(error_patterns["hourly_distribution"].values())
                
                plt.bar(hours, counts, color='red')
                plt.xlabel('Heure')
                plt.ylabel('Nombre d\'erreurs')
                plt.title(f'Distribution horaire des erreurs - Session {session_id}')
                plt.xticks(range(0, 24))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                errors_plot = output_path / f"errors_distribution_{session_id}.png"
                plt.savefig(errors_plot)
                plt.close()
                logger.info(f"Graphique des erreurs sauvegardé dans {errors_plot}")
    
    # Analyser les corrélations si demandé
    if correlation_analysis:
        if not journal.errors_file.exists() or not journal.decisions_file.exists():
            logger.warning("Fichiers de journaux incomplets pour l'analyse de corrélation")
            report["correlations"] = {"message": "Données insuffisantes pour l'analyse de corrélation"}
        else:
            correlations = journal.correlate_errors_with_decisions()
            report["correlations"] = correlations
    
    # Générer une chronologie si demandé
    if timeline:
        timeline_df = journal.get_decision_timeline(symbol, start_date, end_date)
        
        if timeline_df.empty:
            logger.warning("Impossible de générer la chronologie: données insuffisantes")
            report["timeline"] = {"message": "Données insuffisantes pour la chronologie"}
        else:
            timeline_output = output_path / f"timeline_{session_id}.csv"
            timeline_df.to_csv(timeline_output, index=False)
            logger.info(f"Chronologie sauvegardée dans {timeline_output}")
            
            report["timeline"] = {
                "events_count": len(timeline_df),
                "start_time": timeline_df["timestamp"].min().isoformat(),
                "end_time": timeline_df["timestamp"].max().isoformat()
            }
            
            # Générer un graphique de la chronologie si demandé
            if plot:
                plt.figure(figsize=(15, 8))
                
                # Grouper par type de décision et compter par heure
                timeline_df["hour"] = timeline_df["timestamp"].dt.floor("H")
                hourly_counts = timeline_df.groupby(["hour", "decision_type"]).size().unstack().fillna(0)
                
                hourly_counts.plot(kind='bar', stacked=True, ax=plt.gca())
                plt.xlabel('Heure')
                plt.ylabel('Nombre de décisions')
                plt.title(f'Chronologie des décisions - Session {session_id}')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                timeline_plot = output_path / f"timeline_{session_id}.png"
                plt.savefig(timeline_plot)
                plt.close()
                logger.info(f"Graphique de la chronologie sauvegardé dans {timeline_plot}")
    
    # Sauvegarder le rapport complet
    report_output = output_path / f"rapport_analyse_{session_id}.json"
    with open(report_output, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Rapport d'analyse sauvegardé dans {report_output}")
    
    return report

def main():
    """Fonction principale du script."""
    args = parse_args()
    
    session_id = args.session
    
    # Si demandé, trouver la session la plus récente
    if args.latest:
        session_id = find_latest_session()
        if not session_id:
            logger.error("Impossible de trouver la session la plus récente")
            return
    
    if not session_id:
        logger.error("Veuillez spécifier un ID de session avec --session ou utiliser --latest")
        return
    
    # Analyser la session
    analyze_session(
        session_id=session_id,
        symbol=args.symbol,
        decision_type=args.decision_type,
        start_date=args.start_date,
        end_date=args.end_date,
        error_analysis=args.error_analysis,
        correlation_analysis=args.correlation_analysis,
        timeline=args.timeline,
        output_dir=args.output_dir,
        plot=args.plot
    )

if __name__ == "__main__":
    main()
