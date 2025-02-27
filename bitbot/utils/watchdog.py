"""
Watchdog pour surveiller et redémarrer automatiquement le bot en cas de crash.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional, List
import psutil

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from bitbot.utils.logger import logger, setup_logger

class BotProcess:
    """Gestion du processus du bot."""
    
    def __init__(self, cmd: List[str], cwd: str):
        """
        Args:
            cmd: Commande à exécuter pour démarrer le bot
            cwd: Répertoire de travail
        """
        self.cmd = cmd
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.start_time = 0
        self.restart_count = 0
        self.max_restarts = 5
        self.restart_window = 300  # 5 minutes
    
    def start(self) -> bool:
        """
        Démarre le processus du bot.
        
        Returns:
            True si le démarrage est réussi, False sinon
        """
        if self.process and self.process.poll() is None:
            logger.warning("Le bot est déjà en cours d'exécution")
            return False
        
        # Vérifier si on n'a pas trop de redémarrages
        current_time = time.time()
        if current_time - self.start_time < self.restart_window:
            self.restart_count += 1
            if self.restart_count > self.max_restarts:
                logger.error(f"Trop de redémarrages ({self.restart_count}) dans la fenêtre de {self.restart_window}s")
                return False
        else:
            self.restart_count = 1
        
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.start_time = current_time
            logger.info(f"Bot démarré (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du bot: {str(e)}")
            return False
    
    def stop(self):
        """Arrête proprement le processus du bot."""
        if self.process:
            try:
                # Envoyer SIGTERM pour un arrêt propre
                self.process.terminate()
                
                # Attendre jusqu'à 10 secondes
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Si le processus ne s'arrête pas, forcer l'arrêt
                    self.process.kill()
                    logger.warning("Le bot a été forcé de s'arrêter")
                else:
                    logger.info("Le bot s'est arrêté proprement")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du bot: {str(e)}")
            
            finally:
                self.process = None
    
    def is_running(self) -> bool:
        """
        Vérifie si le bot est en cours d'exécution.
        
        Returns:
            True si le bot est en cours d'exécution
        """
        return bool(self.process and self.process.poll() is None)

class BotWatchdog(FileSystemEventHandler):
    """Watchdog pour surveiller et redémarrer le bot."""
    
    def __init__(self, bot_script: str, cwd: str):
        """
        Args:
            bot_script: Chemin vers le script principal du bot
            cwd: Répertoire de travail
        """
        super().__init__()
        self.bot_script = Path(bot_script)
        self.cwd = cwd
        self.bot = BotProcess(
            cmd=[sys.executable, str(self.bot_script)],
            cwd=self.cwd
        )
        self.observer = Observer()
    
    def start(self):
        """Démarre le watchdog et le bot."""
        # Démarrer le bot
        if not self.bot.start():
            logger.error("Impossible de démarrer le bot")
            return
        
        # Configurer la surveillance du système de fichiers
        self.observer.schedule(self, self.cwd, recursive=True)
        self.observer.start()
        logger.info("Watchdog démarré")
        
        try:
            while True:
                # Vérifier l'état du bot
                if not self.bot.is_running():
                    logger.warning("Le bot s'est arrêté, tentative de redémarrage")
                    if not self.bot.start():
                        logger.error("Échec du redémarrage du bot")
                        break
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Arrêt demandé")
        
        finally:
            self.stop()
    
    def stop(self):
        """Arrête le watchdog et le bot."""
        self.observer.stop()
        self.observer.join()
        self.bot.stop()
        logger.info("Watchdog arrêté")
    
    def on_modified(self, event):
        """
        Réagit aux modifications de fichiers.
        Ne redémarre pas le bot pour les fichiers de logs.
        """
        if event.src_path.endswith(('.log', '.pyc', '.dvc')):
            return
        
        logger.info(f"Modification détectée : {event.src_path}")
        self.bot.stop()
        self.bot.start()

def cleanup_zombies():
    """Nettoie les processus zombies du bot."""
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            
            cmdline = proc.cmdline()
            if cmdline and str(Path(cmdline[0]).name) == "python" and any(
                str(Path(cmd).name) == "main.py" for cmd in cmdline[1:]
            ):
                logger.warning(f"Nettoyage du processus zombie : {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def run_watchdog(bot_script: str, cwd: str):
    """
    Point d'entrée principal pour démarrer le watchdog.
    
    Args:
        bot_script: Chemin vers le script principal du bot
        cwd: Répertoire de travail
    """
    setup_logger()
    cleanup_zombies()
    
    # Gestionnaire de signaux pour arrêt propre
    def signal_handler(signum, frame):
        logger.info(f"Signal reçu : {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Démarrer le watchdog
    watchdog = BotWatchdog(bot_script, cwd)
    watchdog.start()

if __name__ == "__main__":
    # Exemple d'utilisation
    script_dir = Path(__file__).parent.parent.parent
    run_watchdog(
        bot_script=str(script_dir / "main.py"),
        cwd=str(script_dir)
    )
