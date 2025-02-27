# BitBot Pro

BitBot Pro est une solution de trading Bitcoin avancée, optimisée pour Raspberry Pi, offrant des performances professionnelles avec un budget maîtrisé.

## Caractéristiques Principales

- 🚀 Trading haute performance multi-exchange (Binance, Bybit, OKX)
- 📊 Analyses techniques, on-chain et sentimentales
- 🔄 Arbitrage intelligent multi-exchange
- 🛡️ Gestion des risques avancée
- 📱 Interface web responsive et alertes Telegram
- 🔒 Sécurité renforcée
- 💻 Optimisé pour Raspberry Pi

## Prérequis

- Python 3.9+
- Redis
- PostgreSQL
- Connexion Internet stable
- Clés API des exchanges (Binance/Bybit/OKX)

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/bitbot-pro.git
cd bitbot-pro
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

4. Lancer le bot :
```bash
python main.py
```

## Configuration Sécurité

- Configuration IP statique
- Pare-feu UFW
- SSH avec clés RSA
- Protection Fail2Ban

## Contribution

Les contributions sont les bienvenues ! Consultez CONTRIBUTING.md pour les guidelines.

## Licence

MIT License
