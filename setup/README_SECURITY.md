# Guide de Sécurisation de BitBot Pro pour macOS

Ce guide détaille les étapes de sécurisation de votre environnement BitBot Pro sur macOS.

## Prérequis

- macOS 10.15 ou supérieur
- Droits administrateur
- Connexion Internet

## Installation

Le script d'installation est divisé en deux parties pour gérer correctement les privilèges :

1. Rendez le script exécutable :
```bash
chmod +x setup/secure_environment.sh
```

2. Exécutez d'abord la partie utilisateur (installation de Homebrew et fail2ban) :
```bash
./setup/secure_environment.sh user
```

3. Puis exécutez la partie root (configuration du pare-feu et SSH) :
```bash
sudo ./setup/secure_environment.sh root
```

## Configuration IP Statique

1. Ouvrez Préférences Système
2. Cliquez sur "Réseau"
3. Sélectionnez votre interface (Wi-Fi ou Ethernet)
4. Cliquez sur "Avancé"
5. Dans l'onglet "TCP/IP" :
   - Configurez "Configurer IPv4" sur "Manuellement"
   - Entrez votre adresse IP
   - Configurez le masque de sous-réseau
   - Ajoutez la passerelle par défaut

## Configuration SSH

1. Générez une paire de clés RSA :
```bash
ssh-keygen -t rsa -b 4096
```

2. Ajoutez votre clé publique au fichier authorized_keys :
```bash
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Vérification de la Sécurité

Exécutez le script de vérification :
```bash
sudo /usr/local/bin/check_security.sh
```

## Points de Sécurité Configurés

- ✅ Pare-feu macOS intégré
- ✅ Fail2Ban via Homebrew
- ✅ SSH sécurisé
- ✅ IP Statique (à configurer via Préférences Système)

## Maintenance

- Vérifiez régulièrement les logs : `sudo log show --predicate 'process == "sshd"' --last 1h`
- Surveillez Fail2Ban : `brew services list | grep fail2ban`
- Maintenez votre système à jour via les Préférences Système > Mise à jour de logiciels
