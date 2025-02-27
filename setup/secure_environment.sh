#!/bin/bash

# Fonction pour vérifier si l'utilisateur est root
check_root() {
    if [ "$(id -u)" -ne 0 ]; then 
        echo "Cette partie du script doit être exécutée en tant que root"
        exit 1
    fi
}

# Fonction pour les tâches nécessitant root
setup_root_tasks() {
    check_root

    echo "🔒 Configuration de la sécurité de l'environnement BitBot Pro pour macOS (tâches root)..."

    # Configuration du pare-feu macOS
    echo "🛡️ Configuration du pare-feu macOS..."
    sudo defaults write /Library/Preferences/com.apple.alf globalstate -int 1
    sudo defaults write /Library/Preferences/com.apple.alf stealthenabled -int 1
    sudo defaults write /Library/Preferences/com.apple.alf loggingenabled -int 1
    sudo pkill -HUP socketfilterfw

    # Configuration de SSH
    echo "🔑 Renforcement de la configuration SSH..."
    if [ ! -f /etc/ssh/sshd_config.original ]; then
        cp /etc/ssh/sshd_config /etc/ssh/sshd_config.original
    fi

    cat > /etc/ssh/sshd_config << EOF
Port 22
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
AllowUsers $(whoami)
EOF

    # Création du script de vérification
    cat > /usr/local/bin/check_security.sh << 'EOF'
#!/bin/bash

echo "🔍 Vérification de la sécurité..."
echo "Pare-feu macOS Status:"
sudo defaults read /Library/Preferences/com.apple.alf globalstate

echo -e "\nSSH Config Check:"
sshd -t

echo -e "\nConnexions actives:"
netstat -an | grep LISTEN
EOF

    chmod +x /usr/local/bin/check_security.sh
}

# Fonction pour les tâches utilisateur
setup_user_tasks() {
    echo "🔒 Configuration de la sécurité de l'environnement BitBot Pro pour macOS (tâches utilisateur)..."

    # Création du dossier SSH
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh

    # Installation de Homebrew si non présent
    if ! command -v brew &> /dev/null; then
        echo "📦 Installation de Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Installation de fail2ban via Homebrew
    echo "🚫 Installation de Fail2Ban..."
    brew install fail2ban

    echo "
🎉 Installation terminée ! Actions à effectuer :

1. Configuration réseau macOS :
   - Ouvrez Préférences Système > Réseau
   - Sélectionnez votre interface
   - Configurez une IP statique dans les paramètres IPv4

2. Générer une clé SSH (si non existante) :
   ssh-keygen -t rsa -b 4096

3. Vérifiez que le pare-feu est actif :
   sudo defaults read /Library/Preferences/com.apple.alf globalstate
   (1 = activé, 0 = désactivé)

4. Démarrer fail2ban :
   brew services start fail2ban

5. Pour vérifier la sécurité :
   sudo /usr/local/bin/check_security.sh
"
}

# Script principal
case "$1" in
    "root")
        setup_root_tasks
        ;;
    "user")
        setup_user_tasks
        ;;
    *)
        echo "Usage: $0 [root|user]"
        echo "  root : Exécuter les tâches nécessitant des privilèges root"
        echo "  user : Exécuter les tâches utilisateur (Homebrew, etc.)"
        exit 1
        ;;
esac
