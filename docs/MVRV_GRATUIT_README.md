# Module MVRV avec API gratuites

## Modifications apportées

Nous avons modifié le module MVRV pour utiliser des API gratuites au lieu de l'API Glassnode qui nécessite un abonnement payant. Voici les principaux changements :

1. **Nouveau client de données on-chain** : `OnChainClient`
   - Utilise des API gratuites comme CoinGecko, Blockchain.info et BlockCypher
   - Respecte les limites de taux des API gratuites pour éviter les blocages
   - Calcule une approximation du ratio MVRV à partir des données de marché

2. **Calcul approximatif du MVRV**
   - Utilise les données de prix et de capitalisation boursière de CoinGecko
   - Calcule une approximation de la capitalisation réalisée en utilisant une moyenne pondérée des prix historiques
   - Fournit un ratio MVRV suffisamment précis pour les besoins d'analyse de marché

3. **Suppression des dépendances à Glassnode**
   - Retrait de la clé API Glassnode des paramètres
   - Remplacement du `GlassnodeClient` par le nouveau `OnChainClient`
   - Adaptation des méthodes pour utiliser les nouvelles sources de données

## Utilisation des API gratuites

Le module utilise désormais les API gratuites suivantes :

1. **CoinGecko** : 
   - Données de prix historiques
   - Données de capitalisation boursière
   - Données de volume de trading

2. **Blockchain.info** : 
   - Statistiques générales de la blockchain
   - Données sur la difficulté de minage

3. **BlockCypher** : 
   - Données UTXO
   - Statistiques de réseau

## Calcul du MVRV

### Méthode originale (Glassnode)
Le ratio MVRV est normalement calculé en divisant la capitalisation boursière (market cap) par la capitalisation réalisée (realized cap). La capitalisation réalisée est calculée en sommant chaque pièce de Bitcoin à sa valeur lorsqu'elle a été déplacée pour la dernière fois.

### Méthode approximative (API gratuites)
Notre nouvelle méthode calcule une approximation de la capitalisation réalisée en utilisant une moyenne pondérée des prix historiques, donnant plus de poids aux prix récents. Cette méthode produit une bonne approximation du ratio MVRV sans nécessiter d'accès aux données on-chain précises.

## Avantages et limites

### Avantages
- **Gratuit** : Aucun abonnement ou paiement requis
- **Simple** : Pas besoin de configurer de clé API payante
- **Accessible** : Fonctionne pour quiconque sans compte Glassnode

### Limites
- **Approximation** : Les valeurs ne sont pas exactement identiques à celles de Glassnode
- **Limites de taux** : Les API gratuites ont des limites de requêtes (50 par minute pour CoinGecko)
- **Couverture réduite** : Certaines métriques avancées ne sont pas disponibles

## Comment utiliser le module

L'utilisation du module reste identique, mais vous n'avez plus besoin de fournir de clé API :

```python
from bitbot.strategie.base.MVRVRatio import MVRVIndicator

# Initialiser l'indicateur MVRV
mvrv_indicator = MVRVIndicator(
    ema_period=50,
    undervalued_threshold=1.0,
    strong_undervalued_threshold=0.75,
    overvalued_threshold=2.5,
    strong_overvalued_threshold=3.5
)

# Récupérer les données MVRV
mvrv_data = mvrv_indicator.get_mvrv_data(asset="BTC", days=365)

# Analyser les données
analysis = mvrv_indicator.analyze(asset="BTC", days=365)
```

## Scripts de test

Les scripts de test ont été mis à jour pour utiliser les API gratuites :

```bash
# Test de l'indicateur MVRV
python scripts/test_mvrv.py --days 365 --asset BTC

# Test de la stratégie MVRV
python scripts/test_mvrv_strategy.py --days 365 --asset BTC
```

## Notes

- Les seuils de MVRV peuvent nécessiter un léger ajustement car ils sont basés sur une approximation
- En cas de dépassement des limites d'API, le système attendra automatiquement
- Pour les analyses très précises ou professionnelles, l'accès à des données on-chain via un abonnement payant peut rester préférable
