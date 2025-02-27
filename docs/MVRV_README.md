# Module MVRV (Market Value to Realized Value)

## Description

Le module MVRV (Market Value to Realized Value) est un outil d'analyse on-chain qui permet d'évaluer si le Bitcoin est surévalué ou sous-évalué par rapport à sa valeur réalisée. Cet indicateur est basé sur le rapport entre la capitalisation boursière (Market Cap) et la capitalisation réalisée (Realized Cap).

## Fonctionnalités

Le module MVRV offre les fonctionnalités suivantes :

1. **Récupération des données MVRV** - Obtient les données du ratio MVRV via l'API Glassnode.
2. **Calcul du Z-score MVRV** - Mesure l'écart du ratio MVRV par rapport à sa moyenne historique.
3. **Génération de signaux** - Fournit des signaux basés sur les seuils de surévaluation et de sous-évaluation.
4. **Analyse du cycle de marché** - Détermine la position actuelle dans le cycle de marché.
5. **Recommandations d'investissement** - Propose des stratégies d'allocation basées sur l'état du marché.

## Prérequis

Pour utiliser ce module, vous avez besoin d'une clé API Glassnode. Vous pouvez obtenir une clé gratuite en vous inscrivant sur [Glassnode](https://glassnode.com/).

## Configuration

Ajoutez votre clé API Glassnode à votre fichier `.env` :

```
GLASSNODE_API_KEY=votre_clé_api
```

## Utilisation

### Initialisation

```python
from bitbot.strategie.base.MVRVRatio import MVRVIndicator, MVRVSignal

# Initialiser l'indicateur MVRV
mvrv_indicator = MVRVIndicator(
    api_key=None,  # Utilise la clé API depuis les variables d'environnement
    ema_period=50,
    undervalued_threshold=1.0,
    strong_undervalued_threshold=0.75,
    overvalued_threshold=2.5,
    strong_overvalued_threshold=3.5
)
```

### Récupération des données MVRV

```python
# Récupérer les données MVRV pour le Bitcoin sur les 365 derniers jours
mvrv_data = mvrv_indicator.get_mvrv_data(asset="BTC", days=365)
```

### Calcul du Z-score

```python
# Calculer le Z-score MVRV
mvrv_data_with_z = mvrv_indicator.calculate_mvrv_z_score(mvrv_data)
```

### Obtention des signaux

```python
# Obtenir le signal actuel
signal = mvrv_indicator.get_signal(mvrv_data)
print(f"Signal MVRV: {signal.value}")

# Vérifier si le marché est sous-évalué ou surévalué
is_undervalued = mvrv_indicator.is_undervalued(mvrv_data)
is_overvalued = mvrv_indicator.is_overvalued(mvrv_data)
```

### Analyse complète

```python
# Obtenir une analyse complète
analysis = mvrv_indicator.analyze(asset="BTC", days=365)
```

### Visualisation

```python
# Créer un graphique du ratio MVRV
fig = mvrv_indicator.plot_mvrv(mvrv_data_with_z)
plt.show()
```

## Stratégie de trading basée sur le MVRV

Le module inclut également une stratégie de trading basée sur le MVRV qui peut être utilisée pour générer des signaux d'achat et de vente, analyser la position dans le cycle de marché et fournir des recommandations d'investissement.

```python
from bitbot.strategie.indicators.mvrv_strategy import MVRVStrategy

# Initialiser la stratégie
strategy = MVRVStrategy(
    api_key=None,  # Utilise la clé API depuis les variables d'environnement
    ema_period=50,
    undervalued_threshold=1.0,
    strong_undervalued_threshold=0.75,
    overvalued_threshold=2.5,
    strong_overvalued_threshold=3.5,
    use_z_score=True,
    z_score_threshold=2.0
)

# Générer des signaux
signals = strategy.generate_signals(asset="BTC", days=365)

# Obtenir la position dans le cycle de marché
cycle_position = strategy.get_market_cycle_position(asset="BTC", days=365)

# Obtenir des recommandations d'investissement
recommendation = strategy.get_investment_recommendation(asset="BTC", days=365)
```

## Scripts de test

Deux scripts de test sont fournis pour démontrer l'utilisation du module MVRV :

1. `scripts/test_mvrv.py` - Teste l'indicateur MVRV et affiche ses résultats.
2. `scripts/test_mvrv_strategy.py` - Teste la stratégie basée sur le MVRV et génère des recommandations d'investissement.

### Exécution des scripts de test

```bash
# Test de l'indicateur MVRV
python scripts/test_mvrv.py --api-key VOTRE_CLE_API --days 365 --asset BTC

# Test de la stratégie MVRV
python scripts/test_mvrv_strategy.py --api-key VOTRE_CLE_API --days 365 --asset BTC
```

## Paramètres

### MVRVIndicator

- `api_key` (optionnel) - Clé API Glassnode. Si non spécifiée, tente de la récupérer depuis les variables d'environnement.
- `ema_period` (défaut: 50) - Période pour le calcul de l'EMA du ratio MVRV.
- `undervalued_threshold` (défaut: 1.0) - Seuil pour considérer le marché comme sous-évalué.
- `strong_undervalued_threshold` (défaut: 0.75) - Seuil pour considérer le marché comme fortement sous-évalué.
- `overvalued_threshold` (défaut: 2.5) - Seuil pour considérer le marché comme surévalué.
- `strong_overvalued_threshold` (défaut: 3.5) - Seuil pour considérer le marché comme fortement surévalué.

### MVRVStrategy

- Tous les paramètres de MVRVIndicator, plus :
- `use_z_score` (défaut: True) - Si True, utilise également le Z-score pour générer des signaux.
- `z_score_threshold` (défaut: 2.0) - Seuil pour le Z-score (valeur absolue).

## Interprétation des signaux

- **Fortement sous-évalué (STRONG_UNDERVALUED)** - Le ratio MVRV est inférieur au seuil de forte sous-évaluation (par défaut 0.75). C'est généralement un excellent moment pour acheter.
- **Sous-évalué (UNDERVALUED)** - Le ratio MVRV est inférieur au seuil de sous-évaluation (par défaut 1.0). C'est généralement un bon moment pour acheter.
- **Neutre (NEUTRAL)** - Le ratio MVRV est entre les seuils de sous-évaluation et de surévaluation. Le marché est dans une zone d'équilibre.
- **Surévalué (OVERVALUED)** - Le ratio MVRV est supérieur au seuil de surévaluation (par défaut 2.5). C'est généralement un bon moment pour prendre des bénéfices partiels.
- **Fortement surévalué (STRONG_OVERVALUED)** - Le ratio MVRV est supérieur au seuil de forte surévaluation (par défaut 3.5). C'est généralement un excellent moment pour vendre.

## Cycles de marché

La stratégie MVRV peut également déterminer la position actuelle dans le cycle de marché :

- **Fond de marché** - Le marché est fortement sous-évalué.
- **Accumulation** - Le marché est sous-évalué.
- **Début de tendance haussière** - Le marché est légèrement sous-évalué.
- **Milieu de cycle** - Le marché est dans une zone neutre.
- **Distribution** - Le marché est surévalué.
- **Sommet de marché** - Le marché est fortement surévalué.

## Recommandations d'investissement

La stratégie MVRV fournit des recommandations d'investissement basées sur l'état du marché :

- **Acheter agressivement** - Lorsque le marché est fortement sous-évalué.
- **Acheter** - Lorsque le marché est sous-évalué.
- **Acheter progressivement** - Lorsque le marché est légèrement sous-évalué.
- **Conserver** - Lorsque le marché est dans une zone neutre.
- **Vendre partiellement** - Lorsque le marché est surévalué.
- **Vendre** - Lorsque le marché est fortement surévalué.

## Exemples de visualisation

Les scripts de test génèrent des visualisations qui sont enregistrées dans les répertoires suivants :

- `/outputs/mvrv/BTC_mvrv_analysis.png` - Visualisation de l'indicateur MVRV.
- `/outputs/mvrv_strategy/BTC_mvrv_strategy.png` - Visualisation de la stratégie MVRV avec recommandations.
