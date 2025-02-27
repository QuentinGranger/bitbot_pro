# Module On-Balance Volume (OBV)

## Description

Le module OBV (On-Balance Volume) est un outil d'analyse technique qui mesure la pression d'achat et de vente sur un marché en utilisant le volume des transactions. Cet indicateur est basé sur le principe que les variations de volume peuvent précéder les variations de prix.

## Fonctionnalités

Le module OBV offre les fonctionnalités suivantes :

1. **Calcul de l'OBV** - Mesure le flux cumulatif du volume en ajoutant le volume lorsque le prix augmente et en le soustrayant lorsque le prix diminue.
2. **Génération de signaux** - Fournit des signaux d'achat et de vente basés sur l'OBV et ses moyennes mobiles.
3. **Détection de divergences** - Identifie les divergences entre le prix et l'OBV, ce qui peut indiquer des retournements potentiels du marché.
4. **Calcul du momentum OBV** - Mesure la force de la tendance de l'OBV.
5. **Calcul du Volume Price Trend (VPT)** - Une variante de l'OBV qui prend en compte les variations de prix.
6. **Analyse complète** - Fournit une analyse détaillée de la pression du marché.

## Utilisation

### Initialisation

```python
from bitbot.strategie.base.OBV import OBVIndicator, OBVSignal
from bitbot.models.market_data import MarketData

# Initialiser l'indicateur OBV
obv_indicator = OBVIndicator(ema_period=20, signal_period=9)

# Créer un objet MarketData avec des données OHLCV
market_data = MarketData(symbol="BTCUSDT", timeframe="1h")
market_data.ohlcv = df  # df est un DataFrame pandas avec des données OHLCV
```

### Calcul de l'OBV

```python
# Calculer l'OBV et ses composants
df_with_obv = obv_indicator.calculate_obv(market_data)
```

### Obtention des signaux

```python
# Obtenir le signal actuel
signal = obv_indicator.get_signal(market_data)
print(f"Signal OBV: {signal.value}")

# Vérifier si l'OBV est en augmentation ou en diminution
is_increasing = obv_indicator.is_increasing(market_data)
is_decreasing = obv_indicator.is_decreasing(market_data)
```

### Détection des divergences

```python
# Détecter les divergences entre le prix et l'OBV
bullish_divergence, bearish_divergence = obv_indicator.detect_divergence(market_data)
```

### Calcul du momentum OBV

```python
# Calculer le momentum de l'OBV
df_momentum = obv_indicator.calculate_obv_momentum(market_data)
momentum = df_momentum['OBV_Momentum'].iloc[-1]
```

### Calcul du Volume Price Trend (VPT)

```python
# Calculer le VPT
df_vpt = obv_indicator.calculate_volume_price_trend(market_data)
```

### Analyse complète

```python
# Obtenir une analyse complète
analysis = obv_indicator.analyze(market_data)
```

## Stratégie de trading basée sur l'OBV

Le module inclut également une stratégie de trading basée sur l'OBV qui peut être utilisée pour générer des signaux d'achat et de vente automatiquement.

```python
from bitbot.strategie.indicators.obv_strategy import OBVStrategy

# Initialiser la stratégie
strategy = OBVStrategy(
    ema_period=20,
    signal_period=9,
    use_divergence=True,
    use_vpt=False
)

# Générer des signaux
signals = strategy.generate_signals(market_data)

# Effectuer un backtest
backtest_results = strategy.backtest(market_data, initial_capital=10000.0)
```

## Scripts de test

Deux scripts de test sont fournis pour démontrer l'utilisation du module OBV :

1. `scripts/test_obv.py` - Teste l'indicateur OBV et affiche ses résultats.
2. `scripts/test_obv_strategy.py` - Teste la stratégie basée sur l'OBV et effectue un backtest.

## Paramètres

### OBVIndicator

- `ema_period` (défaut: 20) - Période pour le calcul de l'EMA de l'OBV.
- `signal_period` (défaut: 9) - Période pour le calcul de la ligne de signal.

### OBVStrategy

- `ema_period` (défaut: 20) - Période pour le calcul de l'EMA de l'OBV.
- `signal_period` (défaut: 9) - Période pour le calcul de la ligne de signal.
- `use_divergence` (défaut: True) - Si True, prend en compte les divergences dans la génération de signaux.
- `use_vpt` (défaut: False) - Si True, utilise le Volume Price Trend au lieu de l'OBV standard.

## Interprétation des signaux

- **Pression acheteuse forte (STRONG_BUY)** - L'OBV augmente fortement et est au-dessus de sa ligne de signal, indiquant une forte pression d'achat.
- **Pression acheteuse (BUY)** - L'OBV est au-dessus de sa ligne de signal, indiquant une pression d'achat.
- **Pression neutre (NEUTRAL)** - L'OBV est proche de sa ligne de signal, indiquant un équilibre entre acheteurs et vendeurs.
- **Pression vendeuse (SELL)** - L'OBV est en-dessous de sa ligne de signal, indiquant une pression de vente.
- **Pression vendeuse forte (STRONG_SELL)** - L'OBV diminue fortement et est en-dessous de sa ligne de signal, indiquant une forte pression de vente.

## Divergences

- **Divergence haussière** - Le prix forme des creux plus bas tandis que l'OBV forme des creux plus hauts, indiquant une possible inversion haussière.
- **Divergence baissière** - Le prix forme des sommets plus hauts tandis que l'OBV forme des sommets plus bas, indiquant une possible inversion baissière.

## Exemples de visualisation

Les scripts de test génèrent des visualisations qui sont enregistrées dans les répertoires suivants :

- `/outputs/obv_test/BTCUSDT_obv_analysis.png` - Visualisation de l'indicateur OBV.
- `/outputs/obv_strategy/BTCUSDT_obv_strategy_backtest.png` - Visualisation du backtest de la stratégie OBV.
