#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratégies basées sur l'analyse du volume.

Ce module contient les stratégies qui utilisent principalement le volume
comme indicateur pour générer des signaux de trading.
"""

from bitbot_pro.strategies.base_strategies.volume.vwap import (
    VWAPStrategy,
    IntraDayVWAPStrategy, 
    MultiPeriodVWAPStrategy
)

from bitbot_pro.strategies.base_strategies.volume.obv import (
    OBVStrategy,
    OBVDivergenceStrategy,
    RateOfChangeOBVStrategy
)

__all__ = [
    'VWAPStrategy',
    'IntraDayVWAPStrategy',
    'MultiPeriodVWAPStrategy',
    'OBVStrategy',
    'OBVDivergenceStrategy',
    'RateOfChangeOBVStrategy'
]
