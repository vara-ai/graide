from omegaconf import MISSING

from graide.conf.lib import BaseConfig, register_config
from graide.constants import DecisionLevel

GROUP = "strategy"


@register_config()
class StrategyBaseConfig(BaseConfig):
    _target_: str = "graide.strategy.ScreeningStrategy"


@register_config(group=GROUP)
class NormalTriaging(StrategyBaseConfig):
    _target_: str = "graide.strategy.NormalTriaging"


@register_config(group=GROUP)
class ReaderReplacement(StrategyBaseConfig):
    _target_: str = "graide.strategy.ReaderReplacement"


@register_config(group=GROUP)
class DecisionDeferral(StrategyBaseConfig):
    _target_: str = "graide.strategy.DecisionDeferral"
    deferral_level: DecisionLevel = MISSING


@register_config(group=GROUP)
class RadLevelDD(DecisionDeferral):
    deferral_level: DecisionLevel = DecisionLevel.RAD


@register_config(group=GROUP)
class ProgramLevelDD(DecisionDeferral):
    deferral_level: DecisionLevel = DecisionLevel.PROGRAM


@register_config(group=GROUP)
class DeferralToASingleReader(StrategyBaseConfig):
    _target_: str = "graide.strategy.DeferralToASingleReader"


@register_config(group=GROUP)
class Graide(StrategyBaseConfig):
    _target_: str = "graide.strategy.Graide"


@register_config(group=GROUP)
class StandaloneAI(StrategyBaseConfig):
    _target_: str = "graide.strategy.StandaloneAI"
