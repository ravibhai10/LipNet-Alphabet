"""Configuration package for the Alphabet dataset.

The single authoritative configuration loader lives in :mod:`config.loader`.
It reads ``configs/alphabet_dataset.yaml`` and produces the typed runtime
configuration consumed by the data / tracking / representation stages.
"""

from config.loader import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ConfigValidationError,
    Phase2Config,
    QualityConfig,
    SplitConfig,
    TemporalConfig,
    load_phase2_config,
    load_config_dict,
)

__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "Phase2Config",
    "QualityConfig",
    "SplitConfig",
    "TemporalConfig",
    "load_phase2_config",
    "load_config_dict",
    "DEFAULT_CONFIG_PATH",
]
