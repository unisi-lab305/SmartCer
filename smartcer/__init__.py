__version__ = '0.6.2'

from gymnasium.envs.registration import register

register(
    id='SmartCer-v0',
    entry_point='smartcer.cer_env:SmartCer',
)
