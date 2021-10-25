from . import GermanTrafficSigns
from . import CIFAR10
from . import HighRes
from . import CarEnv
from types import SimpleNamespace


def get_settings():
    mainConfig = GermanTrafficSigns.config
    return SimpleNamespace(**mainConfig), mainConfig
