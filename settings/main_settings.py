from settings import GermanTrafficSigns
from settings import CIFAR10
from settings import HighRes
from types import SimpleNamespace


def get_settings():
    mainConfig = CIFAR10.config
    return SimpleNamespace(**mainConfig), mainConfig
