from protgps.utils.registry import register_object
from protgps.utils.classes import ProtGPS
from ray.tune.suggest import BasicVariantGenerator


@register_object("basic", "searcher")
class BasicSearch(BasicVariantGenerator, ProtGPS):
    """Description

    See: https://docs.ray.io/en/releases-0.8.4/tune-searchalg.html#variant-generation-grid-search-random-search
    """

    def __init__(self, args):
        super().__init__()
