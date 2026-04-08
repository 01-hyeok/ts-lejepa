from importlib import import_module

from . import univariate

__all__ = ["univariate", "multivariate"]


def __getattr__(name):
    if name == "multivariate":
        return import_module(".multivariate", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
