import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfglib")
except PackageNotFoundError:
    warnings.warn(
        "mfglib is not installed as a package, you are importing it as "
        "a local module. __version__ is set to None."
    )
    __version__ = None  # type: ignore[assignment]
