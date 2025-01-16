import warnings
from importlib.metadata import version

try:
    __version__ = version("mfglib")
except:
    warnings.warn(
        "mfglib is not pre-installed; probably using some local checkout of mfglib; default __version__ to 0.1.1 but please check the mfglib you indeed import by print(mfglib) after import mfglib."
    )
    __version__ = "0.1.1"
