from importlib.metadata import version
import warnings

__TORCH_FLOAT__ = 64
# __TORCH_FLOAT__ = 32

try:
    __version__ = version("mfglib")
except:
    warnings.warn("mfglib is not pre-installed; probably using some local checkout of mfglib; default __version__ to 0.1.1 but please check the mfglib you indeed import by print(mfglib) after import mfglib.")
    __version__ = "0.1.1"


import torch
if __TORCH_FLOAT__ == 64:
    torch.set_default_dtype(torch.float64) 

__all__ = [__TORCH_FLOAT__]

