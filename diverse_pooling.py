# Bridges the standalone `pytorch-pooling` submodule (a directory name that
# isn't a valid Python identifier, so it can't be `import`-ed directly) into
# this project by putting it on `sys.path`, then re-exports the pooling
# modules used as drop-in replacements for `nn.MaxPool2d` elsewhere in this
# codebase.
import os
import sys

_POOLING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch-pooling")
if _POOLING_DIR not in sys.path:
    sys.path.insert(0, _POOLING_DIR)

from Pooling.pooling_method.softpool import SoftPool2d
from Pooling.pooling_method.MixedPool import mixedPool
from Pooling.pooling_method.lip_pooling import SimplifiedLIP

__all__ = ["SoftPool2d", "mixedPool", "SimplifiedLIP"]
