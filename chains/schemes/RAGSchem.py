import sys
from importlib import import_module

# Share the module object so legacy patches and cached resources remain identical.
sys.modules[__name__] = import_module("schemas.rag")
