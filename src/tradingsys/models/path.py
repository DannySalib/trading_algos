
from __future__ import annotations

from pydantic import BaseModel, field_validator
from functools import lru_cache
from typing import Any
import importlib

class _Path(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _must_look_like_a_dotted_path(cls, v: str) -> str:
        if not v or not all(part.isidentifier() for part in v.split(".")):
            raise ValueError(f"{v!r} is not a valid dotted path")
        return v

class DataPath(_Path): pass

class FunctionPath(_Path):pass

def access_path(path: _Path) -> Any:
    return _resolve(path.path)

@lru_cache(maxsize=None)
def _resolve(dotted_path: str) -> Any:
    """Resolve 'pkg.module.attr[.attr...]' to the object it names.

    Tries importing progressively shorter prefixes as a module, then
    walks any remaining segments with getattr.
    """
    parts = dotted_path.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        obj = module
        for attr in parts[i:]:
            obj = getattr(obj, attr)  # raises AttributeError if wrong
        return obj
    raise ImportError(f"Could not resolve {dotted_path!r}: no importable prefix")