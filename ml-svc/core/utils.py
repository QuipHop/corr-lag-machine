# ml-svc/core/utils.py
from __future__ import annotations

from typing import Any

import numpy as np


def to_native(obj: Any) -> Any:
    """
    Рекурсивно перетворює numpy-типи у звичайні python-типи,
    щоб Pydantic/FastAPI могли їх серіалізувати.
    """
    # numpy bool -> bool
    if isinstance(obj, np.bool_):
        return bool(obj)

    # numpy цілі
    if isinstance(obj, np.integer):
        return int(obj)

    # numpy з плаваючою
    if isinstance(obj, np.floating):
        return float(obj)

    # масиви / списки / кортежі
    if isinstance(obj, (list, tuple)):
        converted = [to_native(x) for x in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted

    # словники
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}

    # BaseModel / інші об'єкти – повертаємо як є
    return obj
