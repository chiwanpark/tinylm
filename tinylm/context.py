from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class Context:
    pass


_current_context: Optional[Context] = None


@contextmanager
def current_context(context: Context) -> Generator[None, None, None]:
    global _current_context
    _prev_context = _current_context
    _current_context = context
    try:
        yield
    finally:
        _current_context = _prev_context


def get_context() -> Context:
    if _current_context is None:
        raise RuntimeError("No context is set. Use 'with with_context(...)' to set one.")
    return _current_context
