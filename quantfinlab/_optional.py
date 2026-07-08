from __future__ import annotations

from importlib import import_module
from types import ModuleType

from quantfinlab.common.errors import MissingKernelsError


def has_cpp_kernels() -> bool:
    """Return whether the optional compiled C++ kernels can be imported."""
    try:
        import_module("quantfinlab._kernels")
    except Exception:
        return False
    return True


def get_cpp_kernels(feature: str, fallback_hint: str | None = None) -> ModuleType:
    """Import the optional C++ kernels or raise a user-facing install error."""
    try:
        return import_module("quantfinlab._kernels")
    except Exception as exc:
        hint = fallback_hint or "use engine='auto', engine='numba', or engine='numpy' when a fallback is available"
        raise MissingKernelsError(
            "Compiled QuantFinLab C++ kernels are not installed, so "
            f"{feature} cannot use engine='cpp'. Install a QuantFinLab wheel for your "
            "platform with compiled kernels, or build from source with CMake and a C++17 "
            "compiler. For pure-Python usage, "
            f"{hint}. To intentionally build without kernels, use "
            "`pip install . --config-settings=cmake.define.quantfinlab_build_cpp=off`."
        ) from exc


def prefer_auto_engine(*, allow_cpp: bool = True, allow_numba: bool = True) -> str:
    """Choose the best available engine for functions with a NumPy fallback."""
    if allow_cpp and has_cpp_kernels():
        return "cpp"
    if allow_numba:
        try:
            import_module("numba")
        except Exception:
            pass
        else:
            return "numba"
    return "numpy"


__all__ = ["get_cpp_kernels", "has_cpp_kernels", "prefer_auto_engine"]
