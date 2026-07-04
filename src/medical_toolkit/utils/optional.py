from importlib import import_module
from types import ModuleType


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency with an actionable error message.

    Args:
        module: Fully qualified module name.
        extra: Project extra that installs the dependency.

    Returns:
        The imported module.

    Raises:
        ImportError: If the optional dependency cannot be imported.
    """
    try:
        return import_module(module)
    except ImportError as error:
        package = module.split(".", maxsplit=1)[0]
        raise ImportError(
            f"Optional dependency {package!r} is required. "
            f'Install it with: pip install "medical-imaging-toolkit[{extra}]"'
        ) from error
