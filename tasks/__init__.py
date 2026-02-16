"""Task registry: auto-discovers task modules in this package."""

import importlib
import pkgutil
from pathlib import Path
from typing import Any

TASK_REGISTRY: dict[str, dict[str, Any]] = {}


def _discover_tasks():
    """Scan all modules in the tasks/ package for TASK_CONFIG + render()."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"tasks.{module_info.name}")
        if hasattr(module, "TASK_CONFIG") and hasattr(module, "render"):
            config = module.TASK_CONFIG
            config["_render"] = module.render
            config["_module"] = module_info.name
            TASK_REGISTRY[config["task_name"]] = config


_discover_tasks()
