"""Optional plugins: engines and hooks beyond the public ones, found without importing any of them by name.

A plugin is a module with any of:

- `ENGINES = {name: factory(spec, candidate_set) -> Engine}`: engine specs `name` and `name:...` (engine.connect);
- `HOOKS = {hook: "module.path"}`: providers named by role, imported only when asked for. The one role today is
  `"batch"`, the batch self-play engine (`batch_selfplay_parts`, `batch_arena_engines`, `BatchEngine`,
  `BatchValue`) that `selfplay`/`arena --engine batch`, tools/pool_eval.py, tools/explain_game.py and
  tools/do_check.py use;
- `MOD_OVERLAYS = {"YomiBench/<file>" or "YomiBenchVsBot/<file>": path}`: files copied over (or next to) the mod's
  own when a run dir or the VsBot mod is built.

Plugin modules come from, in order: the `yomibench.engines` entry-point group of installed distributions, the
local config's `[plugins] modules` (yomibench.localconfig), and `$YOMIBENCH_ENGINE_PLUGINS` (comma-separated).
A module that isn't installed is skipped; an error inside one is raised. See docs/PLUGINS.md.

`load()` runs from `yomibench/__init__.py`, on the importing (main) thread, before any engine thread or pool
starts: some native extensions must not be imported first by a short-lived worker thread.
"""
from __future__ import annotations

import functools
import importlib
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

GROUP = "yomibench.engines"
# Names a build may lack: engine.connect gives a clear error for these instead of "unknown spec".
KNOWN_NAMES = ("sim", "batch")


def module_names() -> list[str]:
    from yomibench import localconfig

    names = [ep.module for ep in metadata.entry_points(group=GROUP)]
    names += list(localconfig.load().plugins)
    return list(dict.fromkeys(n for n in names if n))


@functools.lru_cache(maxsize=1)
def load() -> tuple[ModuleType, ...]:
    """Import every plugin module once (cached; `load.cache_clear()` re-discovers)."""
    out = []
    for name in module_names():
        try:
            out.append(importlib.import_module(name))
        except ModuleNotFoundError as e:
            if e.name != name and not name.startswith(f"{e.name}."):
                raise
    return tuple(out)


def engines() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mod in load():
        out.update(getattr(mod, "ENGINES", {}))
    return out


def hook(role: str) -> ModuleType | None:
    """The provider of `role` (the first plugin naming it), imported now; None when no plugin offers it."""
    for mod in load():
        target = getattr(mod, "HOOKS", {}).get(role)
        if target is not None:
            return importlib.import_module(target) if isinstance(target, str) else target
    return None


def batch() -> ModuleType:
    """The batch engine's provider, or SystemExit with the message the public build has always given."""
    b = hook("batch")
    if b is None:
        raise SystemExit("engine 'batch' isn't available in this build; use game-pool:auto (see docs/SETUP.md)")
    return b


def mod_overlays(mod: str) -> dict[str, Path]:
    """{file name: source path} that plugins add to or replace in mod folder `mod` ("YomiBench", "YomiBenchVsBot")."""
    out: dict[str, Path] = {}
    for m in load():
        for key, path in getattr(m, "MOD_OVERLAYS", {}).items():
            folder, _, name = key.partition("/")
            if folder == mod and name:
                out[name] = Path(path)
    return out
