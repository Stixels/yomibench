"""Local, per-machine settings that never go into git: `yomibench.local.toml`.

Everything here is optional. Without a file you get the public defaults: the game must be your own Steam install,
no hub, no machine name and no extra plugins. A training rig names its machine, hub and plugins here instead of in
code, so this repository carries no addresses, hostnames or local paths.

Files, first found wins:
1. `$YOMIBENCH_CONFIG` (a path; `-` or empty means "no file", which tests use),
2. `yomibench.local.toml` at the root of this checkout (gitignored),
3. `~/.yomibench/config.toml`.

Environment variables win over the file: `YOMIBENCH_GAME_DIR`, `YOMIBENCH_BEST`, `YB_REPO`,
`YOMIBENCH_ENGINE_PLUGINS` (added to `plugins.modules`). See `yomibench.local.example.toml` and docs/CONFIG.md.
"""
from __future__ import annotations

import functools
import os
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NAME = "yomibench.local.toml"
ENV = "YOMIBENCH_CONFIG"

# table -> {key: (field, kind)}; kind is "str", "path", "bool" or "list"
_SCHEMA: dict[str, dict[str, tuple[str, str]]] = {
    "machine": {"name": ("machine", "str")},
    "paths": {"data_root": ("data_root", "path"), "game_dir": ("game_dir", "path"),
              "scene_dir": ("scene_dir", "path"), "yomibench_repo": ("yomibench_repo", "path")},
    "hub": {"url": ("hub_url", "str"), "best_url": ("best_url", "str")},
    "game": {"require_steam_install": ("require_steam_install", "bool")},
    "plugins": {"modules": ("plugins", "list")},
    "privacy": {"deny_file": ("privacy_deny_file", "path")},
}
# field -> environment variable that overrides it
_ENV_OVERRIDES = {"game_dir": "YOMIBENCH_GAME_DIR", "best_url": "YOMIBENCH_BEST",
                  "yomibench_repo": "YB_REPO"}


@dataclass(frozen=True)
class LocalConfig:
    machine: str | None = None             # fleet machine id; None for a public user
    data_root: Path | None = None
    game_dir: Path | None = None           # else Steam discovery (game.find_game_dir)
    scene_dir: Path | None = None
    hub_url: str | None = None
    best_url: str | None = None            # human play's current-best (a best.json URL or path)
    require_steam_install: bool = True     # False only on build rigs that run archived copies of the game
    plugins: tuple[str, ...] = ()          # extra plugin modules (yomibench.plugins)
    yomibench_repo: Path | None = None     # for tools that run this checkout's tools/ by path
    privacy_deny_file: Path | None = None  # private terms for tools/privacy_scan.py; keep it outside every repo
    source: Path | None = None             # the file these settings came from


def config_path() -> Path | None:
    """The file `load()` reads, or None."""
    env = os.environ.get(ENV)
    if env is not None:
        return None if env.strip() in ("", "-") else Path(env).expanduser()
    for p in (REPO_ROOT / LOCAL_NAME, Path.home() / ".yomibench" / "config.toml"):
        if p.is_file():
            return p
    return None


def parse(data: dict, source: Path | None = None) -> LocalConfig:
    """Settings from a parsed TOML document (unknown tables and keys: one warning, ignored)."""
    kw: dict = {}
    unknown = []
    for table, body in data.items():
        if table not in _SCHEMA or not isinstance(body, dict):
            unknown.append(table)
            continue
        for key, value in body.items():
            if key not in _SCHEMA[table]:
                unknown.append(f"{table}.{key}")
                continue
            field, kind = _SCHEMA[table][key]
            if kind == "bool":
                if not isinstance(value, bool):
                    raise ValueError(f"{source or 'config'}: {table}.{key} must be true or false")
                kw[field] = value
            elif kind == "list":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ValueError(f"{source or 'config'}: {table}.{key} must be a list of strings")
                kw[field] = tuple(v for v in value if v.strip())
            elif value not in ("", None):
                kw[field] = Path(value).expanduser() if kind == "path" else str(value)
    if unknown:
        warnings.warn(f"{source or 'config'}: ignoring unknown settings {unknown}", stacklevel=3)
    return LocalConfig(source=source, **kw)


def _apply_env(cfg: LocalConfig) -> LocalConfig:
    kw = {f: getattr(cfg, f) for f in LocalConfig.__dataclass_fields__}
    for field, var in _ENV_OVERRIDES.items():
        v = os.environ.get(var)
        if v:
            kw[field] = v if field == "best_url" else Path(v).expanduser()
    extra = [m.strip() for m in os.environ.get("YOMIBENCH_ENGINE_PLUGINS", "").split(",") if m.strip()]
    kw["plugins"] = tuple(dict.fromkeys([*kw["plugins"], *extra]))
    return LocalConfig(**kw)


@functools.lru_cache(maxsize=1)
def load() -> LocalConfig:
    """This machine's settings: environment > file > defaults. Cached; `load.cache_clear()` re-reads."""
    path = config_path()
    cfg = LocalConfig()
    if path is not None:
        with open(path, "rb") as f:
            cfg = parse(tomllib.load(f), path)
    return _apply_env(cfg)
