# Plugins

The public build plays the real game (`game`, `game-pool`), the toy engine, and engines behind `tcp:` or
`cmd:`. Other engines, such as a faster simulator, come from plugins. yomibench never imports a plugin by name.

## What a plugin module can define

```python
ENGINES = {"myengine": connect}          # connect(spec, candidate_set) -> yomibench.engine.Engine
HOOKS = {"batch": "mypkg.batch"}         # a provider by role, imported only when it is asked for
MOD_OVERLAYS = {"YomiBench/Extra.gd": path_to_file}   # files added to or replacing the mod's own at build time
```

- `ENGINES`: `--engine myengine` and `--engine myengine:...` go to `connect`.
- `HOOKS["batch"]`: the batch self-play engine behind `--engine batch[:THREADS]` in `selfplay` and `arena`,
  and in `tools/pool_eval.py`, `tools/explain_game.py` and `tools/do_check.py`. The provider module defines
  `batch_selfplay_parts`, `batch_arena_engines`, `BatchEngine` and `BatchValue`. Without one, `--engine batch`
  exits with "engine 'batch' isn't available in this build".
- `MOD_OVERLAYS`: keys are `YomiBench/<file>` (copied into every server run dir) or `YomiBenchVsBot/<file>`
  (used by `tools/build_vsbot_mod.py` in place of the file of that name).

## How plugins are found

In order, each module imported once at `import yomibench`, on the main thread:

1. the `yomibench.engines` entry-point group of installed packages, e.g. in the plugin's `pyproject.toml`:
   ```toml
   [project.entry-points."yomibench.engines"]
   mysim = "mypkg.engines"
   ```
2. `[plugins] modules` in the local config ([CONFIG.md](CONFIG.md)),
3. `$YOMIBENCH_ENGINE_PLUGINS` (comma-separated module names).

A listed module that isn't installed is skipped; an error inside one is raised. `yomibench.plugins` has the
lookups (`engines()`, `hook(role)`, `batch()`, `mod_overlays(folder)`).
