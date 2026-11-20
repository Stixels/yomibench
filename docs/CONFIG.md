# Local configuration

yomibench needs no configuration. A machine that does more than play (a training rig, a build machine) keeps
its own settings in a TOML file that is never committed, so the repository carries no addresses, machine names
or local paths.

## Where it is read from

The first one found wins:

1. the file named by `$YOMIBENCH_CONFIG` (`-` or an empty value means "no file"; the tests use this),
2. `yomibench.local.toml` at the root of this checkout (listed in `.gitignore`),
3. `~/.yomibench/config.toml`.

Start from [`yomibench.local.example.toml`](../yomibench.local.example.toml). Every key is optional.

## Settings

| key | default | what it does |
|---|---|---|
| `machine.name` | none | a machine id for tools that report per machine |
| `paths.data_root` | none | where a training setup keeps its data |
| `paths.game_dir` | Steam discovery | the game install (`$YOMIBENCH_GAME_DIR` wins) |
| `paths.scene_dir` | none | scene files a plugin engine may read |
| `paths.yomibench_repo` | none | a checkout whose `tools/` other tools run by path (`$YB_REPO` wins) |
| `hub.url` | none | a training hub's address |
| `hub.best_url` | `~/.yomibench/best.json` | where human play's current-best looks (`$YOMIBENCH_BEST` wins) |
| `game.require_steam_install` | `true` | only your own Steam install may be used; `false` only for rigs with archived copies |
| `plugins.modules` | `[]` | extra plugin modules (see [PLUGINS.md](PLUGINS.md)); `$YOMIBENCH_ENGINE_PLUGINS` adds more |
| `privacy.deny_file` | none | a file of private terms that `tools/privacy_scan.py` also refuses |

Environment variables win over the file. Unknown keys print one warning and are ignored. In Python:
`from yomibench import localconfig; localconfig.load()` returns the settings (cached).
