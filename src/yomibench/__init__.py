"""Bots and a tournament arena for Your Only Move Is HUSTLE."""

# Engines beyond the public ones come from optional plugins (yomibench.plugins). They are imported here, on the
# importing (main) thread, before any engine thread or pool starts: every entry point (yomibench.cli, tools/*)
# imports this package at top level. Without plugins: the real game, the toy engine, tcp and cmd engines.
from yomibench import plugins as _plugins

_plugins.load()
