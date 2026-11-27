"""Start a human-vs-bot session: the bot server (yomibench.human_play) in this process and the real
game windowed from a scratch run dir. Shared by the dev launcher (tools/human_play.py) and the
Windows app (packaging/windows), which differ only in where the game build, the bot and the logs
come from.

The game is only ever read: its exe, pack and DLLs are copied into the run dir (yomibench.game
build_run_dir), which must not be inside a Steam library, and the game runs from there with its own
user data folder.
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, ContextManager

from yomibench.anytime import (BANK_DEEP_TURNS, BANK_REFINE_TURNS, BANK_RESERVE_S, LOCK_CAP_S, LOCK_CONVERGED_S,
                               THINK_CAP_S)
from yomibench.game import EXE, PACK, build_run_dir
from yomibench.human_play import Server, Session, load_quick_policy, worker_paused

# The game version the sim follows: the sha256 of 1.10.0-steam's pack (the archived build's MANIFEST).
# A hash, not game content. Another version may desync every turn or break the mod outright.
KNOWN_PACKS = {"9466f7e4fefb3232ce99847df2d096e28dc14dcaf963ea9b111b935516bbf566": "1.10.0-steam"}
USER_DIR = "YomiBenchHuman"  # the game's user data folder in the run (%APPDATA%\YomiBenchHuman)


def inside_steam(path: Path) -> bool:
    return "steamapps" in str(Path(path).resolve()).lower()


def pause_if_present(pause: Path | None, log: Callable[[str], None] = print) -> ContextManager[bool]:
    """The round-3 worker's PAUSE file, held while playing (worker_paused), but only on a yomibench
    machine: where the worker's folder doesn't exist, nothing is created and this is a no-op."""
    if pause is None or not pause.parent.is_dir():
        return contextlib.nullcontext(False)
    return worker_paused(pause, log)


def documents_dir() -> Path:
    """The user's Documents folder (wherever Windows has moved it, e.g. into OneDrive)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:  # CSIDL_PERSONAL
            return Path(buf.value)
    return Path.home() / "Documents"


def game_version(game_dir: Path) -> str | None:
    """The build's version if its pack is one the sim follows, else None."""
    h = hashlib.sha256()
    with open(game_dir / PACK, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return KNOWN_PACKS.get(h.hexdigest())


def run_game(session: Session, game_dir: Path, run_dir: Path, *, port: int = 0, p1: str = "Ninja", p2: str = "Ninja",
             autoplay: bool = False, speed: float | None = None, tie_to_this_process: bool = False,
             mute: bool = False, user_dir: str = USER_DIR, log: Callable[[str], None] = print) -> Server:
    """Build the run dir, serve `session` on 127.0.0.1:`port` (0: any free port) and run the game
    until its window closes (or Ctrl+C). Returns the closed server; an unfinished game is written as
    abandoned. `user_dir`: the game's settings folder (%APPDATA%\\<user_dir>), never the game's own.
    `mute` (implied by `autoplay`): no sound (Godot's Dummy audio driver), for automated and test
    launches. `tie_to_this_process`: on Windows the game is killed with this process, even if this process is
    killed outright (the app's console window closed)."""
    if inside_steam(run_dir):
        raise SystemExit(f"refusing to run the game from inside a Steam library ({run_dir}); pick another --run-dir")
    server = Server(session, port).start()
    try:
        sha = (session.hello.get("build") or {}).get("yomisim_sha") or "?"
        log(f"bot server ready on 127.0.0.1:{server.port} (engine build {sha[:7]})")
        launch_game(game_dir, run_dir, server.port, session.out, p1=p1, p2=p2, autoplay=autoplay, speed=speed,
                    tie_to_this_process=tie_to_this_process, mute=mute, user_dir=user_dir, log=log)
    finally:
        server.close()  # an unfinished game is written as abandoned
    return server


def launch_game(game_dir: Path, run_dir: Path, port: int, out: Path, *, p1: str = "Ninja", p2: str = "Ninja",
                autoplay: bool = False, speed: float | None = None, tie_to_this_process: bool = False,
                mute: bool = False, user_dir: str = USER_DIR, log: Callable[[str], None] = print,
                watch: Callable[[], None] | None = None) -> int | None:
    """Build the run dir and run the game against the bot server on 127.0.0.1:`port` (served here, or a
    separate process, as the Windows app's: a crash there then can't take the game down with it) until its
    window closes (or Ctrl+C). The game's output goes to `out`/game-stdout.log. `watch` runs every 0.5 s while
    the game is up. Returns the game's exit code (None: stopped with Ctrl+C). The other flags: `run_game`."""
    from yomibench.game import _kill_with_this_process

    if inside_steam(run_dir):
        raise SystemExit(f"refusing to run the game from inside a Steam library ({run_dir}); pick another --run-dir")
    run_dir = build_run_dir(game_dir, run_dir, user_dir_name=user_dir, main_autoload=("YomiHuman", "YomiHuman.gd"))
    proc = None
    try:
        game_args = [str(run_dir / EXE), "--windowed", "--resolution", "1280x720",
                     f"--yomi-human={port}", f"--yomi-human-p1={p1}", f"--yomi-human-p2={p2}"]
        if autoplay:
            game_args.append("--yomi-human-autoplay")
            if speed:
                game_args.append(f"--yomi-human-speed={speed:g}")
        if mute or autoplay:  # every automated or test launch is muted (user rule); a person's keeps sound
            game_args += ["--audio-driver", "Dummy"]
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "game-stdout.log", "ab") as stdout:
            # An autoplay (test) game never shows a window; a person's game does.
            from yomibench.game import popen_hidden
            start = popen_hidden if autoplay else subprocess.Popen
            proc = start(game_args, cwd=run_dir, stdout=stdout, stderr=subprocess.STDOUT)
            if tie_to_this_process:
                _kill_with_this_process(proc)
            if autoplay:
                log("game started in autoplay: it plays the scheduled games and quits.")
            else:
                log("game started: pick characters on the YomiBench panel and press Start game.\n"
                    "Close the game window when you're done (Ctrl+C here also stops everything).")
            while proc.poll() is None:
                time.sleep(0.5)
                if watch is not None:
                    watch()
            return proc.returncode
    except KeyboardInterrupt:
        log("\nstopping")
        return None
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def summary(session: Session, log: Callable[[str], None] = print) -> None:
    rs = session.results
    log(f"\n{len(rs)} game(s) this session:")
    for r in rs:
        sync = "in sync" if not r["desync"] else "DESYNCED (invalid)"
        log(f"  {r['characters'][0]} (you) vs {r['characters'][1]} ({session.label}): {r['result']}, "
            f"{r['turns']} turns, {sync}")
    log(f"turn logs: {session.out / 'games'}\nresult rows: {session.ledger_path}")


def turn_timer_arg(args) -> bool | None:
    """--turn-timer on|off as the Session's turn_timer (None: the preset's default)."""
    v = getattr(args, "turn_timer", None)
    return None if v is None else v == "on"


def session_kwargs(args, bot) -> dict:
    """Session keyword arguments from a launcher's play flags (human_play.add_play_args) and its bot."""
    return dict(label=bot.label, model=bot.model, pick=args.pick, show_estimate=args.show_estimate,
                settings=args.settings, think_until_lockin=args.think_until_lockin,
                cpu=getattr(args, "cpu", "light"),
                think_cap=getattr(args, "think_cap", THINK_CAP_S),
                lock_cap=getattr(args, "lock_cap", LOCK_CAP_S),
                bank_reserve=getattr(args, "bank_reserve", BANK_RESERVE_S),
                bank_deep_turns=getattr(args, "bank_deep_turns", BANK_DEEP_TURNS),
                lock_converged=getattr(args, "lock_converged", LOCK_CONVERGED_S),
                bank_refine_turns=getattr(args, "bank_refine_turns", BANK_REFINE_TURNS),
                turn_timer=turn_timer_arg(args),
                deadline_margin=args.deadline_margin, deadline_fraction=args.deadline_fraction,
                engine_spec=getattr(args, "engine", None), quick_policy=load_quick_policy(args))

