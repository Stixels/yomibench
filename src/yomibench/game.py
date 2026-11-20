"""Run the real game as protocol engine servers (the GDScript in mod/YomiBench).

Each server is the game's own exe started from a scratch run directory (never the install):
the exe, pack and DLLs are copied there, plus an `override.cfg` that

- replaces the `SteamHustle` autoload with a stub (Steam init crashes without an app id),
- adds the `YomiServer` autoload (TCP server, protocol, match driver),
- points the user data dir at its own folder so the player's settings are untouched.

Nothing from the game is ever written back to the install or into this repo.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# The game ships Windows and Linux builds; the Linux one runs in Docker (see docker/).
LINUX = sys.platform.startswith("linux")
EXE = "YourOnlyMoveIsHUSTLE.x86_64" if LINUX else "YourOnlyMoveIsHUSTLE.exe"
LIB_GLOB = "*.so" if LINUX else "*.dll"
PACK = "YourOnlyMoveIsHUSTLE.pck"
DEFAULT_PORT = 7777
MOD_SCRIPTS = ("SteamStub.gd", "YomiServer.gd", "YomiMatch.gd", "Candidates.gd", "Candidates4.gd", "Candidates2.gd", "YomiRender.gd",
               "YomiHuman.gd", "candidate_rules_v4.json", "candidate_rules_v5.json", "candidate_rules_v6.json", "Legality.gd",
               "rules_aliases.json", "ActionSpace.gd")
# Plugins may add files to the mod's scripts or replace some (yomibench.plugins.mod_overlays); none publicly.
STEAM_SUBDIR = Path("steamapps") / "common" / "YourOnlyMoveIsHUSTLE"
STEAM_APP_ID = "2212330"
# Only the player's own Steam install: a folder that Steam (or SteamCMD, signed in to the player's account) put
# under a library's steamapps/common, next to the app's manifest. Every headless copy is made from that install,
# on the player's own PC; nothing of the game ships with yomibench. Always required unless this machine's local
# config says otherwise (`[game] require_steam_install = false`, for build rigs that run archived copies).
def _require_steam_install() -> bool:
    from yomibench import localconfig

    return localconfig.load().require_steam_install


REQUIRE_STEAM_INSTALL = _require_steam_install()


def steam_install_ok(game_dir: str | os.PathLike[str]) -> bool:
    """Is `game_dir` a Steam library's install of the game (steamapps/common/<installdir> with its manifest)?"""
    p = Path(game_dir).resolve()
    steamapps = p.parent.parent
    if p.parent.name.lower() != "common" or steamapps.name.lower() != "steamapps":
        return False
    manifest = steamapps / f"appmanifest_{STEAM_APP_ID}.acf"
    if not manifest.exists():
        return False
    m = re.search(r'"installdir"\s+"([^"]+)"', manifest.read_text(encoding="utf-8", errors="replace"))
    return bool(m) and m.group(1).lower() == p.name.lower()


# The game build this release is tested against (arena/spec.json pins the same one for rated games). Another build
# usually works but isn't supported: a game update can change moves, so results may differ.
SUPPORTED_GAME = {"version": "1.10.0", "steam_build_id": "23959498"}
_warned_build = False


def installed_build(game_dir: str | os.PathLike[str]) -> str | None:
    """The Steam build id of the install at `game_dir`, from its library's app manifest (None when there's none)."""
    manifest = Path(game_dir).resolve().parent.parent / f"appmanifest_{STEAM_APP_ID}.acf"
    try:
        m = re.search(r'"buildid"\s+"(\d+)"', manifest.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return m.group(1) if m else None


def warn_if_unsupported(game_dir: str | os.PathLike[str]) -> str | None:
    """Warns once per process (on stderr) when the install isn't the supported game build; returns the warning."""
    global _warned_build
    if os.environ.get("YOMIBENCH_NO_BUILD_WARNING"):
        return None
    build = installed_build(game_dir)
    if build is None or build == SUPPORTED_GAME["steam_build_id"]:
        return None
    msg = (f"warning: your game is Steam build {build}; yomibench supports {SUPPORTED_GAME['version']} (build "
           f"{SUPPORTED_GAME['steam_build_id']}). It may still work, but results can differ after a game update.")
    if not _warned_build:
        print(msg, file=sys.stderr, flush=True)
        _warned_build = True
    return msg


NOT_STEAM = ("yomibench runs only your own Steam install of the game (a library's steamapps/common/"
             "YourOnlyMoveIsHUSTLE, next to appmanifest_2212330.acf); {} is not one")


def mod_dir() -> Path:
    """mod/YomiBench in this checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "mod" / "YomiBench"
        if (candidate / "YomiServer.gd").exists():
            return candidate
    raise FileNotFoundError("mod/YomiBench not found next to the yomibench package")


def default_run_dir() -> Path:
    # Not under AppData: the Microsoft Store Python virtualizes writes there, and the game
    # exe (started outside that sandbox) then can't find its DLLs.
    return Path(os.environ.get("YOMIBENCH_RUN_DIR") or Path.home() / ".yomibench" / "run")


def find_game_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The install: `explicit`, $YOMIBENCH_GAME_DIR, the local config's `[paths] game_dir`, or a Steam library."""
    from yomibench import localconfig

    tried: list[Path] = []
    for raw in (explicit, os.environ.get("YOMIBENCH_GAME_DIR"), localconfig.load().game_dir):
        if raw:
            p = Path(raw)
            if (p / EXE).exists():
                if REQUIRE_STEAM_INSTALL and not steam_install_ok(p):
                    raise FileNotFoundError(NOT_STEAM.format(p))
                warn_if_unsupported(p)
                return p
            tried.append(p)
    for lib in _steam_libraries():
        p = lib / STEAM_SUBDIR
        if (p / EXE).exists() and (not REQUIRE_STEAM_INSTALL or steam_install_ok(p)):
            warn_if_unsupported(p)
            return p
        tried.append(p)
    raise FileNotFoundError(
        "game install not found; pass --game-dir or set YOMIBENCH_GAME_DIR (tried: "
        + ", ".join(str(t) for t in tried) + ")"
    )


def _steam_libraries() -> list[Path]:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam"]
    if sys.platform == "win32":  # Steam installed elsewhere: its own record of where
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                reg = Path(winreg.QueryValueEx(key, "SteamPath")[0])
            if all(reg.resolve() != r.resolve() for r in roots):
                roots.append(reg)
        except OSError:
            pass
    libs: list[Path] = list(roots)
    for root in roots:
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():
            text = vdf.read_text(encoding="utf-8", errors="replace")
            libs += [Path(m.replace("\\\\", "\\")) for m in re.findall(r'"path"\s+"([^"]+)"', text)]
    return libs


def build_run_dir(game_dir: Path, run_dir: Path | None = None, user_dir_name: str = "YomiBenchServer",
                  main_autoload: tuple[str, str] = ("YomiServer", "YomiServer.gd")) -> Path:
    """Copy the game binaries and our scripts into `run_dir` and write override.cfg. Idempotent."""
    if REQUIRE_STEAM_INSTALL and not steam_install_ok(game_dir):  # every copy is made here
        raise FileNotFoundError(NOT_STEAM.format(game_dir))
    run_dir = run_dir or default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    files = [game_dir / EXE, game_dir / PACK, *sorted(game_dir.glob(LIB_GLOB))]
    for src in files:
        dst = run_dir / src.name
        st = src.stat()
        if not dst.exists() or dst.stat().st_size != st.st_size or int(dst.stat().st_mtime) != int(st.st_mtime):
            shutil.copy2(src, dst)
    scripts = run_dir / "yomibench"
    scripts.mkdir(exist_ok=True)
    for name in MOD_SCRIPTS:
        shutil.copyfile(mod_dir() / name, scripts / name)
    from yomibench import plugins

    overlays = plugins.mod_overlays("YomiBench")
    for name, src in overlays.items():
        shutil.copyfile(src, scripts / name)
    for old in scripts.glob("*.gd"):  # a script a plugin added to an earlier build of this run dir
        if old.name not in MOD_SCRIPTS and old.name not in overlays:
            old.unlink()
    # Godot wants forward slashes, even on Windows.
    s = scripts.resolve().as_posix()
    (run_dir / "override.cfg").write_text(
        "[application]\n"
        f'config/custom_user_dir_name="{user_dir_name}"\n\n'
        "[autoload]\n"
        f'SteamHustle="*{s}/SteamStub.gd"\n'
        f'{main_autoload[0]}="*{s}/{main_autoload[1]}"\n',
        encoding="utf-8",
    )
    return run_dir


_job = None


def _kill_with_this_process(proc: subprocess.Popen[bytes]) -> None:
    """On Windows, put `proc` in a job object that kills it when this Python process exits, even
    if it is killed outright (otherwise a game server restarted by `serve` outlives it). Elsewhere
    servers are stopped by `serve`'s own cleanup, or by Docker stopping the container."""
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    if _job is None:
        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job:
            print(f"warning: CreateJobObject failed ({ctypes.get_last_error()}); servers may outlive serve", flush=True)
            return
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            print(f"warning: SetInformationJobObject failed ({ctypes.get_last_error()})", flush=True)
            k32.CloseHandle(job)
            return
        _job = job  # kept open for this process's lifetime; closing it kills the servers
    handle = k32.OpenProcess(0x0101, False, proc.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
    if not handle:
        print(f"warning: OpenProcess({proc.pid}) failed ({ctypes.get_last_error()})", flush=True)
        return
    if not k32.AssignProcessToJobObject(_job, handle):
        print(f"warning: could not tie pid {proc.pid} to serve ({ctypes.get_last_error()})", flush=True)
    k32.CloseHandle(handle)


def popen_hidden(args, **kw) -> subprocess.Popen:
    """Popen for a game instance the harness starts (servers, renders, checks, exports, autoplay tests): no window
    ever shows on your monitors. This game build ignores --no-window, so the process is created on a separate
    desktop that is never shown (yomibench.hidden_desktop). Fallback, if that desktop can't be made or
    YOMIBENCH_HIDE=window: start with SW_HIDE and hide the process's visible windows while it runs. Elsewhere:
    plain Popen."""
    if sys.platform != "win32":
        return subprocess.Popen(args, **kw)
    if os.environ.get("YOMIBENCH_HIDE", "desktop") != "window":
        from yomibench import hidden_desktop
        try:
            return hidden_desktop.popen(args, **kw)
        except OSError as e:
            print(f"warning: hidden desktop unavailable ({e}); hiding the game window instead", flush=True)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    kw.setdefault("creationflags", 0)
    kw["creationflags"] |= subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(args, startupinfo=si, **kw)
    threading.Thread(target=_hide_windows_of, args=(proc,), daemon=True).start()
    return proc


def _hide_windows_of(proc: subprocess.Popen) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]

    def hide_once() -> None:
        def cb(hwnd, _lp):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == proc.pid and user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, 0)  # SW_HIDE
            return True
        user32.EnumWindows(enum_proc(cb), 0)

    t0 = time.monotonic()
    while proc.poll() is None:
        hide_once()
        time.sleep(0.05 if time.monotonic() - t0 < 30 else 2.0)


def launch(
    run_dir: Path,
    port: int,
    max_candidates: int | None = None,
    log: Path | None = None,
    divergence_log: Path | None = None,
    host: str | None = None,
    extra_args: list[str] | None = None,
    below_normal: bool = False,
) -> subprocess.Popen[bytes]:
    """Start one headless server process (returns immediately; see `wait_ready`)."""
    if port_in_use(port, _reach(host)):
        raise RuntimeError(f"port {port} is already in use (an old server still running?)")
    args = [str(run_dir / EXE), "--no-window", "--disable-render-loop", "--audio-driver", "Dummy", f"--yomi-port={port}"]
    if max_candidates is not None:
        args.append(f"--yomi-max-candidates={max_candidates}")
    if host is not None:
        args.append(f"--yomi-host={host}")
    args += list(extra_args or [])
    if divergence_log is not None:
        args.append(f"--yomi-divergence-log={Path(divergence_log).resolve().as_posix()}")
    out = open(log, "ab") if log else subprocess.DEVNULL  # appended: keep a crashed run's log
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    if below_normal and sys.platform == "win32":
        flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
    try:
        proc = popen_hidden(args, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        _kill_with_this_process(proc)
        return proc
    finally:
        if log:
            out.close()  # type: ignore[union-attr]


def _reach(host: str | None) -> str:
    """Where to connect to a server bound to `host` ("*" and the default include loopback)."""
    return "127.0.0.1" if host in (None, "*", "0.0.0.0") else host


def wait_ready(port: int, proc: subprocess.Popen[bytes] | None = None, timeout: float = 60.0,
               host: str = "127.0.0.1") -> None:
    """Block until something accepts connections on `port` (or the process dies)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"game server on port {port} exited with code {proc.returncode} during startup")
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"game server on port {port} not ready after {timeout:.0f}s")


def connect_server(port: int, owned=None, host: str = "127.0.0.1", settle: float = 10.0):
    """An Engine on a server `wait_ready` just saw, with a hello answered. The server takes one client and refuses a
    second (`busy`, then it closes the connection); wait_ready's probe connection counts until the server notices it
    closed (its next frame, slow under load), so a connection right after it can be refused, as `busy` or as a
    reset/aborted connection. Reconnect for `settle` seconds while the server process is alive."""
    from yomibench.engine import Engine, EngineCrashed, EngineError, Tcp

    deadline = time.monotonic() + settle
    proc = getattr(owned, "proc", None)
    while True:
        engine = Engine(Tcp(host, port), owned=None)
        try:
            engine.hello()
        except (EngineError, EngineCrashed) as e:
            engine.close()
            alive = proc is None or proc.poll() is None
            if alive and time.monotonic() < deadline and (isinstance(e, EngineCrashed) or "busy" in str(e)):
                time.sleep(0.25)
                continue
            raise
        engine._owned = owned
        return engine


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        socket.create_connection((host, port), timeout=0.5).close()
        return True
    except OSError:
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GameServer:
    """One server process the caller owns: started on construction, killed on `close`."""

    def __init__(self, game_dir: str | None = None, port: int | None = None, max_candidates: int | None = None,
                 run_dir: Path | None = None, log: Path | None = None, extra_args: list[str] | None = None,
                 below_normal: bool = False):
        self.run_dir = build_run_dir(find_game_dir(game_dir), run_dir)
        self.port = port or free_port()
        self.proc = launch(self.run_dir, self.port, max_candidates, log=log, extra_args=extra_args,
                           below_normal=below_normal)
        try:
            wait_ready(self.port, self.proc)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def serve(count: int, base_port: int, game_dir: str | None, max_candidates: int | None,
          run_dir: Path | None = None, divergence_logs: bool = False, host: str | None = None,
          stagger: float = 0.5) -> None:
    """Run `count` servers on consecutive ports, restarting any that exit, until Ctrl-C.
    Logs go to <run_dir>/logs (server-PORT.log, appended across restarts, and divergence-PORT.jsonl
    if asked).

    A server that fails to start (or crashes) is retried with backoff on its own; one bad port never
    takes the others down. Every process is registered as soon as it exists, so all of them are
    cleaned up on exit. Starts are staggered (`stagger` seconds apart): native crashes cluster
    when many game processes start at the same moment."""
    rd = build_run_dir(find_game_dir(game_dir), run_dir)
    logs = rd / "logs"
    logs.mkdir(exist_ok=True)
    ports = [base_port + i for i in range(count)]
    procs: dict[int, subprocess.Popen[bytes] | None] = {port: None for port in ports}
    next_try = {port: 0.0 for port in ports}
    failures = {port: 0 for port in ports}

    def try_start(port: int) -> None:
        div = logs / f"divergence-{port}.jsonl" if divergence_logs else None
        try:
            p = launch(rd, port, max_candidates, log=logs / f"server-{port}.log", divergence_log=div, host=host)
        except Exception as e:  # e.g. the port is still held by a dying process
            print(f"port {port}: could not start ({e})", flush=True)
            failures[port] += 1
            next_try[port] = time.monotonic() + min(60.0, 2.0 ** failures[port])
            return
        procs[port] = p  # registered before waiting, so cleanup always sees it
        try:
            wait_ready(port, p, host=_reach(host))
        except Exception as e:
            print(f"port {port}: not ready ({e}); retrying", flush=True)
            if p.poll() is None:
                p.kill()
            p.wait()
            procs[port] = None
            failures[port] += 1
            next_try[port] = time.monotonic() + min(60.0, 2.0 ** failures[port])
            return
        failures[port] = 0
        print(f"ready  tcp:{_reach(host)}:{port}  (pid {p.pid})", flush=True)

    try:
        for port in ports:
            try_start(port)
            time.sleep(stagger)
        while True:
            time.sleep(1.0)
            for port, p in procs.items():
                if p is not None and p.poll() is not None:
                    print(f"server on port {port} exited ({p.returncode}); restarting", flush=True)
                    procs[port] = None
                if procs[port] is None and time.monotonic() >= next_try[port]:
                    try_start(port)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            if p is not None:
                if p.poll() is None:
                    p.kill()
                p.wait()
