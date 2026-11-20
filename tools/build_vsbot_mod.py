"""Build the YomiBench "Multiplayer (vs bot)" mod zip for the game's own mod loader (docs/PLAYING.md,
"In the game's menu"):

    python tools/build_vsbot_mod.py [--out DIR]       # -> DIR/YomiBenchVsBot.zip (default dist/)
    python tools/build_vsbot_mod.py --bundle-server C:/path/to/dist/server
                                                      # one step: the frozen bot server inside the mod

The zip holds mod/YomiBenchVsBot/* plus mod/YomiBench/YomiHuman.gd (one source for every way of playing) and
SteamStub.gd (for the headless copy of the game that plays the bot's side),
all under YomiBenchVsBot/ so the loader maps them to res://YomiBenchVsBot/. Never any game file.

With --bundle-server, the frozen server (packaging/windows, PyInstaller onedir, minus the Start-menu
launcher) goes under YomiBenchVsBot/server/ with server/manifest.json: every file's sha256 and size, and a
version that is the hash of that list. On first use the mod copies the files to user://, checks every
sha256, and runs the server from there (VsBot.gd); a new version is copied afresh and older ones removed.

Deterministic: fixed timestamps and order, so the same sources give the same zip (the game hashes mod zips).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod" / "YomiBenchVsBot"
FILES = [MOD / "_metadata", MOD / "ModMain.gd", MOD / "VsBotUILayer.gd", MOD / "VsBot.gd", MOD / "MpClient.gd",
         MOD / "MpLink.gd", ROOT / "mod" / "YomiBench" / "SteamStub.gd",
         ROOT / "mod" / "YomiBench" / "YomiHuman.gd"]
SKIP = {"yomibench-play.exe"}  # the Start-menu launcher; the mod needs the server only
EXE = "yomibench-bot-server.exe"


def _entry(z: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def server_manifest(server: Path) -> dict:
    """Every file of the frozen server with its sha256 and size; `version` hashes the list, so any change
    to any file is a new version."""
    files = []
    for p in sorted(x for x in server.rglob("*") if x.is_file() and x.name not in SKIP):
        files.append({"path": p.relative_to(server).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "size": p.stat().st_size})
    version = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    return {"version": version, "exe": EXE, "files": files}


def build(out_dir: Path, server: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "YomiBenchVsBot.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        from yomibench import plugins

        overlays = plugins.mod_overlays("YomiBenchVsBot")  # files a plugin replaces (none publicly)
        for f in FILES:
            src = overlays.get(f.name, f) if f.parent == MOD else f
            _entry(z, f"YomiBenchVsBot/{f.name}", src.read_bytes().replace(b"\r\n", b"\n"))
        if server is not None:
            man = server_manifest(server)
            if not any(f["path"] == EXE for f in man["files"]):
                raise SystemExit(f"no {EXE} in {server}")
            for f in man["files"]:
                _entry(z, f"YomiBenchVsBot/server/{f['path']}", (server / f["path"]).read_bytes())
            _entry(z, "YomiBenchVsBot/server/manifest.json", json.dumps(man, indent=1).encode())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--bundle-server", type=Path, help="the frozen server folder (packaging/windows build) to put inside")
    a = ap.parse_args(argv)
    out = build(a.out, a.bundle_server)
    print(out, out.stat().st_size, "bytes, sha256", hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
