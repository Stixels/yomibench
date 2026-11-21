"""The YomiBenchVsBot mod zip (tools/build_vsbot_mod.py): what the in-game "Multiplayer (vs bot)" needs from it."""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_vsbot_mod  # noqa: E402

MOD = ROOT / "mod" / "YomiBenchVsBot"


def _zip(tmp_path) -> zipfile.ZipFile:
    return zipfile.ZipFile(build_vsbot_mod.build(tmp_path))


def test_the_zip_carries_every_script_the_client_copy_runs(tmp_path):
    # VsBot.gd copies CLIENT_SCRIPTS out of res://YomiBenchVsBot/ into the headless copy's folder.
    listed = re.search(r"const CLIENT_SCRIPTS = \[([^\]]*)\]", (MOD / "VsBot.gd").read_text()).group(1)
    scripts = re.findall(r'"([^"]+)"', listed)
    assert {"MpClient.gd", "MpLink.gd", "YomiHuman.gd", "SteamStub.gd"} <= set(scripts)
    names = set(_zip(tmp_path).namelist())
    for s in scripts:
        assert f"YomiBenchVsBot/{s}" in names


def test_only_the_shared_link_script_declares_rpcs():
    # Godot 3.5 addresses an RPC by its index in the receiving script's remote methods: both games must
    # declare them in one shared script (MpLink.gd), never in VsBot.gd or MpClient.gd.
    for f in [*MOD.glob("*.gd"), ROOT / "mod" / "YomiBench" / "YomiHuman.gd"]:
        has = bool(re.search(r"^\s*(remote|remotesync|master|puppet)\s+func\b", f.read_text(), re.M))
        assert has == (f.name == "MpLink.gd"), f.name


def test_a_stuck_start_ends_with_a_reason_and_both_logs():
    vsbot = (MOD / "VsBot.gd").read_text()
    for text in ["The bot server didn't start (Windows Security may have blocked yomibench-bot-server.exe",
                 "The bot server stopped: %s", "The bot's copy of the game couldn't reach the bot server.",
                 "Protection history"]:
        assert text in vsbot, text
    assert '"bot-server.log"' in vsbot and "user://yomibench_vsbot.log" in vsbot
    # the client copy's give-up reaches the host
    assert '"bot_unreachable"' in vsbot and "func bot_unreachable" in (MOD / "MpClient.gd").read_text()
    assert "link.bot_unreachable(" in (ROOT / "mod" / "YomiBench" / "YomiHuman.gd").read_text()


def test_the_zip_is_deterministic(tmp_path):
    a = build_vsbot_mod.build(tmp_path / "a").read_bytes()
    b = build_vsbot_mod.build(tmp_path / "b").read_bytes()
    assert a == b


def test_a_mid_match_bot_crash_says_so_on_both_routes():
    human = (ROOT / "mod" / "YomiBench" / "YomiHuman.gd").read_text()
    vsbot = (MOD / "VsBot.gd").read_text()
    client = (MOD / "MpClient.gd").read_text()
    text = "The bot crashed (log: %s). Go back to the main menu and start a new match."
    assert text in human and 'server_file("bot-server.log")' in human
    # Start-menu route (single): a lost connection is the crash message, never the old bare "Lost the bot server."
    assert "Lost the bot server" not in human and "bot_crashed()" in human
    # a dead server sends nothing: the connection is read once a second even with nothing waiting
    assert "probe_at" in human and "get_partial_data(max(n, 1))" in human
    # Multiplayer (vs bot): the client reports the loss before it quits; the host also watches the client leaving
    # and the server's status heartbeat mid-match
    assert "link.bot_lost(crash_text())" in human and 'to_host("bot_crashed", text)' in client
    assert '"bot_crashed"' in vsbot and "func _check_alive" in vsbot and "STALE_PLAYING_S" in vsbot
    assert '_crashed("the bot\'s copy of the game left")' in vsbot and "human.bot_crashed(text)" in vsbot


def test_the_bottom_bar_stays_out_of_the_games_own_bottom_row():
    """The bar sat on the game's options bar (the bottom-center "^" button's ghost and speed settings) and, on the
    main menu, on the version and Mod List labels. It is split around the "^" button, each half clipped, and hidden
    while our panel is up, the options bar is open or the game's HUD is hidden (F1)."""
    human = (ROOT / "mod" / "YomiBench" / "YomiHuman.gd").read_text()
    assert "status_r = Label.new()" in human and "clip_text = true" in human
    assert "const STATUS_HALF_W = 302" in human  # 4 + 302 < 310 (the "^" button's left edge); 640 - 4 - 302 > 330
    room = human.split("func _status_room():", 1)[1].split("\nfunc ", 1)[0]
    assert '"%OptionsBar"' in room and "panel.visible" in room and "ui.visible" in room
    upd = human.split("func _update_status():", 1)[1].split("\nfunc ", 1)[0]
    assert upd.index("_status_room()") < upd.index("if crashed")  # also hidden after a crash, under the panel
    # the estimate and depth go to the right half, never onto the left one
    assert "bot's estimate" not in "".join(line for line in human.splitlines() if "s +=" in line)


def test_the_panel_waits_for_the_bot_and_names_it_as_the_guides_do():
    human = (ROOT / "mod" / "YomiBench" / "YomiHuman.gd").read_text()
    # greyed until the hello (which would reset anything picked while connecting), and after a crash
    assert human.count("_set_controls(false)") == 2 and human.count("_set_controls(true)") == 1
    hello = human.split("func _on_hello(r):", 1)[1].split("\nfunc ", 1)[0]
    assert hello.index("_set_controls(true)") < hello.index("settings_opt.select(")
    assert 'return "HustleBot g" + b.substr(4)' in human
    assert human.count('server.get("bot"') == 1  # bot_name() itself: every player-facing name goes through it
    # the game's post-game Rematch/Quit stay hidden after the result too (phase idle, the match still on screen)
    # (the post-game replay is a new main.game: the check is on the game on screen, not on our match's)
    assert 'phase == "ended" or (phase == "idle" and is_instance_valid(main) and is_instance_valid(main.game))' in human
    # nobody to chat with: the game's chat box is closed when a bot match starts (host side)
    assert 'main.get_node_or_null("%ChatWindow")' in human
