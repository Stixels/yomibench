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


def _func(src: str, head: str) -> str:
    """The body of the GDScript function that starts with `head` (up to the next top-level func)."""
    return src.split(head, 1)[1].split("\nfunc ", 1)[0].split("\nstatic func ", 1)[0]


HUMAN = ROOT / "mod" / "YomiBench" / "YomiHuman.gd"


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
    # v5: the action first, then the log as the README names it (short, no user name on screen), and a button
    text = "The bot stopped. Go back to the main menu to start again."
    assert text in human and "%USERPROFILE%" in _func(human, "static func crash_text():")
    assert 'log_btn.text = "Open log folder"' in human and "OS.shell_open(" in human
    crash = _func(human, "func bot_crashed(")
    assert "log_btn.show()" in crash and "_ignore_mouse(p1_panel)" in crash and 'side == "host"' in crash
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
    # the estimate and the bot's status go to the right half, never onto the left one
    assert "est. " not in "".join(line for line in human.splitlines() if "s +=" in line)
    assert "card.visible" in room  # and under the result card


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


# --- private build v5 (the UX proposal and the untimed after-lock cap) ------------------------------------------

def test_lock_in_reaches_the_server_in_every_match_and_carries_only_the_turn():
    """The untimed after-lock cap counts from the lock-in, so `locked_in` goes out whatever "Bot uses your thinking
    time" says (v4 sent it only with that box ticked). Every one carries exactly its type and the turn."""
    human = HUMAN.read_text()
    sends = [line.strip() for line in human.splitlines() if '"locked_in"' in line and "_notify(" in line]
    assert len(sends) == 3
    for line in sends:
        assert line.endswith('_notify({"type": "locked_in", "turn": pending_turn})') or \
            line.startswith('_notify({"type": "locked_in", "turn": pending_turn})'), line
    for head in ("func _on_net_turn_ready(", "func _on_input(", "func _decision("):
        body = _func(human, head)
        assert "locked_in" in body and "think_on and" not in body, head


def test_the_bot_status_on_the_players_side_never_shows_its_move():
    """Item 2: thinking / locked in / finishing... N s, from the game's own turn-ready flags only (what online play
    shows as an opponent's Ready) and the lock-in time; nothing of the bot's pick or the remote status."""
    human = HUMAN.read_text()
    body = _func(human, "func bot_status():")
    for text in ('"bot: locked in"', '"bot: thinking..."', '"bot: finishing... %d s"', "Network.turns_ready",
                 "lock_cap_s", "p1_lock_msec", 'timer_mode) == "none"'):
        assert text in body, text
    for leak in ("queued_action", "action_inputs", "ghost", "remote_status", "pick"):
        assert leak not in body, leak
    assert "bot_status()" in _func(human, "func _update_host_status():")
    # the cap's length comes from the server's hello
    assert 'lock_cap_s = float(r.lock_cap_s)' in _func(human, "func _on_hello(r):")


def test_a_forfeit_ends_in_a_result():
    """Item 3: the game's forfeit ends the match without game_ended and starts its post-game replay (a new
    main.game), which v4 read as the match being closed. The forfeit is checked first, every frame."""
    human = HUMAN.read_text()
    frame = _func(human, "func _play_frame():")
    assert frame.index("_forfeit_by()") < frame.index('_abort("the match was closed")')
    by = _func(human, "func _forfeit_by():")
    assert "Network.forfeiter" in by and "game.forfeit_player" in by
    ff = _func(human, "func _on_forfeit(who):")
    assert "_send_end(2 - who, who - 1)" in ff  # the other player wins; `forfeit` is the forfeiter's seat
    assert 'msg["forfeit"] = forfeit_seat' in _func(human, "func _send_end(")
    assert '"Game over. You forfeited: %s wins." % bot_name("the bot")' in human
    # a new match clears the game's forfeit state, as its own rematch does
    assert "Network.forfeiter = 0" in _func(human, "func _launch(")


def test_the_result_card_waits_for_the_final_hit_and_offers_three_ways_on():
    """Item 5: a small card 1.5 s after the match ends, with Rematch (the last match again), Change settings and
    Main menu; Rematch waits for the bot's record of the game."""
    human = HUMAN.read_text()
    assert "const CARD_DELAY_MS = 1500" in human
    card = _func(human, "func _make_card(theme):")
    for b in ('"Rematch"', '"Change settings"', '"Main menu"'):
        assert b in card, b
    assert "if hosted:" in card  # Main menu only from the game's menu (the Start-menu app has no menu to go to)
    assert "last_msg.duplicate(true)" in _func(human, "func _on_rematch_pressed():")
    assert "not result_in" in _func(human, "func _on_rematch_pressed():")
    assert "ended_msec = OS.get_ticks_msec()" in _func(human, "func _on_host_game_ended(")
    assert "Saved to Documents" in _func(human, "func _show_result(r):")


def test_plain_words_on_the_panel():
    """Item 6: the renames (the untimed meaning of the think box changed with the cap)."""
    human = HUMAN.read_text()
    for new in ('"Bot uses your thinking time"', '"Turn timer (ranked clock)"', '"Bot style:"',
                '"unpredictable (mixes its moves; recommended)"', '"strongest move only"', '"You vs %s (local)"'):
        assert new in human, new
    for old in ("Bot thinks until you lock in", "the game's online timer; ranked: on", "samples its strategy)\"",
                "over this PC only", '"bot thought: %s"'):
        assert old not in human, old


def test_our_panel_steps_aside_for_the_pause_menu_and_the_chat_stays_closed():
    """Items 7 and 13: the panel and card hide while the game's pause menu is open; the chat box is closed once per
    game, the post-game replay (a new main.game) too, so Enter still opens it."""
    human = HUMAN.read_text()
    pause = _func(human, "func _pause_menu_frame():")
    assert '"%PausePanel"' in pause and "pause_hid" in pause and "[panel, card]" in pause
    assert "_pause_menu_frame()" in _func(human, "func _process(_delta):")
    proc = _func(human, "func _process(_delta):")
    assert "main.game != chat_game" in proc and "chat.hide()" in proc


def test_the_logs_are_named_as_the_readme_names_them():
    vsbot = (MOD / "VsBot.gd").read_text()
    assert "static func _short_path(p):" in vsbot and '["APPDATA", "USERPROFILE"]' in vsbot
    assert "_short_path(HUMAN.server_file(\"bot-server.log\"))" in vsbot


def test_the_bottom_bar_names_the_net_by_its_short_sha256():
    """The hello's `model` (generation, short sha256, source) reaches the bottom bar on both sides: "(model 65888397)",
    plus "from your model folder" when the app plays a newer net from the player's model folder."""
    from types import SimpleNamespace

    from yomibench.human_play import Session

    human = HUMAN.read_text()
    body = _func(human, "func model_text():")
    assert 'server.get("model")' in body and '" (model %s" % str(m.sha)' in body
    assert '"model folder"' in body and "from your model folder" in body
    for head in ("func _update_status():", "func _update_host_status():"):
        assert '[bot_name("?"), model_text()]' in _func(human, head), head
    sha = "65888397a180e46e57ed1d615e90abf63844794318161b8c3169c7296ee65a2f"
    trimmed = SimpleNamespace(model={"generation": 15, "sha256": "79f0" + "0" * 60, "hub_sha256": sha, "source": "bundled"})
    assert Session.hello_model(trimmed) == {"generation": 15, "sha": "65888397", "source": "bundled"}
    plain = SimpleNamespace(model={"generation": 16, "sha256": sha, "source": "model folder"})
    assert Session.hello_model(plain) == {"generation": 16, "sha": "65888397", "source": "model folder"}
    assert Session.hello_model(SimpleNamespace(model=None)) is None
    # the result card names it too ("HustleBot g15 wins."), through bot_name() like the panel
    assert '"%s wins." % bot_name("The bot")' in _func(human, "func _bot_wins():")
    assert human.count("_bot_wins()") == 4 and "The bot wins." not in human
    # and the HUD's name tag over P2 (the game's player list, which the bot's copy joined as "YomiBench bot")
    body = _func(human, "func _name_the_bot():")
    assert "Network.players[id] = bot_name(BOT_JOIN_NAME)" in body and "== BOT_JOIN_NAME" in body
    client = (MOD / "MpClient.gd").read_text()
    assert 'const BOT_JOIN_NAME = "YomiBench bot"' in human and '"YomiBench bot")' in client  # the same name both ends
    assert "_name_the_bot()" in _func(human, "func _on_hello(r):")


def test_qol_prefs_random_swap_tooltips_and_replay_hide():
    human = HUMAN.read_text()
    vsbot = (MOD / "VsBot.gd").read_text()
    # Prefs saved on Start and applied on hello (only when not autoplay)
    assert 'const PREFS = "user://yomibench_prefs.json"' in human
    assert "_save_prefs()" in _func(human, "func _on_start_pressed():")
    assert "_apply_prefs()" in _func(human, "func _on_hello(r):")
    # Random character + Swap button
    assert 'const RANDOM_CHAR = "Random"' in human
    assert "opt.add_item(RANDOM_CHAR)" in _func(human, "func _fill_characters(names):")
    assert "_resolve_char(p1_opt)" in _func(human, "func _on_start_pressed():")
    assert 'swap_btn.text = "Swap"' in _func(human, "func _make_overlay():")
    assert "swap_btn" in _func(human, "func _set_controls(on):")
    # Clear win-estimate label on both sides and dynamic timer checkbox label
    assert '"bot win est. %d%%"' in _func(human, "func _update_status():")
    assert '"bot win est. %d%%"' in _func(human, "func _update_host_status():")
    assert '"Turn timer (per-turn)"' in _func(human, "func _update_timer_label():")
    assert "_update_timer_label()" in _func(human, "func _on_preset_selected(i):")
    # Tooltips on panel options
    overlay = _func(human, "func _make_overlay():")
    for ctrl in ("p1_opt", "p2_opt", "swap_btn", "settings_opt", "pick_opt", "estimate_box", "think_box", "timer_box"):
        assert f"{ctrl}.hint_tooltip =" in overlay, ctrl
    # Session W-L-D tally on result card and panel title
    assert "session_wins" in _func(human, "func _show_result(r):")
    assert "_session_score_text()" in _func(human, "func _panel_title():")
    assert "_session_score_text()" in _func(human, "func _show_result(r):")
    # Watch replay hide/reveal + Open games folder + keyboard focus
    card = _func(human, "func _make_card(theme):")
    assert '"Watch replay (H)"' in card and '"Open games folder"' in card and '"Show menu (H)"' in card
    assert "KEY_H" in _func(human, "func _input(event):")
    assert "reveal_btn" in _func(human, "func _pause_menu_frame():")
    assert "reveal_btn.visible" in _func(human, "func _status_room():")
    assert "start_btn.grab_focus()" in _func(human, "func _show_panel(t, i):")
    assert "rematch_btn.grab_focus()" in _func(human, "func _show_card():")
    # VsBot.gd: live extraction progress + log rotation
    assert "Extracting the bot server (%d/%d)..." in _func(vsbot, "func _bundled_server():")
    assert "Preparing the bot's copy of the game (%s)..." in _func(vsbot, "func _client_run_dir():")
    assert "const LOG_MAX_BYTES = 1048576" in vsbot and 'LOG_PATH + ".old"' in _func(vsbot, "func _log(text):")


def test_match_start_payload_and_host_crash_diagnostics():
    human = HUMAN.read_text()
    vsbot = (MOD / "VsBot.gd").read_text()
    launch = _func(human, "func _launch(msg):")
    # Match vanilla CharacterSelect.get_match_data() so mods reading selected_styles don't trip on a missing key
    assert 'data["selected_styles"] = {1: null, 2: null}' in launch
    # Host match-start breadcrumbs: distinguish crashes before/during _on_match_ready vs setup_game_deferred vs tick 0
    assert 'link.trace("starting match:' in launch
    assert 'link.trace("match_ready returned")' in launch
    assert launch.index('link.trace("starting match:') < launch.index("main._on_match_ready(data)")
    assert launch.index("main._on_match_ready(data)") < launch.index('link.trace("match_ready returned")')
    assert 'link.trace("match attached (tick %d)" % game.current_tick)' in _func(human, "func _attach():")
    # Session header records GPU adapter/driver and active mod list so yomibench_vsbot.log is self-contained
    vlaunch = _func(vsbot, "func launch():")
    assert "_active_mod_names()" in vlaunch and "VisualServer.get_video_adapter_name()" in vlaunch
    assert "func _active_mod_names():" in vsbot
