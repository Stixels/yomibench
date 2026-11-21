extends Node
# YomiBench human-vs-bot play (docs/HUMAN_PLAY.md): a person plays P1 in the game's own UI, a
# yomibench bot plays P2. Loaded as an autoload through override.cfg (tools/human_play.py builds the
# run dir) and active only when the game is started with --yomi-human=PORT; otherwise it does
# nothing. It never touches YomiServer or the engine protocol: it is a TCP client of the bot server
# (src/yomibench/human_play.py) on 127.0.0.1:PORT, one JSON line per message, replies in order.
#
# - Matches are started here, not from the character select: the standard online settings (as
#   YomiServer._standard_settings) with the panel's settings preset applied (the presets come from
#   the server's hello: standard, ranked), the characters picked on this script's panel, a random
#   seed. The match runs through Main._on_match_ready like a singleplayer match from the character
#   select, and match_start reports exactly the settings it was given.
# - At every decision (both turn branches opened, game paused) the mod sends the inputs both
#   fighters were given last turn and a compact snapshot of the real state, and gets the bot's pick
#   for P2, which it submits with P2's own ActionButtons.on_action_submitted: the call the panel's
#   Select button makes, so the game treats it as a normal selection. P2's panel is shown as an
#   online opponent's is, greyed and clickable (Select and keyboard lock-in do nothing): its clicks
#   only set P2's ghost, which starts on Continue, and never reach the bot's move; the bot's pick
#   never touches the panel or the ghost. Both Undo buttons are hidden (an undo would rewind the real match only), as are
#   the in-match replay keys (see _input).
# - Inputs are read from each fighter right after on_action_selected (its queued action, data and
#   extra, as the replay records them), so P1's are exactly what the UI sent.
# - At the end it sends the final snapshot and the winner; the server writes the result.
# - Lock-in: when P1 locks in (P1's action_selected) while the bot's pick is still pending, the mod sends a
#   one-way {"type": "locked_in", "turn": N} (no reply, no callback queued): THAT P1 locked in, never what. P1's
#   input still reaches the server only with the next decision. "Bot uses your thinking time" (panel) has the bot
#   deepen its search until then; with no turn timer the bot answers at most `lock_cap_s` (the server's hello,
#   10 s) after it (the after-lock cap), so it is sent in every match. The bottom bar shows the bot's status:
#   thinking, locked in, or (untimed, after you lock in) "finishing... N s", from the game's own turn-ready flags
#   (the same "the other side locked in" online play shows), never the bot's move.
# - A forfeit (the game's pause menu): the game ends the match without its game_ended signal; the mod sees the
#   game's forfeit flag, shows the result and sends `end` with the winner and `forfeit` (the forfeiter: 0 = P1).
# - "Turn timer" (panel; default on for a preset with a timer, i.e. ranked, as the game plays it, off for
#   standard): the game's own turn timer, from the match's settings (timer_mode, turn_time, turn_min_length,
#   increment_*). Online ("host"/"client") the game runs it itself. Singleplayer the game never starts it, so
#   "single" runs the game's own P1/P2 turn timers here the way the game runs them online (UILayer's
#   on_player_actionable and _on_turn_timer_timeout): the same lengths, the same top-ups, and on expiry the
#   same outcome: P1's panel locks in what it holds (ActionButtons.timeout); the bot, if its answer is late,
#   holds (Continue). The bot answers before the timer: the server's deadline, plus a backstop here that
#   reads the bot's real clock (the game's P2 timer) and sends a one-way `deadline` 1.5 s before it ends.
#   With the timer on, "Bot uses your thinking time" defaults on too: the bot uses its clock, deepening its
#   search until you lock in or its deadline (whichever is first), then locks in.
#   Off: timer_mode "none" (the standard settings' clock).
#
# Options: --yomi-human=PORT (required), --yomi-human-host=ADDR (default 127.0.0.1; tests in Docker),
# --yomi-human-p1=CHAR / --yomi-human-p2=CHAR (the panel's defaults), --yomi-human-autoplay (headless
# test: P1 plays random legal moves through its own action panel, games come from the server),
# --yomi-human-speed=K (physics ticks per second x K; autoplay only), --yomi-human-autoplay-wait=S
# (autoplay P1 waits S seconds before Select: a turn timer test).
#
# Sides. "single" (above): one game drives both fighters. "Multiplayer (vs bot)" (the YomiBenchVsBot mod)
# splits it over two copies of the game joined by the game's own direct-connect multiplayer on 127.0.0.1:
# - "host", the player's game: the panel, the match start and P1. It never talks to the bot server and
#   never learns the bot's pick before the game's own protocol delivers it (after both locked in).
# - "client", a headless copy (MpClient.gd): P2 and the bot server connection. It submits each pick with its
#   own P2 panel (Network.submit_action, then the game's send_action once both are ready), as a player's
#   game does, and relays to the host only what the host's panel shows (the hello, refusals, last turn's
#   estimate, results). P1's lock-in reaches it as the game's own "turn ready" message (no move in it).
# `link` is the node that carries messages between the two (VsBot.gd / MpClient.gd).

const PROTOCOL = 1
const VERSION = "10"  # 2: characters from the server's hello (all five), pick style, win estimate; 3: settings presets; 4: think until lock-in; 5: deadline in every mode; 6: multiplayer sides; 7: turn timer toggle, the game's timer in single; 8: the bot-crashed message; 9: locked_in in every match (the untimed after-lock cap), the bot's status, forfeit results, the result card; 10: the net's short sha256 in the bottom bar
# The characters the panel offers until the server's hello names the ones its sim plays.
var characters = ["Ninja", "Cowboy", "Wizard", "Mutant", "Robot"]

var port = 0
var host = "127.0.0.1"
var autoplay = false
var peer = null
var connected = false
var retry_at = 0
var net_since = 0        # client: when it first tried the bot server (0: not yet)
var listening_since = 0  # client: when the server's status file first said "listening"
var gave_up = false
var wait_title = ""      # host: what the panel says while the bot isn't up yet (VsBot.gd sets it)
var wait_text = ""
var inbuf = ""
var callbacks = []       # reply handlers, in request order
var server = {}          # the server's hello
var settings = {}

var main = null
var game = null
var p1_panel = null
var p2_panel = null
var p2_extra = null      # single: P2's panel's untouched extra at the decision (a pick without extra)
var phase = "boot"       # boot, connecting, idle, starting, playing, ended
var match_ = {}
var turn = 0
var decision_tick = -1000
var inputs_at = {}       # tick of the decision -> [P1 input, P2 input]
var pick = null          # the bot's pick for this decision, until submitted
var pick_pending = false
var desynced = false
var p2_fallback = false
var auto_tick = -1000
var games_played = 0
var rng = RandomNumberGenerator.new()

var layer = null
var status = null        # the bottom bar's left half (who, which turn, sync)
var status_r = null      # its right half (the bot's estimate and depth)
const STATUS_HALF_W = 302  # each half's width: they meet short of the game's "^" button in the middle
var panel = null
var title = null
var info = null
var p1_opt = null
var p2_opt = null
var pick_opt = null
var settings_opt = null
var presets = {"standard": {}}  # the server's settings presets (name -> overrides of the standard settings)
var preset_names = ["standard"]
var estimate_box = null
var start_btn = null
var pick_styles = ["mixed", "strongest"]
var estimate = null      # the bot's win estimate from this decision's pick, shown once the turn is played
var shown_estimate = null
# Think until lock-in: the bot deepens its search until P1 locks in. The server only ever hears
# THAT P1 locked in (one-way `locked_in`, type and turn only), never what P1 picked.
var think_box = null
var think_on = false
var pending_turn = -1    # the decision whose pick is pending
var decision_msec = 0
var deadline_sent = false
var depth_text = ""
# The turn timer (see the top). The game's own clock settings: the server and the game read the same keys.
const TIMER_KEYS = ["timer_mode", "turn_time", "turn_min_length", "increment_starting_time", "increment_per_turn",
	"increment_max_time"]
const BACKSTOP_S = 1.5   # the bot's answer is asked for this long before its clock (the game's P2 timer) ends
var timer_box = null
var lobby_timer = {}     # the game's lobby timer settings (the settings panel's defaults), before standard turns it off
var single_clock = false # single: this match runs the game's turn timers here (the game runs them only online)
var submitted_for = -1   # the decision whose pick this game submitted for P2 (another: P2's clock ran out first)
var p2_timeouts = 0      # this match: turns P2 was locked in by its clock running out, not by the bot's answer


signal leave_requested  # hosted: the player pressed "Back to main menu"
signal quitting         # hosted: autoplay is about to quit the game (the host stops its server first)

var hosted = false       # started by the YomiBenchVsBot menu mod (not by --yomi-human): adds "Back to main menu"
var side = "single"      # single, host or client (see the top)
var link = null          # host and client: the VsBot.gd / MpClient.gd node between the two games
var remote_status = {}   # host: what the client last reported (turn, desynced, last turn's estimate, depth)
var p1_locked_tick = -1000  # client: the decision tick P1 locked in at (the game's "turn ready", no move)
var p1_lock_msec = 0     # host: when P1 locked in this turn
var autoplay_wait = 0.0
var crashed = false      # the bot server went away (a crash): the panel says so until the bot is back
var probe_at = 0         # next read that checks the server connection is still there (a dead peer sends nothing)
# What the panel says when the bot server goes away mid-match (or at any time once it was up).
const CRASH_TEXT = "The bot stopped. Go back to the main menu to start again.\n(Its log: %s)"
var lock_cap_s = 10.0    # untimed: the bot answers within this long after P1 locks in (the server's hello)
var last_msg = {}        # the last match's start message: Rematch plays the same settings and characters again
var card = null          # the result card (Rematch / Change settings / Main menu)
var card_title = null
var card_info = null
var rematch_btn = null
var result_in = false    # the bot's record of the finished match arrived (Rematch waits for it)
var ended_msec = 0       # when the match ended: the card waits a moment so the final hit stays visible
const CARD_DELAY_MS = 1500
var forfeiter = 0        # the match ended on a forfeit by this player (1 or 2), else 0
var pause_hid = []       # our controls hidden while the game's pause menu is open
var chat_game = null     # the game whose chat box was closed (the post-game replay is a new one)
var log_btn = null       # crash: opens the bot's log folder


func _ready():
	var p = _option("human", "")
	if p == "":
		set_process(false)
		set_process_input(false)
		return
	start(int(p), _option("human-host", host), "--yomi-human-autoplay" in OS.get_cmdline_args())


# Connect to the bot server on host:port and show the panel. --yomi-human=PORT calls this at start-up.
# Multiplayer (vs bot): VsBot.gd starts the host side (no server connection: port unused) and MpClient.gd
# the client side (the server connection, no panel), each with itself as `link_`.
func start(port_, host_ = "127.0.0.1", autoplay_ = false, hosted_ = false, side_ = "single", link_ = null):
	port = int(port_)
	host = host_
	autoplay = autoplay_
	hosted = hosted_
	side = side_
	link = link_
	autoplay_wait = float(_option("human-autoplay-wait", "0"))
	pause_mode = PAUSE_MODE_PROCESS
	rng.randomize()
	set_process(false)
	set_process_input(side != "client")
	call_deferred("_boot")


func _option(name, default):
	var flag = "--yomi-%s=" % name
	for a in OS.get_cmdline_args():
		if a.begins_with(flag):
			return a.substr(flag.length())
	return default


func _boot():
	# The game's own main scene stays (unlike YomiServer): wait for it and its character loading.
	while true:
		yield(get_tree(), "idle_frame")
		var s = get_tree().current_scene
		if s != null and s.has_method("_on_match_ready") and Global.mods_loaded:
			break
	settings = _standard_settings()
	var speed = float(_option("human-speed", "1"))
	if autoplay and speed > 1 and side == "single":
		Engine.iterations_per_second = int(60 * speed)
	if side != "client":
		_make_overlay()
	else:
		Network.connect("player_turn_ready", self, "_on_net_turn_ready")
	phase = "connecting"
	set_process(true)


# The same settings YomiServer plays (the settings panel's scene defaults, Sadness on as online
# play forces, no turn timer, no dummy): the settings the bots were trained under.
func _standard_settings():
	var scene = load("res://ui/GameSettingsPanelContainer.tscn")
	if scene == null:
		return {}
	var p = scene.instance()
	add_child(p)
	var s = {}
	if typeof(p.get("scene_defaults")) == TYPE_DICTIONARY:
		s = p.scene_defaults.duplicate(true)
	p.queue_free()
	if s.empty():
		return s
	lobby_timer = {}
	for k in TIMER_KEYS:
		if s.has(k):
			lobby_timer[k] = s[k]
	s["sadness_enabled"] = true
	s["timer_mode"] = "none"
	s["p2_dummy"] = false
	return s


# --- connection --------------------------------------------------------------------------------

func _process(_delta):
	if side != "host":
		_net()
	elif phase == "playing" or phase == "ended" or (phase == "idle" and is_instance_valid(main) and is_instance_valid(main.game)):
		# The game's own Rematch/Quit (and "replay autosaved") under our result panel: this panel starts the next
		# match instead. Also once the result is in (phase idle): a multiplayer match ends in the game's post-game
		# replay (a new main.game; UILayer._on_game_playback_requested shows these), and they would come back beside
		# "Start game", where the game's Rematch would ask the bot's copy for a match we didn't start.
		var post = main.get_node_or_null("%PostGameButtons") if is_instance_valid(main) else null
		if post and post.visible:
			post.hide()
		# The game opens its chat box for every multiplayer game, the post-game replay (a new main.game) too: closed
		# once per game, so the chat key (Enter) still opens it.
		if is_instance_valid(main) and is_instance_valid(main.game) and main.game != chat_game:
			chat_game = main.game
			var chat = main.get_node_or_null("%ChatWindow")
			if chat:
				chat.hide()
	_pause_menu_frame()
	if ended_msec > 0 and OS.get_ticks_msec() - ended_msec >= CARD_DELAY_MS:
		ended_msec = 0
		_show_card()
	match phase:
		"starting":
			_attach()
		"playing":
			_play_frame()
	_update_status()


# The bot server's progress file (packaging/windows/yomibench_app.py write_status): {"phase", "detail", "pid",
# "time"} in %USERPROFILE%\.yomibench\bot-server.status, or {} if there is none (yet) or it can't be read.
static func server_status():
	var f = File.new()
	var path = server_file("bot-server.status")
	if path == "" or not f.file_exists(path) or f.open(path, File.READ) != OK:
		return {}
	var r = JSON.parse(f.get_as_text())
	f.close()
	if r.error != OK or typeof(r.result) != TYPE_DICTIONARY or not r.result.has("phase"):
		return {}
	return r.result


# A file in the bot server's folder in the user's profile (its log and status), "" if the profile isn't known.
static func server_file(name):
	var home = OS.get_environment("USERPROFILE")
	if home == "":
		home = OS.get_environment("HOME")
	return "" if home == "" else home.replace("\\", "/").plus_file(".yomibench").plus_file(name)


# Client: the bot server never answered. Gives up 90 s after the server said it was listening (its first run can
# spend a minute or more building its game data first), or after 7 minutes in any case, and tells the host.
const CLIENT_AFTER_LISTEN_MS = 90000
const CLIENT_MAX_WAIT_MS = 420000


func _client_deadline():
	var now = OS.get_ticks_msec()
	if net_since == 0:
		net_since = now
	if listening_since == 0 and str(server_status().get("phase", "")) == "listening":
		listening_since = now
	if (listening_since > 0 and now - listening_since > CLIENT_AFTER_LISTEN_MS) or now - net_since > CLIENT_MAX_WAIT_MS:
		gave_up = true
		link.bot_unreachable("The bot's copy of the game couldn't reach the bot server on 127.0.0.1:%d (it tried for %d s)." % [
			port, (now - net_since) / 1000])
		return true
	return false


func _net():
	if gave_up:
		return
	if peer == null or (not connected and peer.get_status() in [StreamPeerTCP.STATUS_NONE, StreamPeerTCP.STATUS_ERROR]):
		if OS.get_ticks_msec() < retry_at:
			return
		retry_at = OS.get_ticks_msec() + 1000
		if side == "client" and server.empty() and _client_deadline():
			return
		peer = StreamPeerTCP.new()
		peer.connect_to_host(host, port)
		return
	var st = peer.get_status()
	if st == StreamPeerTCP.STATUS_CONNECTED and not connected:
		connected = true
		peer.set_no_delay(true)
		_send({"type": "hello", "protocol": PROTOCOL, "mod": VERSION}, "_on_hello")
	elif st != StreamPeerTCP.STATUS_CONNECTED and connected:
		connected = false
		callbacks.clear()
		inbuf = ""
		if side == "client":
			link.bot_lost(crash_text())  # the server went away: the host shows why, and this copy quits
			return
		bot_crashed()
		return
	if not connected:
		return
	# A server that died sends nothing: once a second read even with nothing waiting, so a closed or reset
	# connection is noticed (the read then fails and the status above turns to not connected).
	var n = peer.get_available_bytes()
	if n > 0 or OS.get_ticks_msec() >= probe_at:
		probe_at = OS.get_ticks_msec() + 1000
		var r = peer.get_partial_data(max(n, 1))
		if r[0] == OK and r[1].size() > 0:
			inbuf += r[1].get_string_from_utf8()
	var i = inbuf.find("\n")
	while i >= 0:
		var line = inbuf.substr(0, i)
		inbuf = inbuf.substr(i + 1)
		var res = JSON.parse(line)
		var cb = callbacks.pop_front() if not callbacks.empty() else ""
		var reply = res.result if res.error == OK and typeof(res.result) == TYPE_DICTIONARY else {"type": "error", "message": "bad reply"}
		if cb != "":
			call(cb, reply)
		i = inbuf.find("\n")


func _send(msg, cb = ""):
	if not connected:
		return false
	peer.put_data((JSON.print(msg) + "\n").to_utf8())
	callbacks.append(cb if cb != "" else "_on_ignored")
	return true


func _on_ignored(_reply):
	pass


# One-way message: the server sends no reply, so no callback is queued and replies stay in order.
func _notify(msg):
	if connected:
		peer.put_data((JSON.print(msg) + "\n").to_utf8())


func _on_hello(r):
	if r.get("type") != "hello":
		_say("The bot server refused: %s" % str(r.get("message")))
		return
	server = r
	if side == "host":
		_name_the_bot()
	var was_crashed = crashed
	crashed = false
	if side == "client":  # the panel is the host's
		phase = "idle"
		link.to_host("hello", r)
		return
	phase = "idle"
	_set_controls(true)
	if typeof(r.get("lock_cap_s")) in [TYPE_INT, TYPE_REAL]:
		lock_cap_s = float(r.lock_cap_s)
	if log_btn:
		log_btn.hide()
	if typeof(r.get("characters")) == TYPE_ARRAY and not r.characters.empty():
		_fill_characters(r.characters)
	var opts = r.get("options", {})
	if typeof(r.get("settings_presets")) == TYPE_DICTIONARY and not r.settings_presets.empty():
		presets = r.settings_presets
		preset_names = presets.keys()
		settings_opt.clear()
		for n in preset_names:
			settings_opt.add_item(str(n))
	if typeof(opts) == TYPE_DICTIONARY:
		settings_opt.select(max(0, preset_names.find(str(opts.get("settings", "standard")))))
		pick_opt.select(max(0, pick_styles.find(str(opts.get("pick", "mixed")))))
		estimate_box.pressed = bool(opts.get("show_estimate", false))
		timer_box.pressed = _timer_default(str(preset_names[max(0, settings_opt.selected)]))
		# with the turn timer on the bot uses its clock: it deepens until you lock in or its deadline
		think_box.pressed = bool(opts.get("think_until_lockin", false)) or timer_box.pressed
	if autoplay:
		_next_or_quit(r.get("next"))
	elif was_crashed:  # the launcher restarted the bot after a crash
		_show_panel(_panel_title(), "The bot crashed during the last match and has restarted. Go back to the main " +
			"menu, then press Start game for a new match.")
	else:
		_show_panel(_panel_title(), "")


func _panel_title():
	if side == "host":
		return "You vs %s (local)" % bot_name()
	return "You vs %s" % bot_name()


# The bot's name for the player: a promoted generation's label ("r3-g14", as the records keep it) as the site and
# the guides name it ("HustleBot g14"); any other label as it is.
func bot_name(default = "the bot"):
	var b = str(server.get("bot", ""))
	if b == "":
		return default
	if b.begins_with("r3-g") and b.substr(4).is_valid_integer():
		return "HustleBot g" + b.substr(4)
	return b


# Host: the game names the bot's seat (the HUD's tag over P2, the replay's user data) by the username the bot's copy
# of the game joined with ("YomiBench bot", before it knew which bot it plays). Once the hello names the bot, that
# entry of the game's player list becomes the bot's name ("HustleBot g15"), as the panel and the bottom bar say.
func _name_the_bot():
	var me = get_tree().get_network_unique_id() if get_tree().network_peer != null else 1
	for id in Network.players.keys():
		if id != me:
			Network.players[id] = bot_name("YomiBench bot")

# The bottom bar's note on the net (the hello's `model`): " (model 65888397)", its short sha256, plus "from your model
# folder" when the app plays a newer net from the player's model folder instead of its bundled one. "" without one.
func model_text():
	var m = server.get("model")
	if typeof(m) != TYPE_DICTIONARY or str(m.get("sha", "")) == "":
		return ""
	var t = " (model %s" % str(m.sha)
	if str(m.get("source", "")) == "model folder":
		t += ", from your model folder"
	return t + ")"


func _next_or_quit(nxt):
	if typeof(nxt) == TYPE_DICTIONARY:
		_start_match(str(nxt.p1), str(nxt.p2), int(nxt.seed))
	else:
		emit_signal("quitting")
		get_tree().quit()


# --- matches -----------------------------------------------------------------------------------

func _on_start_pressed():
	var ready = connected if side == "single" else not server.empty()
	if phase == "playing" or phase == "starting" or not ready:
		return
	randomize()
	_start_match(characters[p1_opt.selected], characters[p2_opt.selected], randi())


func _start_match(p1, p2, seed_):
	main = get_tree().current_scene
	if main == null or not main.has_method("_on_match_ready"):
		_say("The game's main scene isn't ready yet.")
		return
	if phase == "playing" and game != null:
		_send({"type": "abort", "reason": "a new match was started"})
	var options = {}
	var preset = str(server.get("options", {}).get("settings", "standard"))
	var timer_on = _timer_default(preset)
	if not autoplay:  # the panel's choices; autoplay keeps the launcher's
		preset = str(preset_names[max(0, settings_opt.selected)])
		timer_on = timer_box.pressed
		options = {"pick": pick_styles[max(0, pick_opt.selected)], "show_estimate": estimate_box.pressed,
			"settings": preset, "think_until_lockin": think_box.pressed, "turn_timer": timer_on}
	var think = think_box.pressed if not autoplay else bool(server.get("options", {}).get("think_until_lockin", false)) or timer_on
	if autoplay:  # the launcher's choices, with thinking on the clock as the panel's default
		options = {"think_until_lockin": think, "turn_timer": timer_on}
	var msg = {"p1": p1, "p2": p2, "seed": seed_, "settings": _match_settings(preset, timer_on), "options": options, "think": think}
	last_msg = msg.duplicate(true)
	if side == "host":
		# The client tells the server and starts the same match; both games then run it in lockstep.
		link.to_client("match", msg)
		remote_status = {}
		_launch(msg)
	else:
		start_bot_match(msg)


# Single and client: the server hears of the match, then the game starts it.
func start_bot_match(msg):
	main = get_tree().current_scene
	if main == null or not main.has_method("_on_match_ready"):
		_say("The game's main scene isn't ready yet.")
		return
	_send({"type": "match_start", "p1": msg.p1, "p2": msg.p2, "seed": msg.seed, "settings": msg.settings,
		"options": msg.options, "mod": VERSION}, "_on_match_started")
	_launch(msg)


# Start the match in this game. Host and client build the same match data from the same message.
func _launch(msg):
	match_ = {"p1": msg.p1, "p2": msg.p2, "seed": msg.seed}
	think_on = bool(msg.think)
	var match_settings = msg.settings
	_release_clock()
	var data = match_settings.duplicate(true)
	data["singleplayer"] = side == "single"
	data["selected_characters"] = {1: {"name": msg.p1}, 2: {"name": msg.p2}}
	data["seed"] = int(msg.seed)
	if side != "single":  # as the game's rematch resets them (Network.send_rematch_request)
		Network.forfeiter = 0
		Network.rematch_menu = false
	# As Main._on_game_started(true) then the character select's match_ready, minus the select.
	ReplayManager.playback = false
	ReplayManager.resimulating = false
	ReplayManager.replaying_ingame = false
	Global.frame_advance = false
	main.ui_layer.reset_ui()
	if is_instance_valid(main.game):
		main.game.free()
		main.game = null
		main.stop_ghost()
	main.singleplayer = side == "single"
	main.hide_main_menu(true)
	var css = main.get_node_or_null("%CharacterSelect")
	if css:
		css.hide()
	game = null
	turn = 0
	decision_tick = -1000
	auto_tick = -1000
	inputs_at = {}
	pick = null
	pick_pending = false
	desynced = false
	p2_fallback = false
	estimate = null
	shown_estimate = null
	pending_turn = -1
	depth_text = ""
	p1_locked_tick = -1000
	submitted_for = -1
	p2_timeouts = 0
	forfeiter = 0
	result_in = false
	ended_msec = 0
	phase = "starting"
	if panel:
		panel.hide()
	if card:
		card.hide()
	main._on_match_ready(data)


# The standard settings with a preset's overrides, each in the type the game's own setting has
# (JSON numbers arrive as floats; the game's integer settings must stay integers).
func _with_preset(name):
	var s = settings.duplicate(true)
	var over = presets.get(name, {})
	for k in over:
		var v = over[k]
		match typeof(s.get(k)):
			TYPE_INT:
				v = int(v)
			TYPE_REAL:
				v = float(v)
			TYPE_STRING:
				v = str(v)
			TYPE_BOOL:
				v = bool(v)
		s[k] = v
	return s


# The panel's "Turn timer" default for a preset: on when the preset has a timer (ranked: the game's ranked clock),
# off otherwise (standard). The launcher's turn_timer option, when it gives one, decides instead.
func _timer_default(name):
	var opts = server.get("options", {})
	if typeof(opts) == TYPE_DICTIONARY and typeof(opts.get("turn_timer")) == TYPE_BOOL:
		return opts.turn_timer
	return str(presets.get(name, {}).get("timer_mode", "none")) != "none"


# A match's settings: the preset's, with the game's turn timer on or off. On: the preset's own timer (ranked: a
# chess clock), or for a preset without one the game's lobby timer (the settings panel's defaults; a per-turn
# timer if those name none). Off: no timer (the standard settings' clock keys, timer_mode "none").
func _match_settings(name, timer_on):
	var s = _with_preset(name)
	if timer_on and str(s.get("timer_mode", "none")) == "none":
		for k in lobby_timer:
			s[k] = lobby_timer[k]
		if str(s.get("timer_mode", "none")) == "none":
			s["timer_mode"] = "default"
	elif not timer_on:
		for k in TIMER_KEYS:
			if settings.has(k):
				s[k] = settings[k]
	return s


# The result card's Rematch: the last match's characters and settings again, with a new seed (the panel's own
# choices are unchanged, so "Change settings" still shows them).
func _on_rematch_pressed():
	var ready = connected if side == "single" else not server.empty()
	if last_msg.empty() or phase == "playing" or phase == "starting" or not ready or not result_in:
		return
	randomize()
	var msg = last_msg.duplicate(true)
	msg["seed"] = randi()
	if side == "host":
		link.to_client("match", msg)
		remote_status = {}
		_launch(msg)
	else:
		start_bot_match(msg)


func _on_change_settings_pressed():
	if card:
		card.hide()
	_show_panel(_panel_title(), "")


func _on_preset_selected(i):
	timer_box.pressed = str(presets.get(str(preset_names[max(0, i)]), {}).get("timer_mode", "none")) != "none"
	think_box.pressed = timer_box.pressed  # on the clock, the bot thinks until you lock in (or its deadline)


func _on_match_started(r):
	if not r.get("ok", false):
		desynced = true
		_say("The bot can't play this match (it won't count): %s" % str(r.get("message", "the server refused it")))


func _attach():
	if not is_instance_valid(main) or not is_instance_valid(main.game) or not main.game.game_started:
		return
	game = main.game
	p1_panel = main.get_node("%P1ActionButtons")
	p2_panel = main.get_node("%P2ActionButtons")
	if side == "host":  # P2's moves come from the other game, through the game's own protocol
		game.connect("game_ended", self, "_on_host_game_ended")
		game.p1.connect("action_selected", self, "_on_host_move", [1])
		game.p2.connect("action_selected", self, "_on_host_move", [2])
		# The game opens its chat box for every multiplayer match; nobody is on the other end here, and it covers
		# the right of the stage. The chat key (Enter) still opens it.
		var chat = main.get_node_or_null("%ChatWindow")
		if chat:
			chat.hide()
		phase = "playing"
		return
	game.connect("game_ended", self, "_on_game_ended")
	game.p1.connect("action_selected", self, "_on_input", [1])
	game.p2.connect("action_selected", self, "_on_input", [2])
	if side == "single":
		# P2 is the bot's: shown greyed and clickable, as an online opponent's (its clicks set only P2's
		# ghost), but no keyboard lock-in, no auto-continue and a Select that does nothing.
		p2_panel.set_process_input(false)
		p2_panel.get_node("%AutoButton").set_pressed_no_signal(false)
		var sel = p2_panel.get_node("%SelectButton")
		if sel.is_connected("pressed", p2_panel, "_on_submit_pressed"):
			sel.disconnect("pressed", p2_panel, "_on_submit_pressed")
		p2_panel.modulate = Color("b3b3b3")  # the game's tint for an online opponent's panel
		_take_clock()
	phase = "playing"


func _play_frame():
	# A forfeit first: the game ends the match at once and starts its post-game replay (a new main.game), which
	# would otherwise read as the match being closed.
	var f = _forfeit_by()
	if f != 0:
		_on_forfeit(f)
		return
	if not is_instance_valid(game) or not is_instance_valid(main) or main.game != game:
		_abort("the match was closed")
		return
	if side == "single":
		for p in [p1_panel, p2_panel]:
			var u = p.get_node("%UndoButton")
			if u.visible:
				u.hide()
	if game.game_finished:
		return
	if side == "host":
		_host_frame()
		return
	if game.game_paused and game.is_waiting_on_player() and game.p1_turn and game.p2_turn and game.current_tick != decision_tick:
		_decision()
	if pick != null and p2_panel.active and game.p2.state_interruptable:
		_submit_pick()
	# A turn timer: tell the server to answer now, BACKSTOP_S before the bot's clock (the game's own P2 timer,
	# read as it runs) ends, in any mode and search. The server stops itself earlier (its deadline margin).
	if pick_pending and not deadline_sent:
		var left = _bot_clock_left()
		if left >= 0 and left <= BACKSTOP_S:
			deadline_sent = true
			_notify({"type": "deadline", "turn": pending_turn})
	if autoplay and auto_tick != decision_tick and decision_tick == game.current_tick and p1_panel.active and game.p1.state_interruptable:
		auto_tick = decision_tick
		_autoplay_p1()


func _decision():
	var prev = decision_tick
	decision_tick = game.current_tick
	if single_clock:
		_clock_turn_start()
	var msg = {"type": "decision", "turn": turn, "snapshot": _snapshot()}
	if turn > 0:
		if submitted_for != pending_turn:  # P2's clock ran out before the bot's answer: P2 played the timeout's move
			p2_fallback = true
			p2_timeouts += 1
		msg["inputs"] = _inputs_for(prev)
		msg["p2_fallback"] = p2_fallback
	p2_fallback = false
	if side == "single":  # before any click on P2's panel can change it
		p2_extra = p2_panel.get_extra()
	shown_estimate = estimate  # last turn's estimate: that turn has now been played
	estimate = null
	pick = null
	pick_pending = true
	pending_turn = turn
	decision_msec = OS.get_ticks_msec()
	deadline_sent = false
	depth_text = ""
	_send(msg, "_on_pick")
	var locked = p1_locked_tick == decision_tick if side == "client" else inputs_at.has(decision_tick) and inputs_at[decision_tick][0] != null
	if locked:  # P1 locked in before this was sent (every match: untimed, the after-lock cap counts from it)
		_notify({"type": "locked_in", "turn": pending_turn})
	turn += 1
	_report()


# Client: what the host's bottom bar shows. Last turn's estimate only (that turn has been played).
func _report():
	if side == "client":
		link.to_host("status", {"turn": turn, "desynced": desynced, "estimate": shown_estimate, "depth": depth_text})


# Client: the game's own "turn ready" (Network.multiplayer_turn_ready): THAT a player locked in. P1's
# input itself arrives only once both are ready (Network.send_action).
func _on_net_turn_ready(id):
	if id != 1 or not is_instance_valid(game):
		return
	p1_locked_tick = game.current_tick
	if pick_pending and phase == "playing" and decision_tick == game.current_tick:
		_notify({"type": "locked_in", "turn": pending_turn})


func _on_pick(r):
	# An answer to an earlier decision (P2's clock ran out first and the next turn has opened): too late, dropped.
	if typeof(r.get("turn")) in [TYPE_INT, TYPE_REAL] and int(r.turn) != pending_turn:
		return
	pick_pending = false
	if phase != "playing":
		return
	if r.get("type") != "pick":
		_say("Bot server error: %s" % str(r.get("message")))
		r = {"action": "Continue", "data": null, "extra": null, "fallback": true}
	desynced = r.get("desynced", false) or desynced
	if r.get("fallback", false):
		p2_fallback = true
	estimate = r.get("estimate", null)
	var d = r.get("depth")
	depth_text = str(d.get("label", "")) if typeof(d) == TYPE_DICTIONARY else ""
	pick = r
	_report()


func _submit_pick():
	var a = str(pick.get("action", "Continue"))
	var d = _from_json(pick.get("data"))
	var e = _from_json(pick.get("extra"))
	pick = null
	# In sync the pick is legal by construction. Desynced, a move the real P2 can't use would turn
	# into a forfeit (BaseChar.on_action_selected), so P2 holds instead.
	if desynced and a != "Continue":
		var st = game.p2.state_machine.get_state(a)
		if st == null or not st.is_usable():
			a = "Continue"
			d = null
			e = null
			p2_fallback = true
	# Single: P2's panel's own Select call. Client: this game's player is P2, so the same call also does
	# Network.submit_action, exactly as a player's Select does online.
	# A pick without extra would take the panel's, which the player may have clicked: the untouched one.
	if e == null and side == "single":
		e = p2_extra
	var left = _bot_clock_left()  # read before the lock-in pauses it
	submitted_for = pending_turn
	p2_panel.on_action_submitted(a, d, e)
	var clock = "" if left < 0 else "; its clock had %.2f s left" % left
	if side != "single":
		link.trace("turn %d: P2 locked in %d ms after the turn opened%s" % [pending_turn, OS.get_ticks_msec() - decision_msec, clock])
	elif left >= 0:
		print("YomiHuman: turn %d: P2 locked in %d ms after the turn opened%s" % [pending_turn, OS.get_ticks_msec() - decision_msec, clock])


# --- the turn timer ------------------------------------------------------------------------------

# Seconds left on the bot's clock (the game's own P2 turn timer) while it runs this turn, else -1 (no timer, or
# it isn't running: before the turn's clock starts, or after P2 locked in).
func _bot_clock_left():
	if not is_instance_valid(main) or main.ui_layer == null or str(main.ui_layer.timer_mode) == "none":
		return -1.0
	var t = main.ui_layer.p2_turn_timer
	if t.is_stopped() or t.paused:
		return -1.0
	return t.time_left


func _turn_timer(id):
	return main.ui_layer.p1_turn_timer if id == 1 else main.ui_layer.p2_turn_timer


# Single, at the match's start: with a timer in its settings, this script runs the game's two turn timers (the
# game starts them only online) and takes their expiry, so the bot's side times out as the bot, not as a click
# on its greyed panel. The labels show as online.
func _take_clock():
	var ui = main.ui_layer
	single_clock = str(ui.timer_mode) != "none"
	if not single_clock:
		return
	for id in [1, 2]:
		var t = _turn_timer(id)
		if t.is_connected("timeout", ui, "_on_turn_timer_timeout"):
			t.disconnect("timeout", ui, "_on_turn_timer_timeout")
		if not t.is_connected("timeout", self, "_on_clock_timeout"):
			t.connect("timeout", self, "_on_clock_timeout", [id])
	ui.get_node("%P1TurnTimerLabel").show()
	ui.get_node("%P2TurnTimerLabel").show()


# Gives the game back its own expiry handling (a new match, or a game started from the game's own menus).
func _release_clock():
	single_clock = false
	if not is_instance_valid(main) or main.ui_layer == null:
		return
	var ui = main.ui_layer
	for id in [1, 2]:
		var t = _turn_timer(id)
		if t.is_connected("timeout", self, "_on_clock_timeout"):
			t.disconnect("timeout", self, "_on_clock_timeout")
		if not t.is_connected("timeout", ui, "_on_turn_timer_timeout"):
			t.connect("timeout", ui, "_on_turn_timer_timeout", [id])


# A turn opens (single): the clocks as the game's UILayer.on_player_actionable sets them online. The first turn
# starts both banks (main.gd sized them from the settings); later ones restart a per-turn timer, add an
# increment (the game's own _apply_increment), or top a chess bank up to its per-turn minimum. P1's lock-in
# pauses its clock through the game's own end_turn_for, as online.
func _clock_turn_start():
	var ui = main.ui_layer
	var mode = str(ui.timer_mode)
	if not ui.game_started:
		if mode == "chess":
			ui.MIN_TURN_TIME = game.match_data.get("turn_min_length", ui.MIN_TURN_TIME)
		elif mode == "increment":
			ui.MIN_TURN_TIME = game.match_data.get("increment_per_turn", 0)
		ui.p1_turn_timer.start()
		ui.p2_turn_timer.start()
		ui.game_started = true
	elif mode == "default":
		ui.p1_turn_timer.start(ui.turn_time)
		ui.p2_turn_timer.start(ui.turn_time)
	elif mode == "increment":
		ui._apply_increment(1)
		ui._apply_increment(2)
		ui.p1_time_run_out = false
		ui.p2_time_run_out = false
	else:
		for t in [ui.p1_turn_timer, ui.p2_turn_timer]:
			if t.time_left < ui.MIN_TURN_TIME:
				t.start(ui.MIN_TURN_TIME)
	ui.p1_turn_timer.paused = false
	ui.p2_turn_timer.paused = false
	ui.get_node("%P1TurnTimerBar").show()
	ui.get_node("%P2TurnTimerBar").show()


# A clock ran out (single): as the game's UILayer._on_turn_timer_timeout does for a player online. P1's panel
# locks in what it holds (the game's ActionButtons.timeout). The bot's side: if its answer hasn't come, P2
# holds (Continue, the server's own move when its deadline cuts even its quick search); the late answer is
# dropped (_on_pick).
func _on_clock_timeout(id):
	if not single_clock or not is_instance_valid(game) or not is_instance_valid(main):
		return
	var ui = main.ui_layer
	var t = _turn_timer(id)
	if str(ui.timer_mode) == "increment":
		var inc = float(game.match_data.get("increment_per_turn", 0))
		if id == 1:
			ui.p1_debt += inc
			ui.p1_just_ran_out = true
		else:
			ui.p2_debt += inc
			ui.p2_just_ran_out = true
		t.start(inc)
		_timed_out(id)
		return
	_timed_out(id)
	t.wait_time = ui.MIN_TURN_TIME
	t.start()
	t.paused = true


func _timed_out(id):
	if id == 1:
		p1_panel.timeout()
	elif p2_panel.active and not game.game_finished:
		print("YomiHuman: turn %d: P2's clock ran out before the bot answered: P2 holds" % pending_turn)
		pick = null
		p2_panel.on_action_submitted("Continue", null, p2_extra)


func _on_input(_action, _data, _extra, id):
	# The input exactly as the fighter got it (before an unusable move becomes Forfeit), keyed by the
	# decision it answers. Copied now: the fighter's states may change the dicts during the turn.
	if not is_instance_valid(game):
		return
	if id == 1 and pick_pending and phase == "playing":
		_notify({"type": "locked_in", "turn": pending_turn})  # that P1 locked in, never what
	var f = game.p1 if id == 1 else game.p2
	var t = game.current_tick
	if not inputs_at.has(t):
		inputs_at[t] = [null, null]
	inputs_at[t][id - 1] = {"action": f.queued_action, "data": _dup(f.queued_data), "extra": _dup(f.queued_extra)}


func _inputs_for(tick):
	var out = inputs_at.get(tick, [null, null]).duplicate()
	inputs_at.erase(tick)
	for i in range(2):
		if out[i] == null:  # fall back on the replay's record of the input (the tick that used it)
			var rec = ReplayManager.frames[i + 1].get(tick + 1)
			if rec != null:
				out[i] = {"action": rec.action, "data": _dup(rec.data), "extra": _dup(rec.extra)}
	return out


func _on_game_ended(_winner = null):
	if phase != "playing":
		return
	var w = null
	if game.p1.hp > game.p2.hp:
		w = 0
	elif game.p2.hp > game.p1.hp:
		w = 1
	_send_end(w, null)


# Single and client: the match is over (a finish, or a forfeit by `forfeit_seat`, 0 = P1): the server gets the last
# inputs, the final snapshot and the winner.
func _send_end(w, forfeit_seat):
	phase = "ended"
	if single_clock or (side == "client" and str(main.ui_layer.timer_mode) != "none"):
		if turn > 0 and submitted_for != pending_turn:
			p2_timeouts += 1
		var note = "turn timer %s: P2 timed out on %d of %d turns" % [str(main.ui_layer.timer_mode), p2_timeouts, turn]
		if side == "client":
			link.trace(note)
		else:
			print("YomiHuman: ", note)
	var msg = {"type": "end", "turn": turn, "inputs": _inputs_for(decision_tick), "winner": w}
	if is_instance_valid(game):
		msg["snapshot"] = _snapshot()
	if forfeit_seat != null:
		msg["forfeit"] = forfeit_seat
	_send(msg, "_on_result")
	if side == "single" and not autoplay:
		ended_msec = OS.get_ticks_msec()
		var res = "You win!" if w == 0 else (_bot_wins() if w == 1 else "Draw.")
		_card_text(_forfeit_text() if forfeit_seat != null else "Game over. %s (%d turns)" % [res, turn], "Saving the game...")


# The player (1 or 2) whose forfeit ended the match, else 0. The game's forfeit (Network.player_forfeit ->
# game.forfeit) ends the match without its game_ended signal; Network.forfeiter outlives the game it ended.
func _forfeit_by():
	if side == "single":
		if is_instance_valid(game) and game.forfeit and game.forfeit_player != null:
			return int(game.forfeit_player.id)
		return 0
	if int(Network.forfeiter) in [1, 2]:
		return int(Network.forfeiter)
	if is_instance_valid(game) and game.forfeit and game.forfeit_player != null:
		return int(game.forfeit_player.id)
	return 0


func _on_forfeit(who):
	if phase != "playing":
		return
	forfeiter = who
	_release_clock()
	if side == "host":
		phase = "ended"
		link.trace("turn %d: P%d forfeited" % [turn, who])
		if autoplay:
			return
		ended_msec = OS.get_ticks_msec()
		_card_text(_forfeit_text(), "Saving the game...")
		return
	if side == "single":
		print("YomiHuman: turn %d: P%d forfeited" % [turn, who])
	else:
		link.trace("turn %d: P%d forfeited" % [turn, who])
	_send_end(2 - who, who - 1)  # the other player wins


func _forfeit_text():
	if forfeiter == 1:
		return "Game over. You forfeited: %s wins." % bot_name("the bot")
	return "Game over. %s forfeited: you win!" % bot_name("The bot")


# "HustleBot g15 wins." (the bot by its name, as the panel and the bottom bar name it).
func _bot_wins():
	return "%s wins." % bot_name("The bot")


func _on_result(r):
	if side == "client":
		phase = "idle"
		link.to_host("result", r)
		return
	games_played += 1
	if autoplay:
		print("YOMIHUMAN_RESULT ", JSON.print(r))
		yield(get_tree().create_timer(0.5), "timeout")
		_next_or_quit(r.get("next"))
		return
	_show_result(r)


func _abort(reason):
	_release_clock()
	if phase == "playing" or phase == "starting":
		if side == "host":
			link.to_client("abort", reason)
		else:
			_send({"type": "abort", "reason": reason})
	phase = "idle"
	game = null
	if not autoplay:
		_show_panel(_panel_title(), "The last match was closed before it ended.")


# --- multiplayer, host side ----------------------------------------------------------------------

# What the client relays (MpClient.gd -> VsBot.gd): the server's hello, the
# bottom bar's status, a message, the result.
func on_bot_message(kind, r):
	match kind:
		"hello":
			_on_hello(r)
		"status":
			remote_status = r if typeof(r) == TYPE_DICTIONARY else {}
		"say":
			_say(str(r))
		"result":
			_on_bot_result(r)


func _host_frame():
	if game.game_paused and game.is_waiting_on_player() and game.p1_turn and game.p2_turn and game.current_tick != decision_tick:
		decision_tick = game.current_tick
		turn += 1
		# The check that the bot's move stays hidden: everything of P2's this game holds as the turn opens.
		link.trace("turn %d opened (tick %d): P2 here: queued %s, network input %s, ghost %s, P2 locked in %s" % [
			turn, decision_tick, str(game.p2.queued_action), str(Network.action_inputs[2].action),
			str(main.p2_ghost_action), str(Network.turns_ready.get(2, false))])
	if autoplay and auto_tick != decision_tick and decision_tick == game.current_tick and p1_panel.active and game.p1.state_interruptable:
		auto_tick = decision_tick
		_autoplay_p1()


func _on_host_move(_action, _data, _extra, id):
	if not is_instance_valid(game):
		return
	if id == 1:  # P1's own panel: the fighter has it at once; the other game gets it once both are ready
		p1_lock_msec = OS.get_ticks_msec()
		link.trace("turn %d: P1 locked in (tick %d); P2 here: queued %s, network input %s, ghost %s, P2 locked in %s" % [
			turn, game.current_tick, str(game.p2.queued_action), str(Network.action_inputs[2].action),
			str(main.p2_ghost_action), str(Network.turns_ready.get(2, false))])
	else:  # Network.send_action: the bot's move, delivered by the game's own protocol
		link.trace("turn %d: P2's move arrived (tick %d) %d ms after P1 locked in: %s" % [
			turn, game.current_tick, OS.get_ticks_msec() - p1_lock_msec, str(game.p2.queued_action)])


func _on_host_game_ended(_winner = null):
	if phase != "playing":
		return
	phase = "ended"
	if autoplay:
		return
	var res = "You win!" if game.p1.hp > game.p2.hp else (_bot_wins() if game.p2.hp > game.p1.hp else "Draw.")
	ended_msec = OS.get_ticks_msec()  # the card waits a moment: the final hit, the HP bars and the clocks stay visible
	_card_text("Game over. %s (%d turns)" % [res, turn], "Saving the game...")


func _on_bot_result(r):
	games_played += 1
	if autoplay:
		print("YOMIHUMAN_RESULT ", JSON.print(r))
		link.trace("result " + JSON.print(r))
		yield(get_tree().create_timer(0.5), "timeout")
		_next_or_quit(r.get("next"))
		return
	_show_result(r)


# The result card (single and host): who won, then where the game was saved, with Rematch / Change settings / Main menu.
func _show_result(r):
	phase = "idle"
	result_in = true
	if r.get("type") != "result":
		_card_text("Game over", "The bot couldn't save this game: %s" % str(r.get("message")))
		_card_now()
		return
	var res = {"human": "You win!", "bot": _bot_wins(), "draw": "Draw."}.get(r.get("result"), str(r.get("result")))
	var t = "Game over. %s (%d turns)" % [res, int(r.get("turns", 0))]
	if typeof(r.get("forfeit")) in [TYPE_INT, TYPE_REAL] or forfeiter != 0:
		t = _forfeit_text()
	var saved = ("Saved to Documents\\YomiBench (it counts for the bot's stats)." if r.get("valid", false) else
		"Saved to Documents\\YomiBench, marked invalid: the bot's simulation lost track of the match.")
	_card_text(t, saved)
	_card_now()


func _card_text(t, i):
	if card == null:
		return
	card_title.text = t
	card_info.text = i
	rematch_btn.disabled = not result_in


# Show the card now, unless it is still waiting out its delay after the match ended (then _process shows it).
func _card_now():
	if ended_msec == 0:
		_show_card()


func _show_card():
	if autoplay or card == null or crashed:
		return
	if panel and panel.visible:  # "Change settings" was pressed: the panel stays, with the result on it
		info.text = card_title.text + "\n" + card_info.text
		return
	rematch_btn.disabled = not result_in
	card.show()


# The game's in-match replay ("playback", R) rebuilds the match from its replay, and "edit_replay"
# cuts the match at a replayed tick and plays on from there: the real match would leave the sim
# behind. Both are swallowed while one of our matches is on (autoloads see input first).
func _input(event):
	if (phase == "playing" or phase == "ended") and (event.is_action_pressed("playback") or event.is_action_pressed("edit_replay")):
		get_tree().set_input_as_handled()


func _notification(what):
	if what == MainLoop.NOTIFICATION_WM_QUIT_REQUEST and connected and phase == "playing" and side != "host":
		_send({"type": "abort", "reason": "the game was closed"})


# --- state -------------------------------------------------------------------------------------

# Compact real state for the sync check: the fields of YomiMatch._player / _detail the server
# compares (human_play.PLAYER_FIELDS, OBJECT_FIELDS), computed the same way.
func _snapshot():
	var ps = []
	for p in [game.p1, game.p2]:
		var pos = p.get_pos()
		var vel = p.get_vel()
		var st = p.current_state()
		ps.append({"hp": p.hp, "x": pos.x, "y": pos.y, "vx": str(vel.x), "vy": str(vel.y),
			"facing": p.get_facing_int(), "state": st.state_name, "state_tick": st.current_tick,
			"hitlag": p.hitlag_ticks, "busy_interrupt": p.busy_interrupt, "grounded": p.is_grounded(),
			"meter": p.supers_available, "meter_partial": p.super_meter, "burst": p.bursts_available,
			"burst_partial": p.burst_meter, "combo": p.combo_count, "feints": p.feints, "air_options": p.air_option_bar})
	var objs = []
	for o in game.objects:
		if is_instance_valid(o) and not o.disabled:
			var opos = o.get_pos()
			var ost = o.current_state()
			objs.append({"kind": o.filename.get_file().get_basename(), "owner": o.id, "x": opos.x, "y": opos.y,
				"state": ost.state_name if ost else ""})
	return {"tick": game.current_tick, "players": ps, "objects": objs}


func _dup(v):
	if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY:
		return v.duplicate(true)
	return v


# JSON numbers arrive as floats; the game's inputs are integers (ReplayManager.force_ints does the
# same for loaded replays). "[Object:null]" is how JSON prints a null Object (an option's icon).
func _from_json(v):
	match typeof(v):
		TYPE_REAL:
			return int(v) if v == floor(v) else v
		TYPE_STRING:
			return null if v == "[Object:null]" else v
		TYPE_DICTIONARY:
			var d = {}
			for k in v:
				d[k] = _from_json(v[k])
			return d
		TYPE_ARRAY:
			var a = []
			for x in v:
				a.append(_from_json(x))
			return a
	return v


# --- headless autoplay: P1 through its own panel -----------------------------------------------

func _autoplay_p1():
	var p = p1_panel
	for _i in range(3):
		yield(get_tree(), "idle_frame")
	if phase != "playing" or not p.active:
		return
	var legal = []
	for b in p.buttons:
		if is_instance_valid(b) and b.visible and not b.get_disabled():
			legal.append(b)
	if not legal.empty():
		var b = legal[rng.randi() % legal.size()]
		if not b.is_pressed():
			b.set_pressed(true)  # the button's toggled -> was_pressed -> on_action_selected, as a click
		for _i in range(3):
			yield(get_tree(), "idle_frame")
		if b.data_node and is_instance_valid(b.data_node):
			_randomize(b.data_node)
	var di = p.get_node("%DI")
	if di.is_visible_in_tree() and rng.randf() < 0.6:
		_random_xy(di)
	for name in ["%FeintButton", "%ReverseButton"]:
		var t = p.get_node(name)
		if t.is_visible_in_tree() and not t.disabled and rng.randf() < 0.25:
			t.pressed = not t.pressed
			p.send_ui_action()
	for _i in range(3):
		yield(get_tree(), "idle_frame")
	if autoplay_wait > 0:  # a slow player: the game's own turn timer may lock P1 in first
		yield(get_tree().create_timer(autoplay_wait), "timeout")
	if phase == "playing" and p.active:
		p._on_submit_pressed()  # the Select button


func _random_xy(n):
	var a = rng.randf() * TAU
	var r = sqrt(rng.randf()) * n.panel_radius
	n.update_value(Vector2(cos(a), sin(a)) * r, false, false)
	n.emit_signal("data_changed")


# Random values for a move's widgets, set through the widgets as a player's clicks would.
func _randomize(node):
	for c in node.get_children():
		if not (c is Control) or not c.is_visible_in_tree():
			continue
		if c.has_method("update_value") and c.get("panel_radius") != null:
			_random_xy(c)
		elif c is Range:
			c.value = c.min_value + rng.randf() * (c.max_value - c.min_value)
		elif (c is CheckBox or c is CheckButton) and rng.randf() < 0.3:
			c.pressed = not c.pressed
		else:
			_randomize(c)


# --- overlay -----------------------------------------------------------------------------------

func _make_overlay():
	layer = CanvasLayer.new()
	layer.layer = 120
	add_child(layer)
	var theme = load("res://theme.tres")
	# The bottom bar, in the game's bottom row (y 344-356 of the 640x360 UI), split around the row's middle: the
	# game's options-bar button ("^") sits at x 310-330. Left: who and which turn; right: the bot's estimate and
	# depth. Each is clipped to its half, and both hide while the game uses the row itself (_status_room).
	status = Label.new()
	status.rect_position = Vector2(4, 346)
	status.rect_size = Vector2(STATUS_HALF_W, 12)
	status.clip_text = true
	status_r = Label.new()
	status_r.rect_position = Vector2(640 - 4 - STATUS_HALF_W, 346)
	status_r.rect_size = Vector2(STATUS_HALF_W, 12)
	status_r.clip_text = true
	status_r.align = Label.ALIGN_RIGHT
	for l in [status, status_r]:
		l.mouse_filter = Control.MOUSE_FILTER_IGNORE
		if theme:
			l.theme = theme
		layer.add_child(l)
	panel = PanelContainer.new()
	if theme:
		panel.theme = theme
	panel.rect_position = Vector2(170, 40)
	panel.rect_min_size = Vector2(300, 0)
	layer.add_child(panel)
	var box = VBoxContainer.new()
	panel.add_child(box)
	title = Label.new()
	title.autowrap = true
	box.add_child(title)
	var row = HBoxContainer.new()
	box.add_child(row)
	var l1 = Label.new()
	l1.text = "You (P1):"
	row.add_child(l1)
	p1_opt = OptionButton.new()
	row.add_child(p1_opt)
	var l2 = Label.new()
	l2.text = "  Bot (P2):"
	row.add_child(l2)
	p2_opt = OptionButton.new()
	row.add_child(p2_opt)
	_fill_characters(characters)
	var row3 = HBoxContainer.new()
	box.add_child(row3)
	var l4 = Label.new()
	l4.text = "Settings:"
	row3.add_child(l4)
	settings_opt = OptionButton.new()
	for n in preset_names:
		settings_opt.add_item(n)
	row3.add_child(settings_opt)
	var row2 = HBoxContainer.new()
	box.add_child(row2)
	var l3 = Label.new()
	l3.text = "Bot style:"
	row2.add_child(l3)
	pick_opt = OptionButton.new()
	pick_opt.add_item("unpredictable (mixes its moves)")  # pick "mixed": samples its strategy
	pick_opt.add_item("strongest move only")              # pick "strongest": its most likely move
	row2.add_child(pick_opt)
	estimate_box = CheckBox.new()
	estimate_box.text = "Show the bot's win estimate"
	box.add_child(estimate_box)
	think_box = CheckBox.new()
	# think_until_lockin: the bot keeps searching while you decide (untimed it still answers within lock_cap_s of
	# your lock-in; on the clock, within its own time)
	think_box.text = "Bot uses your thinking time"
	box.add_child(think_box)
	timer_box = CheckBox.new()
	timer_box.text = "Turn timer (ranked clock)"
	timer_box.pressed = _timer_default(str(preset_names[0]))
	box.add_child(timer_box)
	settings_opt.connect("item_selected", self, "_on_preset_selected")
	start_btn = Button.new()
	start_btn.text = "Start game"
	start_btn.connect("pressed", self, "_on_start_pressed")
	box.add_child(start_btn)
	if hosted:
		var back_btn = Button.new()
		back_btn.text = "Back to main menu"
		back_btn.connect("pressed", self, "_on_back_pressed")
		box.add_child(back_btn)
	info = Label.new()
	info.autowrap = true
	box.add_child(info)
	log_btn = Button.new()  # crash: the bot's log folder
	log_btn.text = "Open log folder"
	log_btn.connect("pressed", self, "_on_log_pressed")
	log_btn.hide()
	box.add_child(log_btn)
	_make_card(theme)
	# Until the bot's hello the choices can't apply (the hello sets the presets and the defaults, and Start does
	# nothing): greyed, so nothing picked while connecting is silently reset. "Back to main menu" stays live.
	_set_controls(false)
	_show_panel(wait_title if wait_title != "" else "YomiBench: connecting to the bot server...", wait_text)
	if autoplay:
		panel.hide()


# The result card: small, in the lower middle (the stage's floor and the fighters stay visible above it), shown a
# moment after the match ends: who won, where the game was saved, and Rematch (the same characters and settings),
# Change settings (the full panel) and, from the game's menu, Main menu.
func _make_card(theme):
	card = PanelContainer.new()
	if theme:
		card.theme = theme
	card.rect_position = Vector2(150, 228)
	card.rect_min_size = Vector2(340, 0)
	layer.add_child(card)
	var box = VBoxContainer.new()
	card.add_child(box)
	card_title = Label.new()
	card_title.autowrap = true
	card_title.align = Label.ALIGN_CENTER
	box.add_child(card_title)
	var row = HBoxContainer.new()
	row.alignment = BoxContainer.ALIGN_CENTER
	row.add_constant_override("separation", 10)  # three distinct buttons, not one line of text
	box.add_child(row)
	rematch_btn = Button.new()
	rematch_btn.text = "Rematch"
	rematch_btn.rect_min_size = Vector2(96, 0)
	rematch_btn.connect("pressed", self, "_on_rematch_pressed")
	row.add_child(rematch_btn)
	var change = Button.new()
	change.text = "Change settings"
	change.rect_min_size = Vector2(96, 0)
	change.connect("pressed", self, "_on_change_settings_pressed")
	row.add_child(change)
	if hosted:
		var menu_btn = Button.new()
		menu_btn.text = "Main menu"
		menu_btn.rect_min_size = Vector2(96, 0)
		menu_btn.connect("pressed", self, "_on_back_pressed")
		row.add_child(menu_btn)
	card_info = Label.new()
	card_info.autowrap = true
	card_info.align = Label.ALIGN_CENTER
	box.add_child(card_info)
	card.hide()


func _set_controls(on):
	for c in [p1_opt, p2_opt, settings_opt, pick_opt, estimate_box, think_box, timer_box, start_btn]:
		if c != null:
			c.disabled = not on


# Host: VsBot.gd's progress and failure texts while the bot isn't up (the server's phase, or why it gave up).
func show_wait(t, text):
	wait_title = t
	wait_text = text
	if server.empty():
		_show_panel(t, text)


# The bot server's log (where a crash's details are), as the short Windows path the README gives.
static func crash_text():
	return CRASH_TEXT % "%USERPROFILE%\\.yomibench\\bot-server.log"


func _on_log_pressed():
	var f = server_file("bot-server.log")
	if f != "":
		OS.shell_open(f.get_base_dir())


# The bot server went away (it crashed, or something ended it): say so, on top of the match. Single: the lost
# connection; host: the client's report, its leaving, or the server's status file going stale (VsBot.gd).
func bot_crashed(text = ""):
	if crashed:
		return
	crashed = true
	if text == "":
		text = crash_text()
	print("YOMIHUMAN_CRASH ", text)
	_release_clock()
	pick_pending = false
	if phase == "playing" or phase == "starting" or phase == "ended":
		phase = "idle"
	if autoplay:  # a test: end with an error the rig can see
		emit_signal("quitting")
		get_tree().quit(4)
		return
	if panel != null:
		_set_controls(false)  # no new match against a server that's gone (single: the launcher's restart re-enables)
		title.text = "The bot stopped"
		info.text = text
		log_btn.show()
		if card:
			card.hide()
		panel.show()
	# The match can't go on: the player's own action panel greyed and out of reach.
	if side == "host" and is_instance_valid(p1_panel):  # (single: the launcher restarts the bot in the same game)
		p1_panel.modulate = Color("b3b3b3")
		p1_panel.set_process_input(false)
		_ignore_mouse(p1_panel)
	if status != null:
		status.text = "YomiBench: the bot stopped. Go back to the main menu to start again."
		status_r.text = ""


func _ignore_mouse(n):
	if n is Control:
		n.mouse_filter = Control.MOUSE_FILTER_IGNORE
	for c in n.get_children():
		_ignore_mouse(c)


func _on_back_pressed():
	if (phase == "playing" or phase == "starting") and side == "single":
		_send({"type": "abort", "reason": "back to the main menu"})  # host: the client goes with the session
	phase = "ended"
	emit_signal("leave_requested")


# The character pickers: `names` in order, keeping each picker's choice if it's still offered (else
# the launcher's --yomi-human-p1/p2, else the first).
func _fill_characters(names):
	var keep = [_option("human-p1", "Ninja"), _option("human-p2", "Ninja")]
	if p1_opt.get_item_count() > 0 and p1_opt.selected >= 0 and p2_opt.selected >= 0:
		keep = [characters[p1_opt.selected], characters[p2_opt.selected]]
	characters = []
	for c in names:
		characters.append(str(c))
	for k in range(2):
		var opt = [p1_opt, p2_opt][k]
		opt.clear()
		for c in characters:
			opt.add_item(c)
		opt.select(max(0, characters.find(keep[k])))


func _show_panel(t, i):
	if autoplay or panel == null:
		return
	title.text = t
	info.text = i
	panel.show()


func _say(t):
	print("YomiHuman: ", t)
	if side == "client":
		link.to_host("say", t)
		return
	if info:
		info.text = t
		if not autoplay and phase != "playing":
			panel.show()


func _update_status():
	if status == null:
		return
	var room = _status_room()
	status.visible = room
	status_r.visible = room
	if crashed:
		return
	if side == "host":
		_update_host_status()
		return
	var s = "YomiBench vs %s%s" % [bot_name("?"), model_text()] if connected else "YomiBench: waiting for the bot server on port %d" % port
	var r = []
	if phase == "playing":
		s += "  |  game %d, turn %d" % [games_played + 1, turn]
		if desynced:
			s += "  |  OUT OF SYNC: won't count"
		if pick_pending and is_instance_valid(game) and not game.p1.state_interruptable:
			r.append("bot: thinking...")
		if shown_estimate != null:
			r.append("est. %d%% for the bot" % int(shown_estimate))
		if _deep_label(depth_text) != "":
			r.append(_deep_label(depth_text))
	status.text = s
	status_r.text = PoolStringArray(r).join("  |  ")


# Host: the bot's status from the game's own turn-ready flags (Network.turns_ready: THAT a player locked in, what online
# play shows as "Ready"), never its move.
func _update_host_status():
	var s = "YomiBench vs %s%s" % [bot_name("?"), model_text()] if not server.empty() else "YomiBench: starting the bot's game..."
	if server.empty() and wait_text != "":
		s = "YomiBench: " + wait_text.split("\n")[0]
	var r = []
	if phase == "playing":
		s += "  |  game %d, turn %d" % [games_played + 1, turn]
		if remote_status.get("desynced", false) or desynced:
			s += "  |  OUT OF SYNC: won't count"
		var b = bot_status()
		if b != "":
			r.append(b)
		if remote_status.get("estimate") != null:
			r.append("est. %d%%" % int(remote_status.estimate))
		if _deep_label(str(remote_status.get("depth", ""))) != "":
			r.append(_deep_label(str(remote_status.depth)))
	status.text = s
	status_r.text = PoolStringArray(r).join("  |  ")


# Host, a turn open: "bot: thinking...", "bot: locked in", or untimed once you locked in, "bot: finishing... N s" (the
# after-lock cap: the bot answers within lock_cap_s of your lock-in). "" when no turn is waiting on the players.
func bot_status():
	if not is_instance_valid(game) or game.game_finished or not game.game_paused or decision_tick != game.current_tick:
		return ""
	var ready = Network.turns_ready if typeof(Network.turns_ready) == TYPE_DICTIONARY else {}
	if ready.get(2, false):
		return "bot: locked in"
	var untimed = not is_instance_valid(main) or main.ui_layer == null or str(main.ui_layer.timer_mode) == "none"
	if ready.get(1, false) and untimed and lock_cap_s > 0:
		var left = lock_cap_s - (OS.get_ticks_msec() - p1_lock_msec) / 1000.0
		return "bot: finishing... %d s" % int(ceil(max(0.0, left)))
	return "bot: thinking..."


# The bot's search depth in plain words, only when it went beyond its full search ("searched deep"); "" otherwise.
func _deep_label(d):
	return "searched deep" if str(d).begins_with("2-decision") or str(d) == "widened" else ""


# The game's pause menu (Escape) sits under our layer: our panel and card step aside while it is open.
func _pause_menu_frame():
	var m = main if is_instance_valid(main) else get_tree().current_scene
	var pp = m.get_node_or_null("%PausePanel") if m != null and m.has_method("_on_match_ready") else null
	var open = pp != null and pp.is_visible_in_tree()
	if open and pause_hid.empty():
		for c in [panel, card]:
			if c != null and c.visible:
				c.hide()
				pause_hid.append(c)
	elif not open and not pause_hid.empty():
		for c in pause_hid:
			if is_instance_valid(c):
				c.show()
		pause_hid = []


# Whether the bottom bar may show: not while our panel is up (it says the same, and on the main menu the bar would
# sit on the game's version and Mod List labels), and not while the game uses its bottom row itself: its options
# bar (the "^" button's ghost and speed settings), or its HUD hidden (F1 hides the game's UI layer).
func _status_room():
	if autoplay:
		return true
	if panel != null and panel.visible:
		return false
	if card != null and card.visible:
		return false
	var m = get_tree().current_scene
	if m == null or not m.has_method("_on_match_ready"):
		return true
	var ui = m.get("ui_layer")
	if ui != null and is_instance_valid(ui) and not ui.visible:
		return false
	var bar = m.get_node_or_null("%OptionsBar")
	return bar == null or not bar.visible
