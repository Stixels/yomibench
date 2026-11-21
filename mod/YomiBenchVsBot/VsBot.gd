extends Node
# Runs one "Multiplayer (vs bot)" session (docs/HUMAN_PLAY.md, "In the game's menu"). Three processes, all on
# this PC and all stopped with the game:
#  - the YomiBench bot server on a free 127.0.0.1 port (it plays from its own simulation, never from the game);
#  - a headless copy of this game (the "client": P2, MpClient.gd), started from user://yomibench-mp-client with its
#    own user folder, mods off, no window and no sound;
#  - this game, the "host": P1 and the panel (YomiHuman.gd, side "host").
# The two games play through the game's own direct-connect multiplayer (Network.gd), with the host's ENet bound to
# 127.0.0.1 only: simultaneous hidden choices, the online ghost, the online turn timers. The client talks to the
# bot server and submits each pick as a player's game does; this game never sees the bot's pick before the game's
# own protocol delivers it (once both locked in). Our own messages (the panel's choices, the relayed hello, status
# and results) are RPCs through MpLink.gd, at /root/YomiBenchVsBot/MpLink in both games.
#
# The server is found in this order:
#  1. user://yomibench_vsbot.json: {"server": "<exe>", "args": [...extra args before ours...], "out": "<folder>",
#     "exit_with": false, "app": false} (development: e.g. a Python interpreter and a script; use forward slashes in
#     paths; "app": true when it is packaging/windows/yomibench_app.py, which takes --game-dir and writes its status);
#  2. the server inside the one-step mod (res://YomiBenchVsBot/server, copied to user:// on first use);
#  3. the YomiBench install: %LOCALAPPDATA%\Programs\YomiBench\server\yomibench-bot-server.exe.
# Games go to "out" (default: Documents\YomiBench\games-root), marked held out of training by the server.
# Nothing here listens beyond 127.0.0.1, and it refuses to start while any online session or Steam lobby is on.

const NAME = "YomiBenchVsBot"
const HUMAN = preload("res://YomiBenchVsBot/YomiHuman.gd")
const CONFIG = "user://yomibench_vsbot.json"
const CLIENT_DIR = "user://yomibench-mp-client"
const CLIENT_SCRIPTS = ["MpClient.gd", "MpLink.gd", "YomiHuman.gd", "SteamStub.gd"]
const LINK = preload("res://YomiBenchVsBot/MpLink.gd")
const HELLO_TIMEOUT_MS = 90000
# Waiting for the bot (the app's server writes its progress to bot-server.status: YomiHuman.server_status). Every
# deadline ends the wait with the reason on the panel, never a silent "connecting..." forever.
const NO_STATUS_MS = 30000       # no status file from our server this long after starting it: it never ran
const AFTER_LISTEN_MS = 60000    # listening, but no hello through the bot's copy this long after
const STALE_S = 30               # the status file's heartbeat (5 s) stopped this long: the server is gone
const STALE_STARTING_S = 120     # ... while it still imports (before its heartbeat starts)
const STALE_PLAYING_S = 20       # once the bot is up: its heartbeat stopped this long, the server is gone (a crash)
const PHASE_TEXT = {
	"starting": "Starting the bot server...",
	"finding game": "Finding your game install...",
	"building game data": "Building the bot's game data from your game (first run, up to a minute)...",
	"loading bot": "Loading the bot...",
	"listening": "Starting the bot's copy of the game...",
}

var pid = -1
var client_pid = -1
var launcher_pid = -1   # the bot server exe in --launch-hidden mode: it holds the client (ending it ends the client)
var human = null
var port = 0
var enet = null
var life = null          # TCP_Server on 127.0.0.1: the client quits as soon as its connection to this game closes
var life_conn = null
var token = ""
var client_peer = 0      # the client's network id, once it gave the token
var started_msec = 0
var leaving = false
var link = null         # MpLink.gd: our RPCs (the same script, at the same path, in the client)
var app_server = false  # the YomiBench app's server (bundled or installed): it takes --game-dir, writes its status
var waiting = false     # the bot isn't up yet (no hello)
var failed = false
var spawn_msec = 0
var spawn_unix = 0
var server_pid = 0      # the pid in the status file we follow (ours: pid, or the first written after our start)
var last_phase = ""
var listen_msec = 0
var next_poll = 0
var crashed = false     # the bot server went away after it was up (YomiHuman.bot_crashed shows it)


static func instance(tree):
	return tree.root.get_node_or_null(NAME)


static func autoplay_schedule():
	for a in OS.get_cmdline_args():
		if a.begins_with("--yomibench-vsbot-autoplay="):
			return a.substr("--yomibench-vsbot-autoplay=".length())
	return ""


# Why "Multiplayer (vs bot)" can't start now, or "" if it can: never next to a real online session.
static func online_busy(tree):
	if Network.multiplayer_active or Network.steam or Network.direct_connect or tree.network_peer != null:
		return "a multiplayer session is open"
	if SteamLobby.LOBBY_ID != 0 or SteamLobby.SPECTATING:
		return "you are in a Steam lobby"
	# Gameplay mods (server-side, not characters: the ones online play makes both players share): the bot's copy of
	# the game and its simulation play the base game only, so the two games would play different matches.
	if not Network.normal_mods.empty():
		var names = []
		for h in Network.normal_mods:
			names.append(str(Network.hash_to_folder.get(h, h)).replace("res://", ""))
		return "gameplay mods are on (%s)" % PoolStringArray(names).join(", ")
	return ""


func _init():
	name = NAME
	pause_mode = PAUSE_MODE_PROCESS
	link = LINK.new()
	link.name = "MpLink"
	link.target = self
	add_child(link)


# The bot's copy of the game is made from this install, so this game must be the player's own copy, started
# through Steam: the game's own Steam autoload (SteamHustle) reports IS_OWNED from Steam itself.
static func steam_owned(tree):
	var s = tree.root.get_node_or_null("SteamHustle")
	return s != null and bool(s.get("IS_OWNED"))


func launch():
	var busy = online_busy(get_tree())
	if busy != "":
		_message("Multiplayer (vs bot) is off while %s." % busy)
		return false
	if not steam_owned(get_tree()):
		_message("Multiplayer (vs bot) needs the game started through Steam, from your own copy.")
		return false
	var cmd = _server_command()
	if cmd.empty():
		_message(("The YomiBench bot server isn't installed, or Windows Security removed it (Windows Security > " +
			"Virus & threat protection > Protection history: restore or allow yomibench-bot-server.exe). Install " +
			"YomiBench or re-add the mod, or set \"server\" in %s.\n\n%s") % [ProjectSettings.globalize_path(CONFIG), _log_paths()])
		return false
	port = _free_port()
	app_server = cmd.app
	var args = cmd.args.duplicate()
	args += ["--port", str(port), "--out", cmd.out]
	if cmd.exit_with:  # the installed server stops by itself if the game ends without saying so (a crash)
		args += ["--exit-with", str(OS.get_process_id())]
	if app_server:  # the scenes come from this very game (no Steam library search)
		args += ["--game-dir", OS.get_executable_path().get_base_dir()]
	var sched = autoplay_schedule()
	if sched != "":
		args += ["--autoplay", sched]
	_log("starting the bot server: %s %s" % [cmd.exe, str(args)])
	spawn_unix = OS.get_unix_time()
	pid = _spawn(cmd.exe, args)
	if pid <= 0:
		_message(("Couldn't start the bot server (%s). Windows Security may have blocked it: check Windows Security > " +
			"Virus & threat protection > Protection history.\n\n%s") % [cmd.exe, _log_paths()])
		return false
	spawn_msec = OS.get_ticks_msec()
	waiting = true
	var err = _host()
	if err != "":
		_stop_children()
		if get_tree().network_peer != null:
			Network.stop_multiplayer()
		_message("Couldn't start Multiplayer (vs bot): %s." % err)
		return false
	human = HUMAN.new()
	human.name = "YomiHuman"
	add_child(human)
	human.connect("leave_requested", self, "leave")
	human.connect("quitting", self, "_stop_children")
	human.start(0, "127.0.0.1", sched != "", true, "host", self)
	Network.connect("multiplayer_stopped", self, "_on_multiplayer_stopped")
	return true


# The host's half of Network.host_game_direct, with the ENet socket bound to 127.0.0.1 (the game's own binds every
# interface) and one peer at most; then the client copy is started to join it.
func _host():
	var e = null
	var enet_port = 0
	for _i in range(20):
		enet_port = 7800 + randi() % 100
		e = NetworkedMultiplayerENet.new()
		e.set_bind_ip("127.0.0.1")
		if e.create_server(enet_port, 1) == OK:
			break
		e = null
	if e == null:
		return "no free local port"
	life = TCP_Server.new()
	var life_port = 0
	for _i in range(20):
		life_port = 7700 + randi() % 100
		if life.listen(life_port, "127.0.0.1") == OK:
			break
		life_port = 0
	if life_port == 0:
		e.close_connection()
		return "no free local port"
	Network._reset()
	Network.second_register = false  # the registration check starts afresh (the game never resets it)
	var me = str(Global.get_player_data().get("username", "")).strip_edges()
	Network.player_name = me if me != "" else "You"
	Network.peer = e
	enet = e
	get_tree().network_peer = e
	Network.multiplayer_active = true
	Network.direct_connect = true
	Network.multiplayer_host = true
	# As the game's own direct host registers (cl_port/Network.gd): its mod lists ride along, and the client's
	# registration is checked against them.
	Network.rpc_("register_player", [Network.player_name, get_tree().get_network_unique_id(), {"active_mods": ModLoader.active_mods,
		"normal_mods": Network.normal_mods, "hash_to_folder": Network.hash_to_folder, "char_mods": Network.char_mods,
		"version": Global.VERSION}])
	get_tree().connect("network_peer_connected", self, "_on_peer_connected", [], CONNECT_DEFERRED)
	get_tree().connect("network_peer_disconnected", self, "_on_peer_disconnected", [], CONNECT_DEFERRED)
	randomize()
	token = "%08x%08x" % [randi(), randi()]
	var dir = _client_run_dir()
	if dir == "":
		return "couldn't prepare the bot's copy of the game"
	var cfg = {"enet": enet_port, "life": life_port, "bot": port, "token": token,
		"version": Marshalls.variant_to_base64(Global.VERSION)}
	var exe = dir.plus_file(OS.get_executable_path().get_file())
	var args = ["--no-window", "--disable-render-loop", "--audio-driver", "Dummy",
		"--yomibench-mp-client=" + Marshalls.utf8_to_base64(JSON.print(cfg))]
	_log("starting the bot's game: %s (ENet 127.0.0.1:%d, lifeline %d)" % [exe, enet_port, life_port])
	client_pid = _spawn_hidden(exe, args)
	if client_pid <= 0:
		return "couldn't start the bot's copy of the game"
	started_msec = OS.get_ticks_msec()
	set_process(true)
	return ""


func _process(_delta):
	if life == null:
		return
	if life_conn == null and life.is_connection_available():
		life_conn = life.take_connection()  # held open, never read: closing it (or this game ending) ends the client
		life.stop()
	if waiting and human != null:
		if not human.server.empty():
			waiting = false
			_log("the bot is up (%.1f s after starting the server)" % ((OS.get_ticks_msec() - spawn_msec) / 1000.0))
		elif OS.get_ticks_msec() >= next_poll:
			next_poll = OS.get_ticks_msec() + 1000
			_check_server()
	elif not waiting and not failed and not crashed and human != null and OS.get_ticks_msec() >= next_poll:
		next_poll = OS.get_ticks_msec() + 2000
		_check_alive()
	if client_peer == 0 and started_msec > 0 and OS.get_ticks_msec() - started_msec > HELLO_TIMEOUT_MS and not failed:
		started_msec = 0
		_fail("The bot's copy of the game didn't start. Go back to the main menu and try again.")


# Once a second while the bot isn't up: the server's phase on the panel, and the deadlines.
func _check_server():
	var now = OS.get_ticks_msec()
	var st = HUMAN.server_status() if app_server else {}
	var spid = int(st.get("pid", 0))
	var ours = not st.empty() and (spid == pid or spid == server_pid or (server_pid == 0 and float(st.get("time", 0)) >= spawn_unix - 1))
	if ours:
		if server_pid == 0:
			server_pid = spid
		var phase = str(st.phase)
		var detail = str(st.get("detail", ""))
		if phase != last_phase:
			last_phase = phase
			_log("bot server: %s%s (%.1f s)" % [phase, ": " + detail if detail != "" else "", (now - spawn_msec) / 1000.0])
			if phase == "listening":
				listen_msec = now
		if phase == "error":
			_fail("The bot server stopped: %s" % detail)
		elif phase == "stopped":
			_fail("The bot server stopped before the bot's copy of the game reached it.")
		elif OS.get_unix_time() - float(st.get("time", 0)) > (STALE_STARTING_S if phase == "starting" else STALE_S):
			_fail("The bot server stopped without saying why (at \"%s\"; antivirus may have closed it). Its last log line: %s" % [
				phase, _last_log_line()])
		elif listen_msec > 0 and now - listen_msec > AFTER_LISTEN_MS:
			_fail("The bot's copy of the game couldn't reach the bot server.")
		elif human:
			human.show_wait("YomiBench: connecting to the bot server...", PHASE_TEXT.get(phase, phase))
	elif app_server and server_pid == 0 and now - spawn_msec > NO_STATUS_MS:
		_fail("The bot server didn't start (Windows Security may have blocked yomibench-bot-server.exe: check Windows " +
			"Security > Virus & threat protection > Protection history, and restore or allow it).")


# Every 2 s once the bot is up (mid-match too): the server's status file still beats and doesn't say it stopped.
func _check_alive():
	if not app_server or server_pid == 0:
		return
	var st = HUMAN.server_status()
	if st.empty() or int(st.get("pid", 0)) != server_pid:
		return  # another server wrote it since (or it can't be read now): the client's connection still tells
	var phase = str(st.get("phase", ""))
	if phase == "error" or phase == "stopped" or OS.get_unix_time() - float(st.get("time", 0)) > STALE_PLAYING_S:
		_crashed("the bot server's status says %s (last beat %d s ago)" % [phase, OS.get_unix_time() - int(float(st.get("time", 0)))])


# The bot server went away after it was up (it crashed mid-match, or something ended it): tell the player, end the
# bot's copy and the server (the match can't go on), and wait for them to go back to the main menu.
func _crashed(why, text = ""):
	if crashed or leaving:
		return
	crashed = true
	_log("the bot crashed: %s; its last log line: %s" % [why, _last_log_line()])
	_stop_children()
	if human:
		human.bot_crashed(text)


# Stop waiting: say why on the panel (and in the log), with where the logs are; end the server and the bot's copy.
func _fail(text):
	if failed:
		return
	failed = true
	waiting = false
	_log("never ready within %.0f s: %s" % [(OS.get_ticks_msec() - spawn_msec) / 1000.0, text])
	_stop_children()
	if human:
		human.show_wait("YomiBench: the bot couldn't start", text + "\n\n" + _log_paths() +
			"\nGo back to the main menu to try again.")
	if autoplay_schedule() != "":
		get_tree().quit(3)  # a test: end with an error


# The two logs, as the README names them (%USERPROFILE% / %APPDATA%: short, and no user name on screen).
func _log_paths():
	return "Details: %s\nand %s" % [_short_path(HUMAN.server_file("bot-server.log")),
		_short_path(ProjectSettings.globalize_path("user://yomibench_vsbot.log"))]


static func _short_path(p):
	p = str(p).replace("/", "\\")
	for v in ["APPDATA", "USERPROFILE"]:  # the longer prefix first: APPDATA is inside USERPROFILE
		var root = OS.get_environment(v).replace("/", "\\")
		if root != "" and p.to_lower().begins_with(root.to_lower() + "\\"):
			return "%" + v + "%" + p.substr(root.length())
	return p


func _last_log_line():
	var f = File.new()
	var path = HUMAN.server_file("bot-server.log")
	if path == "" or not f.file_exists(path) or f.open(path, File.READ) != OK:
		return "(no log)"
	var n = f.get_len()
	f.seek(max(0, n - 4096))
	var lines = f.get_buffer(n - f.get_position()).get_string_from_utf8().strip_edges().split("\n")
	f.close()
	var last = str(lines[lines.size() - 1]).strip_edges() if lines.size() > 0 else ""
	return last.left(300) if last != "" else "(empty)"


func _on_peer_connected(id):
	_log("peer %d connected" % id)
	yield(get_tree().create_timer(15.0), "timeout")
	if not leaving and client_peer != id and get_tree().network_peer == enet and enet != null:
		_log("peer %d never gave the token; disconnecting it" % id)
		enet.disconnect_peer(id)


func _on_peer_disconnected(id):
	if id == client_peer:
		_log("the bot's game left")
		client_peer = 0
		if not waiting and not failed and not leaving:  # mid-session: it only leaves when the bot server is gone
			_crashed("the bot's copy of the game left")


# Client -> host: the client copy gives the token; the game's own sync_ids then makes this game P1 and it P2.
func on_mp_hello(sender, tok):
	if str(tok) != token or client_peer != 0:
		_log("peer %d gave a wrong token" % sender)
		enet.disconnect_peer(sender)
		return
	client_peer = sender
	_log("the bot's game joined as peer %d" % sender)
	Network.rpc_("sync_ids", {1: Network.get_local_id(), 2: sender})
	_lower_priority(_client_game_pid())


# The bot's copy must never show a window, not even briefly (this game build ignores --no-window and shows one once
# its settings load). GDScript can't choose the desktop a process starts on, so the bot server exe starts it, in its
# --launch-hidden mode, on a separate desktop that is never shown (yomibench.hidden_desktop), at below-normal
# priority, and writes its pid to a file. That launcher lives as long as the copy and ending it ends the copy.
# Returns the launcher's pid (what _stop_children ends), or the copy's own if the fallback below was used.
func _spawn_hidden(exe, args):
	var cmd = _server_command()
	var pid_file = ProjectSettings.globalize_path(CLIENT_DIR).plus_file("client.pid")
	Directory.new().remove(pid_file)
	if not cmd.empty():
		var largs = cmd.args.duplicate()
		largs += ["--launch-hidden", "--pid-file", pid_file, exe]
		largs += args
		launcher_pid = _spawn(cmd.exe, largs)
		if launcher_pid > 0:
			_log("the bot's game starts on the hidden desktop (launcher pid %d)" % launcher_pid)
			return launcher_pid
	_log("no hidden-desktop launcher; starting the bot's game directly and hiding its window")
	_start_window_hider(exe)  # before the spawn, so it's ready when the copy's window appears
	return _spawn(exe, args)


# The copy's own pid: from the launcher's pid file, else the pid _spawn gave.
func _client_game_pid():
	if launcher_pid <= 0:
		return client_pid
	var f = File.new()
	var path = ProjectSettings.globalize_path(CLIENT_DIR).plus_file("client.pid")
	if f.open(path, File.READ) != OK:
		return -1
	var text = f.get_as_text().strip_edges()
	f.close()
	return int(text) if text.is_valid_integer() else -1


# Fallback, when the launcher can't be started: a headless PowerShell (no window of its own) hides every top-level window of any process running the bot copy's
# executable. It starts before the copy (compiling its user32 helper takes a few seconds) and lives as long as this
# game, so one per game session is enough.
func _start_window_hider(exe):
	var conhost = OS.get_environment("SystemRoot").plus_file("System32").plus_file("conhost.exe")
	if get_tree().root.has_meta("yomibench_window_hider") or not File.new().file_exists(conhost):
		return
	get_tree().root.set_meta("yomibench_window_hider", true)
	var path = str(exe).replace("/", "\\").replace("'", "''")
	var ps = ("Add-Type -Namespace YB -Name W -MemberDefinition '" +
		"[DllImport(\"user32.dll\")] public static extern bool ShowWindow(System.IntPtr h, int c);" +
		"[DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(System.IntPtr h);" +
		"public delegate bool EnumProc(System.IntPtr h, System.IntPtr l);" +
		"[DllImport(\"user32.dll\")] public static extern bool EnumWindows(EnumProc f, System.IntPtr l);" +
		"[DllImport(\"user32.dll\")] public static extern uint GetWindowThreadProcessId(System.IntPtr h, out uint p);';" +
		"$x = '%s'; $t = [DateTime]::Now; while (Get-Process -Id %d -ErrorAction SilentlyContinue) {" +
		" $ids = @(Get-Process -ErrorAction SilentlyContinue | ? { $_.Path -eq $x } | % { $_.Id });" +
		" if ($ids.Count) { [YB.W]::EnumWindows({ param($h, $l) $q = 0; [void][YB.W]::GetWindowThreadProcessId($h, [ref]$q);" +
		" if ($ids -contains $q -and [YB.W]::IsWindowVisible($h)) { [void][YB.W]::ShowWindow($h, 0) }; $true }, [System.IntPtr]::Zero) | Out-Null };" +
		" if (([DateTime]::Now - $t).TotalSeconds -lt 60) { Start-Sleep -Milliseconds 50 } else { Start-Sleep -Seconds 1 } }") % [path, OS.get_process_id()]
	# -EncodedCommand (UTF-16LE base64): the C# quotes would not survive a plain -Command argument. ASCII only.
	var raw = PoolByteArray()
	for b in ps.to_ascii():
		raw.append(b)
		raw.append(0)
	var r = OS.execute(conhost, ["--headless", "powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
		Marshalls.raw_to_base64(raw)], false)
	_log("hiding the bot's game windows (%s): %s" % [path, r])


# The engine raises every copy of the game to AboveNormal priority as it starts; the bot's copy goes to BelowNormal
# once it has joined, so the player's game comes first. PowerShell through a headless console host: no window.
func _lower_priority(p):
	var conhost = OS.get_environment("SystemRoot").plus_file("System32").plus_file("conhost.exe")
	if p <= 0 or not File.new().file_exists(conhost):
		return
	var ps = "$p = Get-Process -Id %d -ErrorAction SilentlyContinue; if ($p) { $p.PriorityClass = 'BelowNormal' }" % p
	var r = OS.execute(conhost, ["--headless", "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps], false)
	_log("lowering the bot's game to BelowNormal (pid %d): %s" % [p, r])


# Client -> host: what the panel shows (see YomiHuman.on_bot_message).
func on_mp_from_bot(sender, kind, payload):
	if client_peer == 0 or sender != client_peer:
		return
	if str(kind) == "trace":
		_log("bot's game: " + str(payload))
	elif str(kind) == "bot_unreachable":
		if waiting:
			_fail(str(payload))
	elif str(kind) == "bot_crashed":
		if waiting:
			_fail("The bot server stopped before the bot's copy of the game reached it.")
		else:
			_crashed("the bot's copy of the game lost the bot server", str(payload))
	elif human:
		human.on_bot_message(str(kind), payload)


func to_client(kind, payload):
	if client_peer != 0:
		link.rpc_id(client_peer, "mp_to_bot", kind, payload)


func trace(text):
	_log(text)


func leave():
	_shutdown()
	Global.reload()  # back to a fresh main menu, as the game's own Quit does


# The game's own quit (pause menu, a disconnect) ended the multiplayer session: end ours with it.
func _on_multiplayer_stopped():
	if not leaving:
		_log("the game stopped multiplayer")
		_shutdown()


func _shutdown():
	if leaving:
		return
	leaving = true
	_stop_children()
	if enet != null and get_tree().network_peer == enet:
		Network.stop_multiplayer()
	enet = null
	if human:
		human.queue_free()
		human = null
	queue_free()


func _notification(what):
	if what == MainLoop.NOTIFICATION_WM_QUIT_REQUEST:  # the window is closing; the engine is still up
		_stop_children()
	elif what == NOTIFICATION_PREDELETE:  # teardown: no blocking calls here, just end the processes
		_stop_children()


func _stop_children():
	if life_conn != null:
		life_conn.disconnect_from_host()
		life_conn = null
	if life != null:
		life.stop()
	for p in [pid, client_pid]:
		if p > 0:
			OS.kill(p)  # no console window (a taskkill would flash one while the player is in the game)
	pid = -1
	client_pid = -1
	launcher_pid = -1


# The client copy's folder: the game's exe, pack and libraries next to an override.cfg that swaps the Steam
# autoload for a stub, adds MpClient.gd and gives the copy its own user folder (<this one's>-YomiBenchClient).
# Mods are off there (its modded.json), so neither this mod nor any other loads in it. Files are copied only when
# changed. Returns the folder's path, or "" on failure.
func _client_run_dir():
	var src = OS.get_executable_path().get_base_dir()
	var dest = ProjectSettings.globalize_path(CLIENT_DIR)
	var d = Directory.new()
	if d.make_dir_recursive(dest.plus_file("yomibench")) != OK:
		return ""
	var exe = OS.get_executable_path().get_file()
	var names = [exe, exe.get_basename() + ".pck"]
	if d.open(src) == OK:
		d.list_dir_begin(true, true)
		var n = d.get_next()
		while n != "":
			if not d.current_is_dir() and n.get_extension().to_lower() == "dll":
				names.append(n)
			n = d.get_next()
		d.list_dir_end()
	var f = File.new()
	for n in names:
		var a = src.plus_file(n)
		var b = dest.plus_file(n)
		if not f.file_exists(a):
			continue  # e.g. a build with its pack inside the exe
		if f.file_exists(b) and f.get_modified_time(a) <= f.get_modified_time(b) and _size(a) == _size(b):
			continue
		if d.copy(a, b) != OK:
			_log("couldn't copy %s" % a)
			return ""
	for s in CLIENT_SCRIPTS:
		if not _copy("res://YomiBenchVsBot".plus_file(s), dest.plus_file("yomibench").plus_file(s)):
			_log("couldn't write %s" % s)
			return ""
	var scripts = dest.plus_file("yomibench").replace("\\", "/")
	var user_name = OS.get_user_data_dir().get_file() + "-YomiBenchClient"
	if f.open(dest.plus_file("override.cfg"), File.WRITE) != OK:
		return ""
	f.store_string("[application]\nconfig/use_custom_user_dir=true\nconfig/custom_user_dir_name=\"%s\"\n\n" % user_name +
		"[autoload]\nSteamHustle=\"*%s/SteamStub.gd\"\n%s=\"*%s/MpClient.gd\"\n" % [scripts, NAME, scripts])
	f.close()
	var udir = OS.get_user_data_dir().get_base_dir().plus_file(user_name)
	d.make_dir_recursive(udir)
	if f.open(udir.plus_file("modded.json"), File.WRITE) == OK:
		f.store_string(JSON.print({"modsEnabled": false}))
		f.close()
	return dest


func _size(path):
	var f = File.new()
	if f.open(path, File.READ) != OK:
		return -1
	var n = f.get_len()
	f.close()
	return n


# Start the server without waiting for it. OS.execute(..., false) first; if this machine refuses that
# (ERR_CANT_FORK), PowerShell's Start-Process, through the blocking call the game itself uses, starts it hidden and
# prints its process id (Start-Process passes no handles on, so the call returns at once).
func _spawn(exe, args):
	var direct = OS.execute(exe, args, false)
	_log("OS.execute: %s" % direct)
	if direct > 0:
		return direct
	var quoted = []
	for a in args:
		quoted.append("'\"" + str(a).replace("'", "''").replace("\"", "\\\"") + "\"'")
	var ps = "$p = Start-Process -FilePath '%s' -ArgumentList @(%s) -WindowStyle Hidden -PassThru; $p.Id" % [
		str(exe).replace("'", "''"), PoolStringArray(quoted).join(",")]
	var out = []
	var code = OS.execute("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], true, out)
	var text = PoolStringArray(out).join("").strip_edges()
	_log("powershell exit %s: %s | output: %s" % [code, ps, text])
	return int(text) if text.is_valid_integer() else -1


func _server_command():
	var docs = OS.get_system_dir(OS.SYSTEM_DIR_DOCUMENTS)
	var out = docs.plus_file("YomiBench").plus_file("games-root")
	if docs == "":  # no Documents folder known: never a path relative to the game's folder
		out = ProjectSettings.globalize_path("user://yomibench-games")
	var f = File.new()
	if f.file_exists(CONFIG) and f.open(CONFIG, File.READ) == OK:
		var cfg = parse_json(f.get_as_text())
		f.close()
		if typeof(cfg) == TYPE_DICTIONARY and str(cfg.get("server", "")) != "":
			return {"exe": str(cfg.server), "args": Array(cfg.get("args", [])), "out": str(cfg.get("out", out)),
				"exit_with": bool(cfg.get("exit_with", false)), "app": bool(cfg.get("app", false))}
	var bundled = _bundled_server()
	if bundled != "":
		return {"exe": bundled, "args": [], "out": out, "exit_with": true, "app": true}
	var local = OS.get_environment("LOCALAPPDATA")
	if local != "":
		var exe = local.plus_file("Programs").plus_file("YomiBench").plus_file("server").plus_file("yomibench-bot-server.exe")
		if f.file_exists(exe):
			return {"exe": exe, "args": [], "out": out, "exit_with": true, "app": true}
	return {}


# The one-step mod carries the frozen server in its own pack (res://YomiBenchVsBot/server/, with manifest.json:
# each file's sha256 and size, and a version). A pack file can't be run, so on first use the files are copied to
# user://yomibench-server/<version>/ and every sha256 is checked; a ".complete" marker then lets later launches check
# only the server exe. A new mod version extracts to a new folder and the old ones are removed. Returns the exe's
# path, or "" when the mod carries no server or a check fails (then the installed app is tried).
const BUNDLE = "res://YomiBenchVsBot/server"
const EXTRACT_ROOT = "user://yomibench-server"


func _bundled_server():
	var f = File.new()
	if not f.file_exists(BUNDLE + "/manifest.json") or f.open(BUNDLE + "/manifest.json", File.READ) != OK:
		return ""
	var man = parse_json(f.get_as_text())
	f.close()
	if typeof(man) != TYPE_DICTIONARY or not man.has("files"):
		return ""
	var dest = EXTRACT_ROOT.plus_file(str(man.version))
	var exe_entry = null
	for e in man.files:
		if e.path == man.exe:
			exe_entry = e
	if exe_entry == null:
		return ""
	var exe = dest.plus_file(man.exe)
	if f.file_exists(dest.plus_file(".complete")) and f.file_exists(exe) and f.get_sha256(exe) == exe_entry.sha256:
		return ProjectSettings.globalize_path(exe)
	var t0 = OS.get_ticks_msec()
	var d = Directory.new()
	for e in man.files:
		var target = dest.plus_file(e.path)
		if f.file_exists(target) and f.get_sha256(target) == e.sha256:
			continue
		d.make_dir_recursive(target.get_base_dir())
		if not _copy(BUNDLE.plus_file(e.path), target) or f.get_sha256(target) != e.sha256:
			_log("bundled server: %s failed its sha256 check after copying" % e.path)
			return ""
	if f.open(dest.plus_file(".complete"), File.WRITE) == OK:
		f.store_string(str(man.version))
		f.close()
	_log("bundled server %s extracted (%d files) in %d ms" % [man.version, man.files.size(), OS.get_ticks_msec() - t0])
	_remove_other_versions(str(man.version))
	return ProjectSettings.globalize_path(exe)


func _copy(src, dst):
	var a = File.new()
	var b = File.new()
	if a.open(src, File.READ) != OK:
		return false
	if b.open(dst, File.WRITE) != OK:
		a.close()
		return false
	var left = a.get_len()
	while left > 0:
		var n = min(left, 4 << 20)
		b.store_buffer(a.get_buffer(n))
		left -= n
	a.close()
	b.close()
	return true


func _remove_other_versions(keep):
	var d = Directory.new()
	if d.open(EXTRACT_ROOT) != OK:
		return
	d.list_dir_begin(true, true)
	var name = d.get_next()
	while name != "":
		if d.current_is_dir() and name != keep:
			_remove_tree(EXTRACT_ROOT.plus_file(name))
		name = d.get_next()
	d.list_dir_end()


func _remove_tree(path):
	var d = Directory.new()
	if d.open(path) != OK:
		return
	d.list_dir_begin(true, false)
	var name = d.get_next()
	while name != "":
		var p = path.plus_file(name)
		if d.current_is_dir():
			_remove_tree(p)
		else:
			d.remove(p)
		name = d.get_next()
	d.list_dir_end()
	Directory.new().remove(path)


func _free_port():
	var probe = TCP_Server.new()
	for _i in range(50):
		var p = 7900 + randi() % 100
		if probe.listen(p, "127.0.0.1") == OK:
			probe.stop()
			return p
	return 7999


# One line per event in user://yomibench_vsbot.log (the game's stdout is buffered when redirected).
func _log(text):
	var f = File.new()
	var mode = File.READ_WRITE if f.file_exists("user://yomibench_vsbot.log") else File.WRITE
	if f.open("user://yomibench_vsbot.log", mode) == OK:
		f.seek_end()
		f.store_line("%s %s" % [OS.get_datetime_from_unix_time(OS.get_unix_time()), text])
		f.close()


func _message(text):
	_log(text)
	var d = AcceptDialog.new()
	d.dialog_text = text
	d.window_title = "YomiBench"
	get_tree().root.add_child(d)
	d.popup_centered()
	queue_free()
