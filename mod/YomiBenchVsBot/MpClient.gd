extends Node
# YomiBench "Multiplayer (vs bot)": the bot's side, in a headless copy of the player's game (VsBot.gd builds the
# copy's folder and starts it with --no-window --disable-render-loop --audio-driver Dummy). An autoload named
# YomiBenchVsBot through that folder's override.cfg, so it sits at /root/YomiBenchVsBot like the host's VsBot node,
# and the two exchange our few RPCs through their MpLink.gd children. Without --yomibench-mp-client it does nothing.
#
# It joins the host's direct-connect game on 127.0.0.1 (Network.join_game_direct) as P2 and runs YomiHuman.gd as
# side "client": the bot server connection, the per-turn sync check, and P2's picks, submitted with this game's own
# P2 panel (Network.submit_action, then the game's send_action once both are ready). It quits when the host goes:
# the lifeline TCP connection to the host closes (at once, even when the host is killed), the ENet session ends,
# or the bot server goes away (the server stops with the game: --exit-with).

const ARG = "--yomibench-mp-client="

var cfg = {}
var life = null
var life_up = false
var boot_msec = 0
var human = null
var joined = false
var link = null          # MpLink.gd at /root/YomiBenchVsBot/MpLink, as in the host
var outbox = []          # messages for the host from before the session was up (e.g. the server's hello)


func _ready():
	var raw = ""
	for a in OS.get_cmdline_args():
		if a.begins_with(ARG):
			raw = a.substr(ARG.length())
	if raw == "":
		set_process(false)
		return
	var parsed = JSON.parse(Marshalls.base64_to_utf8(raw))
	if parsed.error != OK or typeof(parsed.result) != TYPE_DICTIONARY:
		_quit("bad --yomibench-mp-client")
		return
	cfg = parsed.result
	link = load(get_script().resource_path.get_base_dir().plus_file("MpLink.gd")).new()
	link.name = "MpLink"
	link.target = self
	add_child(link)
	pause_mode = PAUSE_MODE_PROCESS
	Engine.target_fps = 60  # nothing is drawn: don't spin a core between physics ticks
	boot_msec = OS.get_ticks_msec()
	life = StreamPeerTCP.new()
	life.connect_to_host("127.0.0.1", int(cfg.life))
	call_deferred("_boot")


func _boot():
	while true:  # the game's main scene and its character loading first (as YomiHuman's own boot)
		yield(get_tree(), "idle_frame")
		var s = get_tree().current_scene
		if s != null and s.has_method("_on_match_ready") and Global.mods_loaded:
			break
	# The same version string as the host (the game refuses a peer whose version differs; the host's carries its
	# mods' hash, and this copy runs none).
	Global.VERSION = Marshalls.base64_to_variant(str(cfg.version))
	get_tree().connect("connected_to_server", self, "_on_connected")
	get_tree().connect("server_disconnected", self, "_quit", ["the host's game closed the session"])
	get_tree().connect("connection_failed", self, "_quit", ["couldn't reach the host's game"])
	Network.join_game_direct("127.0.0.1", int(cfg.enet), "YomiBench bot")
	human = load(get_script().resource_path.get_base_dir().plus_file("YomiHuman.gd")).new()
	human.name = "YomiHuman"
	add_child(human)
	human.start(int(cfg.bot), "127.0.0.1", false, true, "client", self)


func _process(_delta):
	if life == null:
		return
	var st = life.get_status()
	if st == StreamPeerTCP.STATUS_CONNECTED:
		life_up = true
	elif life_up or st == StreamPeerTCP.STATUS_ERROR or (st == StreamPeerTCP.STATUS_NONE and OS.get_ticks_msec() - boot_msec > 5000):
		_quit("the host's game is gone")
	elif not life_up and OS.get_ticks_msec() - boot_msec > 30000:
		_quit("never reached the host's game")


func _on_connected():
	joined = true
	link.rpc_id(1, "mp_hello", str(cfg.token))
	for m in outbox:  # reliable and in order, after the token
		link.rpc_id(1, "mp_from_bot", m[0], m[1])
	outbox.clear()


func to_host(kind, payload):
	if joined:
		link.rpc_id(1, "mp_from_bot", kind, payload)
	else:
		outbox.append([kind, payload])


func trace(text):
	to_host("trace", text)


# Host -> client: start a match (the panel's choices), or abort the current one.
func on_mp_to_bot(sender, kind, payload):
	if sender != 1 or human == null:
		return
	match str(kind):
		"match":
			human.start_bot_match(payload)
		"abort":
			human._abort(str(payload))


func bot_lost():
	_quit("lost the bot server")


# YomiHuman (client side) gave up on the bot server: the host shows why, then this copy quits (after a moment, so
# the message leaves first; if the host never joined, its own deadlines say it).
func bot_unreachable(text):
	to_host("bot_unreachable", text)
	yield(get_tree().create_timer(3.0), "timeout")
	_quit(text)


func _quit(why):
	print("YomiBench client: ", why)
	set_process(false)
	get_tree().quit()
