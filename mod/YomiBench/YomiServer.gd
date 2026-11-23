extends Node
# YomiBench engine server: docs/PROTOCOL.md (JSON lines) over TCP on 127.0.0.1.
#
# Loaded as an autoload through override.cfg next to a scratch copy of the game exe
# (see src/yomibench/game.py). Options, as command-line args or environment variables:
#   --yomi-port=N            YOMI_PORT             (default 7777)
#   --yomi-max-candidates=N  YOMI_MAX_CANDIDATES   (default 30, per seat)
#   --yomi-copy-mode=MODE    YOMI_COPY_MODE        dirty (default) | full: which states a preview copies
#   --yomi-divergence-log=P  YOMI_DIVERGENCE_LOG   append one JSON line per turn whose preview of the
#                                                  played pair differed from the real step
#   --yomi-log=PATH          YOMI_LOG              append startup messages and errors to PATH
#   --yomi-candidate-set=S   YOMI_CANDIDATE_SET    4 (default) or 5: the set hello reports and
#                                                  new_match plays when it names none
#   --yomi-set5=SWITCHES     YOMI_SET5             set 5's switches: "all" (default), "none", or the
#                                                  ones on, comma-separated (draw_pairs,gun_aim_fine,
#                                                  bullet_di,draw_base)
#   --yomi-set6=SWITCHES     YOMI_SET6             set 6's switches: "default" (pairs and clicks on for every
#                                                  character, triples and toggle_only off), or the ones
#                                                  changed from it, comma-separated, each name=0|1
#                                                  (pairs_<c>, clicks_<c>, toggle_only_<c>, triples)
#   --yomi-candidate-rules=H YOMI_CANDIDATE_RULES  a candidate_rules hash (set 4's or a set-5
#                                                  combination's): picks the set and switches it names,
#                                                  so a recorded game's server can be started from its
#                                                  header (overrides the two above)
#   --yomi-legality=MODE     YOMI_LEGALITY         copy (default) | game: which moves are legal comes
#                                                  from LegalityCopy.gd (a copy of the game's
#                                                  ActionButtons.update_buttons loop) or from a hidden
#                                                  instance of the game's own ActionButtons panel
#                                                  (Legality.gd); same answers. Without LegalityCopy.gd
#                                                  the default is game
#   --yomi-legality-check=P  YOMI_LEGALITY_CHECK   answer every legality question both ways, use the
#                                                  selected mode's, and append any difference to P
#                                                  (counts in `stats.legality`)
#
# Beyond the protocol: `stats` (timings, JSON cost per command, divergence counters),
# `copy_diff` (fighter variables a preview copy still misses), `deep_diff` (every script variable
# and body position that differs after a preview copy: game, fighters, all states and hitboxes,
# objects), `trace: true` on preview/step (a per-tick snapshot), and for move-coverage testing
# `moves` (every menu move, legality, widgets) and `probe` (candidates at chosen widget values).
# `step_raw` plays each seat's exact input ({action, data, extra}, on or off the candidate grid),
# refusing inputs a player could not have sent (YomiMatch.step_raw).
# `raw_option_by_id: true` on step_raw checks option widgets' data against the selected item's
# panel output instead of the default item's (off by default; YomiMatch.step_raw).
# `action_space` describes everything each seat can choose now, read off the game's own action
# menu: legal moves, each move's widgets and ranges, reverse / feint / DI, the extra panel's
# controls and which toggle reveals each (ActionSpace.gd; read-only).

const ENGINE_VERSION = "2"  # 2: standard online settings, timing and raw state
# Bumped whenever generate() can return a different candidate list for the same position, so
# recorded games say which move list their candidate ids index.
# 3: feint, reverse, extra toggles, dash timing steps; 4: every variant, factors + expand;
# 5: set 4 plus four Cowboy additions, each behind a switch (Candidates.gd, README "Candidate set 5")
var CANDIDATE_SET = "4"   # this server's default set (--yomi-candidate-set)
const CANDIDATE_SETS = ["6", "5", "4", "2"]
const SET5_KEYS = ["draw_pairs", "gun_aim_fine", "bullet_di", "draw_base"]  # bit k of a combination
var set5 = {}             # set 5's switches, every set-5 seat on this server
const SET6_KEYS = ["pairs_ninja", "pairs_cowboy", "pairs_wizard", "pairs_mutant", "pairs_robot", "triples",
	"toggle_only_ninja", "toggle_only_cowboy", "toggle_only_wizard", "toggle_only_mutant", "toggle_only_robot",
	"clicks_ninja", "clicks_cowboy", "clicks_wizard", "clicks_mutant", "clicks_robot"]
var set6 = {}             # set 6's switches, every set-6 seat on this server
var rules_tokens = {}     # rules file name -> its tokens (_rules_text), for the set-6 hashes computed on demand
var rules_alias = {}      # rules_aliases.json: computed hash -> the frozen hash it stands for
var rules_hashes = {}     # "4" and "5:<bits>" -> candidate_rules

var port = 7777
var candidate_rules = ""  # hash of the default set's rules (hello, records)
var max_candidates = 0  # per-seat budget; 0: every variant (candidate set 4's default)
var match_settings = {}
var mod_md5 = ""   # fingerprint of these scripts, so a pool can refuse servers running other code
var copy_mode = "dirty"
var divergence_log = ""
var legality = ""          # "copy" or "game" (--yomi-legality)
var legality_check = ""    # parity log path (--yomi-legality-check)
var legality_counts = {"calls": 0, "mismatches": 0, "recent": []}   # across matches
var cmd_stats = {}   # cmd -> {n, handle_us, json_us, bytes}
var divergence = {"compared": 0, "diverged": 0, "fields": {}, "recent": []}   # across matches

var server = null
var clients = []
const IDLE_TAKEOVER_MS = 300000
var dir = ""
var MatchScript = null
var match_ = null
var job = null       # a preview batch in progress, run in slices across frames


func _ready():
	pause_mode = PAUSE_MODE_PROCESS
	dir = get_script().resource_path.get_base_dir()
	port = int(_option("port", str(port)))
	max_candidates = int(_option("max-candidates", str(max_candidates)))
	copy_mode = _option("copy-mode", copy_mode)
	divergence_log = _option("divergence-log", "")
	legality_check = _option("legality-check", "")
	set_process(false)
	call_deferred("_boot")


func _log(msg):
	print(msg)
	var path = _option("log", "")
	if path == "":
		return
	var f = File.new()
	var mode = File.READ_WRITE if f.file_exists(path) else File.WRITE
	if f.open(path, mode) == OK:
		f.seek_end()
		f.store_line(str(msg))
		f.close()


func _option(name, default):
	var flag = "--yomi-%s=" % name
	for a in OS.get_cmdline_args():
		if a.begins_with(flag):
			return a.substr(flag.length())
	var env = OS.get_environment("YOMI_" + name.to_upper().replace("-", "_"))
	return env if env != "" else default


func _boot():
	# Let the game's own autoloads and main scene finish starting, then drop the main menu.
	for _i in range(5):
		yield(get_tree(), "idle_frame")
	var scene = get_tree().current_scene
	if scene:
		scene.queue_free()
	yield(get_tree(), "idle_frame")
	MatchScript = load(dir + "/YomiMatch.gd")
	match_settings = _standard_settings()
	# Over the text with line endings normalized: a Windows checkout has CRLF, the same code elsewhere LF.
	var f = File.new()
	for name in ["YomiServer.gd", "YomiMatch.gd", "Candidates.gd", "Candidates4.gd", "Candidates2.gd",
			"SteamStub.gd", "candidate_rules_v4.json", "candidate_rules_v5.json", "Legality.gd", "LegalityCopy.gd",
			"rules_aliases.json", "ActionSpace.gd", "candidate_rules_v6.json"]:
		if f.open(dir + "/" + name, File.READ) == OK:
			mod_md5 += f.get_as_text().replace("\r", "").md5_text().substr(0, 8)
			f.close()
	# Rules versions: a rules file and set 4's frozen generator reduced to their tokens (comments and
	# whitespace outside strings removed; tools/rules_hash.py is the same function). Set 5's is the v5
	# file with its `set5` switches as played, so each combination has its own hash.
	var tokens = {}
	for name in ["candidate_rules_v4.json", "candidate_rules_v5.json", "candidate_rules_v6.json", "Candidates4.gd"]:
		if f.open(dir + "/" + name, File.READ) == OK:
			tokens[name] = _rules_text(f.get_as_text(), name.ends_with(".gd"))
			f.close()
	rules_tokens = tokens
	rules_hashes["4"] = (tokens["candidate_rules_v4.json"] + tokens["Candidates4.gd"]).md5_text().substr(0, 12)
	for b in range(16):
		var t = tokens["candidate_rules_v5.json"]
		for k in range(4):
			t = t.replace('"%s":true' % SET5_KEYS[k], '"%s":%s' % [SET5_KEYS[k], "true" if b & (1 << k) else "false"])
		rules_hashes["5:%d" % b] = (t + tokens["Candidates4.gd"]).md5_text().substr(0, 12)
	# rules_aliases.json: computed hash -> the frozen hash it stands for, for file changes that
	# change no list (Candidates4.gd's legality loop moved out; tools/rules_hash.py does the same).
	if f.open(dir + "/rules_aliases.json", File.READ) == OK:
		var al = JSON.parse(f.get_as_text()).result
		f.close()
		if typeof(al) == TYPE_DICTIONARY and typeof(al.get("aliases")) == TYPE_DICTIONARY:
			rules_alias = al.aliases
			for k in rules_hashes:
				rules_hashes[k] = al.aliases.get(rules_hashes[k], rules_hashes[k])
	var err_set = _pick_set()
	if err_set == "":
		err_set = _pick_legality()
	if err_set != "":
		_log("YOMIBENCH_ERROR " + err_set)
		get_tree().quit(4)
		return
	if CANDIDATE_SET == "4":
		candidate_rules = rules_hashes["4"]
	elif CANDIDATE_SET == "6":
		candidate_rules = set6_hash(set5, set6)
	else:
		candidate_rules = rules_hashes["5:%d" % _set5_bits()]
	if CANDIDATE_SET in ["5", "6"]:
		mod_md5 += candidate_rules.substr(0, 8)  # a fingerprint per switch combination
	if match_settings.empty():
		_log("YOMIBENCH_ERROR could not read the game's default match settings")
		get_tree().quit(3)
		return
	server = TCP_Server.new()
	var err = server.listen(port, _option("host", "127.0.0.1"))  # "*" inside a container
	if err != OK:
		_log("YOMIBENCH_ERROR listen failed on port %d: %d" % [port, err])
		get_tree().quit(2)
		return
	_log("YOMIBENCH_READY port=%d game=%s" % [port, Global.VERSION])
	set_process(true)


# The default set and set 5's switches from the options; "" or why they can't be used.
func _pick_set():
	for k in SET5_KEYS:
		set5[k] = true
	var spec = _option("set5", "all")
	if spec != "all":
		for k in SET5_KEYS:
			set5[k] = false
		if spec != "none":
			for k in spec.split(","):
				if not k.strip_edges() in SET5_KEYS:
					return "unknown set-5 switch %s (have %s)" % [k, str(SET5_KEYS)]
				set5[k.strip_edges()] = true
	set6 = _set6_default()
	var spec6 = _option("set6", "default")
	if spec6 != "default":
		for kv in spec6.split(","):
			var parts = kv.strip_edges().split("=")
			if parts.size() != 2 or not parts[0] in SET6_KEYS or not parts[1] in ["0", "1"]:
				return "set-6 switches are name=0|1, comma-separated (have %s), not %s" % [str(SET6_KEYS), kv]
			set6[parts[0]] = parts[1] == "1"
	CANDIDATE_SET = _option("candidate-set", CANDIDATE_SET)
	if not CANDIDATE_SET in ["4", "5", "6"]:
		return "candidate set must be 4, 5 or 6, not %s" % CANDIDATE_SET
	var want = _option("candidate-rules", "")
	if want != "":
		if want == rules_hashes["4"]:
			CANDIDATE_SET = "4"
			return ""
		for b in range(16):
			if want == rules_hashes["5:%d" % b]:
				CANDIDATE_SET = "5"
				for k in range(4):
					set5[SET5_KEYS[k]] = b & (1 << k) != 0
				return ""
		# Set 6: set 5's combinations with set 6's default switches (a recorded game's header names one).
		for b in range(16):
			var s5 = {}
			for k in range(4):
				s5[SET5_KEYS[k]] = b & (1 << k) != 0
			if want == set6_hash(s5, _set6_default()):
				CANDIDATE_SET = "6"
				set5 = s5
				set6 = _set6_default()
				return ""
		return "no candidate set here has rules %s" % want
	return ""


# The legality mode from the options; "" or why it can't be used. LegalityCopy.gd is optional: a
# package without it runs in game mode.
func _pick_legality():
	var have_copy = File.new().file_exists(dir + "/LegalityCopy.gd")
	legality = _option("legality", "copy" if have_copy else "game")
	if not legality in ["copy", "game"]:
		return "legality must be copy or game, not %s" % legality
	if (legality == "copy" or legality_check != "") and not have_copy:
		return "LegalityCopy.gd is not installed: only --yomi-legality=game is available (no check)"
	return ""


# Set 6 as released: every character's pairs and single clicks on, triples and toggle_only off.
func _set6_default():
	var d = {}
	for k in SET6_KEYS:
		d[k] = k.begins_with("pairs_") or k.begins_with("clicks_")
	return d


# hello.candidate_rules for set 6 with these switches: the v6 rules file with every set-5 and set-6 switch's
# literal as played, then set 4's frozen generator, reduced to tokens (as set 5's; tools/rules_hash.py).
func set6_hash(s5, s6):
	var t = rules_tokens["candidate_rules_v6.json"]
	for k in SET5_KEYS:
		t = t.replace('"%s":true' % k, '"%s":%s' % [k, "true" if s5[k] else "false"])
	for k in SET6_KEYS:
		t = t.replace('"%s":true' % k, '"%s":%s' % [k, "true" if s6[k] else "false"])
	var h = (t + rules_tokens["Candidates4.gd"]).md5_text().substr(0, 12)
	return rules_alias.get(h, h)


func _set5_bits():
	var b = 0
	for k in range(4):
		if set5[SET5_KEYS[k]]:
			b |= 1 << k
	return b


# Match settings of a standard online lobby, read from the game's own settings panel: the
# scene's defaults (GameSettingsPanelContainer.scene_defaults), with Sadness on, as
# update_multiplayer() forces for online play, and no turn timer (bots don't play on a clock).
# new_match may override individual keys.
func _standard_settings():
	var scene = load("res://ui/GameSettingsPanelContainer.tscn")
	if scene == null:
		return {}
	var panel = scene.instance()
	add_child(panel)
	var s = {}
	if typeof(panel.get("scene_defaults")) == TYPE_DICTIONARY:
		s = panel.scene_defaults.duplicate(true)
	panel.queue_free()
	if s.empty():
		return s
	s["sadness_enabled"] = true
	s["timer_mode"] = "none"
	s["p2_dummy"] = false
	return s


func _process(_delta):
	# Poll tightly for a short while so request latency stays well under a frame, then return
	# so the engine can flush deferred calls and frees. Preview batches run in slices for the
	# same reason: the game queues frees and particle spawns for the end of the frame, and a
	# batch of hundreds of previews in one frame piles up enough of them to exhaust Godot's
	# memory pools.
	var deadline = OS.get_ticks_msec() + 20
	while OS.get_ticks_msec() < deadline:
		if job != null:
			_run_job(deadline)
			return
		if not _poll():
			OS.delay_usec(200)


func _run_job(deadline):
	while job.i < job.pairs.size() and OS.get_ticks_msec() < deadline:
		job.results.append(match_.preview_one(job.pairs[job.i], job.detail, job.want, job.trace))
		job.i += 1
	if job.i < job.pairs.size():
		return
	var j = job
	job = null
	_send(j.peer, {"results": j.results}, j.cmd, j.t0)


func _send(peer, resp, cmd, t0):
	var t1 = OS.get_ticks_usec()
	var out = (JSON.print(resp) + char(10)).to_utf8()
	var t2 = OS.get_ticks_usec()
	peer.put_data(out)
	var st = cmd_stats.get(cmd, {"n": 0, "handle_us": 0, "json_us": 0, "bytes": 0})
	st.n += 1
	st.handle_us += t1 - t0
	st.json_us += t2 - t1
	st.bytes += out.size()
	cmd_stats[cmd] = st


func _poll():
	var worked = false
	for c in clients.duplicate():
		if c.peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			clients.erase(c)
	while server.is_connection_available():
		var peer = server.take_connection()
		peer.set_no_delay(true)
		# One server runs one match, so a second client would silently replace the first one's
		# match (two machines sharing a pool, or a test pointed at a busy server). Refuse it, unless
		# the current client has gone silent (a client machine that died can leave a half-open
		# connection behind that would otherwise hold the server forever).
		if not clients.empty() and OS.get_ticks_msec() - clients[0].last > IDLE_TAKEOVER_MS:
			clients[0].peer.disconnect_from_host()
			clients.clear()
		if not clients.empty():
			peer.put_data((JSON.print({"error": {"code": "busy", "message": "this server already has a client"}}) + "\n").to_utf8())
			peer.disconnect_from_host()
			continue
		clients.append({"peer": peer, "buf": PoolByteArray(), "last": OS.get_ticks_msec()})
	for c in clients.duplicate():
		var peer = c.peer
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			clients.erase(c)
			continue
		var n = peer.get_available_bytes()
		if n > 0:
			c.last = OS.get_ticks_msec()
			var got = peer.get_partial_data(n)
			if got[0] == OK:
				# Pool arrays are values: append to a local and store it back.
				var buf = c.buf
				buf.append_array(got[1])
				c.buf = buf
		while true:
			# Requests are ASCII JSON, so byte and character offsets agree.
			var text = c.buf.get_string_from_utf8()
			var nl = text.find(char(10))
			if nl < 0:
				break
			var line = text.substr(0, nl)
			c.buf = text.substr(nl + 1).to_utf8()
			if line.strip_edges() == "":
				continue
			var t0 = OS.get_ticks_usec()
			var cmd = [""]
			var resp = _handle_line(line, cmd)
			worked = true
			if resp.has("_job"):
				job = resp
				job.peer = peer
				job.cmd = cmd[0]
				job.t0 = t0
				return true   # the rest of this client's lines wait for the batch
			_send(peer, resp, cmd[0], t0)
	return worked


# Debug: descendant counts per node path, to find what accumulates.
func _census(node, depth, max_depth):
	var out = {}
	for c in node.get_children():
		var n = _count(c)
		if n >= 50:
			var key = str(c.name) + ":" + c.get_class()
			out[key] = {"n": n}
			if depth < max_depth:
				out[key]["children"] = _census(c, depth + 1, max_depth)
	return out


func _count(node):
	var n = 1
	for c in node.get_children():
		n += _count(c)
	return n


func _err(code, message):
	return {"error": {"code": code, "message": message}}


func _handle_line(line, cmd_out):
	var parsed = JSON.parse(line)
	if parsed.error != OK or typeof(parsed.result) != TYPE_DICTIONARY:
		return _err("bad_request", "not a JSON object")
	var msg = parsed.result
	var cmd = msg.get("cmd")
	cmd_out[0] = str(cmd)
	var detail = msg.get("detail") == true
	var want_cands = msg.get("candidates", true) != false
	if cmd == "hello":
		return {
			"engine": "yomi", "version": ENGINE_VERSION, "game_version": Global.VERSION,
			"characters": Global.name_paths.keys(), "max_candidates": max_candidates,
			"candidate_set": CANDIDATE_SET, "candidate_rules": candidate_rules,
			# Set 5's switches and their hash: what a seat on set "5" plays (new_match candidate_sets).
			"set5": set5, "set5_rules": rules_hashes["5:%d" % _set5_bits()],
			"set6": set6, "set6_rules": set6_hash(set5, set6),
			"copy_mode": copy_mode, "match_settings": match_settings, "mod_md5": mod_md5,
			"legality": legality, "legality_check": legality_check != "",
		}
	if cmd == "new_match":
		return _new_match(msg)
	if cmd == "quit":
		get_tree().quit()
		return {"ok": true}
	if cmd == "stats":
		var out = {"commands": cmd_stats, "engine": {
			"objects": Performance.get_monitor(Performance.OBJECT_COUNT),
			"nodes": Performance.get_monitor(Performance.OBJECT_NODE_COUNT),
			"orphan_nodes": Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT),
			"resources": Performance.get_monitor(Performance.OBJECT_RESOURCE_COUNT),
			"static_memory_mb": Performance.get_monitor(Performance.MEMORY_STATIC) / 1048576.0,
		}}
		if match_ != null:
			out["match"] = match_.stats
		out["divergence"] = divergence
		out["legality"] = {"mode": legality, "check": legality_check != "", "calls": legality_counts.calls,
			"mismatches": legality_counts.mismatches, "recent": legality_counts.recent}
		out["rules_hashes"] = rules_hashes  # "4" and "5:<switch bits>" (bit k: SET5_KEYS[k])
		return out
	if match_ == null:
		return _err("no_match", "send new_match first")
	if cmd == "census":
		return {"census": _census(get_tree().root, 0, 4)}
	if cmd == "copy_diff":
		return {"diff": match_.copy_diff()}
	if cmd == "deep_diff":
		return {"diff": match_.deep_diff(int(msg.get("limit", 300)))}
	if cmd == "save_replay":
		var path = msg.get("path")
		if path != null and (typeof(path) != TYPE_STRING or not path.begins_with("user://") or ".." in path):
			return _err("bad_request", "path must be under user:// (or omitted to get the bytes back)")
		return match_.save_replay(path)
	if cmd == "observe":
		return match_.observation(detail)
	if cmd == "action_space":
		if match_.done():
			return _err("match_over", "the match has ended")
		return {"seats": match_.action_space()}
	if cmd == "moves":
		if match_.done():
			return _err("match_over", "the match has ended")
		return {"seats": [match_.describe(0), match_.describe(1)]}
	if cmd == "expand":
		# Candidate set 4: candidates with factors (DI, feint, reverse, extra set) overridden,
		# appended to this decision (YomiMatch.expand).
		var specs = msg.get("specs")
		if typeof(specs) != TYPE_ARRAY:
			return _err("bad_request", "specs must be a list of {seat, candidate, extra_override}")
		if match_.done():
			return _err("match_over", "the match has ended")
		var out = []
		for sp in specs:
			if typeof(sp) != TYPE_DICTIONARY or not (typeof(sp.get("seat")) in [TYPE_INT, TYPE_REAL]) or not (sp.get("seat") in [0, 1]):
				out.append({"error": "each spec needs seat 0 or 1"})
				continue
			out.append(match_.expand(int(sp.seat), sp))
		return {"results": out}
	if cmd == "probe":
		# Debug: candidates at chosen widget values (Candidates.probe), appended to this decision.
		var probes = msg.get("probes")
		if typeof(probes) != TYPE_ARRAY:
			return _err("bad_request", "probes must be a list of {seat, action, widgets, extra, di, ...}")
		if match_.done():
			return _err("match_over", "the match has ended")
		var out = []
		for p in probes:
			if typeof(p) != TYPE_DICTIONARY or not (p.get("seat") in [0, 1, 0.0, 1.0]):
				out.append({"error": "each probe needs seat 0 or 1"})
				continue
			for key in ["widgets", "extra", "di"]:
				if p.has(key) and typeof(p[key]) != TYPE_DICTIONARY:
					p.erase(key)
			out.append(match_.probe(int(p.seat), p))
		return {"results": out}
	if cmd == "step":
		var ids = _parse_pair(msg.get("actions"), true)
		if typeof(ids) == TYPE_STRING:
			return _err("bad_request", ids)
		if match_.done():
			return _err("match_over", "the match has ended")
		match_.step(ids[0], ids[1], msg.get("trace") == true)
		var sobs = match_.observation(detail)
		if msg.get("trace") == true:
			sobs["trace"] = match_.last_trace
		return sobs
	if cmd == "step_raw":
		if match_.done():
			return _err("match_over", "the match has ended")
		var why = match_.step_raw(msg.get("inputs"), msg.get("raw_option_by_id") == true)
		if why != "":
			return _err("bad_request", why)
		return match_.observation(detail)
	if cmd == "preview":
		var pairs = msg.get("pairs")
		if typeof(pairs) != TYPE_ARRAY:
			return _err("bad_request", "pairs must be a list")
		if match_.done():
			return _err("match_over", "the match has ended")
		var parsed_pairs = []
		for p in pairs:
			var ids = _parse_pair(p, true)
			if typeof(ids) == TYPE_STRING:
				return _err("bad_request", ids)
			parsed_pairs.append(ids)
		return {"_job": true, "pairs": parsed_pairs, "i": 0, "results": [], "detail": detail, "want": want_cands,
			"trace": msg.get("trace") == true}
	if cmd == "continue":
		var h = msg.get("handle")
		if typeof(h) != TYPE_REAL and typeof(h) != TYPE_INT:
			return _err("bad_request", "handle must be a number")
		var pair = null
		if msg.get("pair") != null:
			pair = _parse_pair(msg.get("pair"), false)
			if typeof(pair) == TYPE_STRING:
				return _err("bad_request", pair)
		var until = msg.get("until", "decision")
		var min_ticks = 0
		if typeof(until) == TYPE_DICTIONARY and (typeof(until.get("ticks")) == TYPE_REAL or typeof(until.get("ticks")) == TYPE_INT) and until.ticks >= 0:
			min_ticks = int(until.ticks)
		elif until != "decision":
			return _err("bad_request", 'until must be "decision" or {"ticks": N}')
		return match_.cont(int(h), pair, min_ticks, detail, want_cands)
	if cmd == "release":
		var hs = msg.get("handles")
		if typeof(hs) != TYPE_ARRAY:
			return _err("bad_request", "handles must be a list")
		var ints = []
		for h in hs:
			if typeof(h) == TYPE_REAL or typeof(h) == TYPE_INT:
				ints.append(int(h))
		return {"released": match_.release(ints)}
	return _err("unknown_cmd", str(cmd))


# Parses [id_p1, id_p2]; with check_real, against the real match's current candidates (continue
# checks against its handle's list). Returns [int, int] or an error string.
func _parse_pair(p, check_real):
	if typeof(p) != TYPE_ARRAY or p.size() != 2:
		return "expected [id_p1, id_p2]"
	var out = []
	for seat in range(2):
		var v = p[seat]
		if typeof(v) != TYPE_REAL and typeof(v) != TYPE_INT:
			return "seat %d: candidate id must be a number" % seat
		if float(v) != floor(float(v)):
			return "seat %d: candidate id must be an integer" % seat
		var i = int(v)
		if check_real and (i < 0 or i >= match_.candidate_count(seat)):
			return "seat %d: no candidate %d" % [seat, i]
		out.append(i)
	return out


func _new_match(msg):
	var p1 = msg.get("p1")
	var p2 = msg.get("p2")
	for name in [p1, p2]:
		if typeof(name) != TYPE_STRING or not Global.name_paths.has(name):
			return _err("bad_character", "unknown character %s; have %s" % [str(name), str(Global.name_paths.keys())])
	var seed_ = msg.get("seed", 0)
	if typeof(seed_) != TYPE_REAL and typeof(seed_) != TYPE_INT:
		return _err("bad_request", "seed must be a number")
	var cap = msg.get("max_candidates", max_candidates)
	if typeof(cap) != TYPE_REAL and typeof(cap) != TYPE_INT:
		return _err("bad_request", "max_candidates must be a number")
	var settings = match_settings.duplicate(true)
	var overrides = msg.get("settings", {})
	if typeof(overrides) != TYPE_DICTIONARY:
		return _err("bad_request", "settings must be an object")
	for k in overrides:
		if not settings.has(k):
			return _err("bad_request", "unknown setting %s" % str(k))
		settings[k] = overrides[k]
	var sets = msg.get("candidate_sets", [CANDIDATE_SET, CANDIDATE_SET])
	if typeof(sets) != TYPE_ARRAY or sets.size() != 2:
		return _err("bad_request", "candidate_sets must be [p1 set, p2 set]")
	for i in range(2):
		if sets[i] == null:
			sets[i] = CANDIDATE_SET
		elif typeof(sets[i]) in [TYPE_INT, TYPE_REAL]:
			if int(sets[i]) != sets[i]:
				return _err("bad_request", "candidate sets must be integers or strings")
			sets[i] = str(int(sets[i]))
		elif typeof(sets[i]) != TYPE_STRING:
			return _err("bad_request", "candidate sets must be integers or strings")
		if not sets[i] in CANDIDATE_SETS:
			return _err("bad_request", "candidate sets available: 6, 5, 4 and 2")
	# Validate the whole request before replacing the live match.
	var next_handle = 1  # handle ids never repeat across matches, so an old one is stale, not reused
	if match_ != null:
		next_handle = match_.next_handle
		match_.queue_free()
		match_ = null
	match_ = MatchScript.new()
	match_.next_handle = next_handle
	match_.legality_mode = legality
	match_.legality_check = legality_check
	match_.legality_counts = legality_counts
	add_child(match_)
	match_.start(p1, p2, int(seed_), int(cap), settings, copy_mode, divergence_log, divergence, sets, set5, set6)
	var obs = match_.observation(msg.get("detail") == true)
	obs.state["raw_start"] = match_.raw_start  # full start values; later states send changes only
	return obs


func _rules_text(text, comments):
	var out = PoolStringArray()
	var quote = ""
	var i = 0
	var n = text.length()
	while i < n:
		var c = text[i]
		if quote != "":
			out.append(c)
			if c == "\\" and i + 1 < n:
				out.append(text[i + 1])
				i += 1
			elif c == quote:
				quote = ""
		elif c == "\"" or c == "'":
			quote = c
			out.append(c)
		elif comments and c == "#":
			while i < n and text[i] != "\n":
				i += 1
			continue
		elif not (c in [" ", "\t", "\n", "\r", "\f", "\v"]):
			out.append(c)
		i += 1
	return out.join("")
