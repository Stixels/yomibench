extends Node
# One real match plus a reused preview game, driven the way the game's own singleplayer loop
# does it (game.gd process_tick), minus the UI and frame pacing.
#
# Previews copy the real match into the preview game with the game's own copy_to, plus fixes for
# what copy_to misses (_sync_fighter, _sync_state) and for reusing one preview game for the whole
# match, which the game itself never does (its in-game ghost is a fresh Game.tscn per preview).
# Reuse is not only faster: copying a mid-match position into a freshly instanced game can crash
# the native code at a throw release, so the preview game is created once, at turn 0.
#
# Handles (docs/PROTOCOL.md) are recipes, not snapshots: a handle is its parent plus the pair and
# stopping rule applied to it, so any handle can be rebuilt by copying the real match and
# replaying its path (the simulation is deterministic). The preview game remembers which handle
# it currently holds, so continuing a rollout chain costs only the new simulation.

const GameScene = preload("res://Game.tscn")
const TICK_CAP = 100000

# Fighter vars never synced by _sync_fighter: identity, wiring, and native-data mirrors.
const FIGHTER_SKIP = {
	"name": true, "id": true, "is_ghost": true, "objs_map": true, "opponent": true, "fighter_owner": true,
	"creator": true, "hooks": true, "data": true, "obj_data": true, "state_variables": true,
}
const NEUTRAL_DI = {"x": 0, "y": 0}
# game.gd vars never synced by _sync_game: wiring, UI, replay, network and ghost-mode state.
const GAME_SKIP = {
	"is_ghost": true, "objs_map": true, "objects": true, "active_objects": true, "effects": true,
	"detached_husks": true, "p1": true, "p2": true, "camera": true, "objects_node": true, "fx_node": true,
	"hooks": true, "match_data": true, "ghost_game": true, "mouse_pressed": true, "mouse_position": true,
	"drag_position": true, "max_replay_tick": true, "undoing": true, "singleplayer": true,
	"game_paused": true, "buffer_playback": true, "buffer_edit": true, "frame_passed": true,
	"ghost_cleaned": true, "quitter_focus": true, "quitter_focus_ticks": true, "advance_frame_input": true,
	"frame_by_frame": true, "network_simulate_ready": true, "is_afterimage": true, "ghost_hidden": true,
	"ghost_speed": true, "ghost_tick": true, "simulated_once": true, "started_multiplayer": true,
	"p1_username": true, "p2_username": true, "my_id": true, "draw_stage": true, "snapping_camera": true,
	"waiting_for_player_prev": true, "spectate_tick": true, "camera_snap_position": true,
	"ghost_freeze": true, "network_sync_tick": true, "ghost_simulated_ticks": true, "is_in_replay": true,
	"ghost_actionable_freeze_ticks": true, "ghost_p1_actionable": true, "ghost_p2_actionable": true,
	"made_afterimage": true, "p1_ghost_ready_tick": true, "p2_ghost_ready_tick": true, "spectating": true,
	"ghost_time": true, "camera_zoom": true,
}

var real = null
var ghost = null
var match_data = {}
var raw_start = [{}, {}]   # each fighter's raw values at the start of the match (see _raw)
# The real match's inputs in ReplayManager's format, for save_replay. ReplayManager.frames itself
# can't be saved as is: the preview game is not a ghost, so previews record into it too.
var replay_frames = {1: {}, 2: {}, "finished": false, "emotes": {1: {}, 2: {}}}
var copy_mode = "dirty"
var turn = 0
var gens = []            # Candidates per seat, bound to the real game's fighters
var ghost_gens = []      # Candidates per seat, bound to the preview game's fighters
var cands = [[], []]     # the real match's current candidates per seat
var real_version = 0     # bumps whenever the real match changes
var ghost_version = -1   # real_version the preview game last had a full copy of
var dirty = [{}, {}]     # per preview fighter: states entered or exited since its last copy
var initial_states = []  # per preview fighter: the state a fresh Game.tscn fighter starts in
var var_cache = {}       # script -> [property names], for the sync passes
var unlisted_cache = {}  # instance id -> [names outside its property_list], see _unlisted

var handles = {}         # id -> {id, parent (0 = real), pair (Array or null), min_ticks, turn, cands}
var next_handle = 1
var ghost_holds = 0      # handle the preview game currently holds (0 = none)

var trace = null         # debug: per-tick snapshots during one preview / step, when asked
var capture = null       # [[states p1 entered], [p2]] during a preview, when asked (detail)
var capture_real = null  # the same for the real step
var last_trace = null    # debug: trace of the last real step, when asked
var entered_real = [[], []]  # states entered during the last real step
var monitor = {}         # "i,j" -> detail of this turn's preview of that pair (divergence check)
var divergence_log = ""
var div = {"compared": 0, "diverged": 0, "fields": {}, "recent": []}
var stats = {"copy_us": 0, "sim_us": 0, "sim_ticks": 0, "previews": 0, "continues": 0, "replays": 0,
	"gen_us": 0, "gens": 0, "step_us": 0, "steps": 0, "full_copies": 0, "obs_us": 0, "phase_us": {}}
var _pt = 0              # profiling: start of the current copy phase


func _phase(name):
	var t = OS.get_ticks_usec()
	stats.phase_us[name] = stats.phase_us.get(name, 0) + t - _pt
	_pt = t


func start(p1_name, p2_name, seed_, max_candidates, settings, copy_mode_, divergence_log_, div_, sets_ = null, set5_ = {},
		set6_ = {}):
	if sets_ != null:
		candidate_sets = sets_
	set5 = set5_
	set6 = set6_
	characters = [p1_name, p2_name]
	copy_mode = copy_mode_
	div = div_
	divergence_log = divergence_log_
	# A standard online lobby's settings (YomiServer._standard_settings), plus the matchup and seed.
	match_data = settings.duplicate(true)
	match_data["selected_characters"] = {1: {"name": p1_name}, 2: {"name": p2_name}}
	match_data["seed"] = seed_
	ReplayManager.playback = false
	ReplayManager.resimulating = false
	ReplayManager.init()
	real = _new_game()
	raw_start = [_raw_full(real.p1), _raw_full(real.p2)]
	gens = _make_gens(real, max_candidates)
	for i in range(2):
		[real.p1, real.p2][i].state_machine.connect("state_changed", self, "_on_real_state_changed", [i])
	_make_ghost(max_candidates)
	_advance(real)
	_begin_turn()


func _new_game():
	var g = GameScene.instance()
	# Not a ghost: a few states deliberately behave differently in the in-game ghost preview
	# (e.g. Cowboy's Brandish), and states only build the property lists copy_to relies on when
	# their game isn't a ghost. Previews should match the real outcome.
	g.is_ghost = false
	add_child(g)
	g.set_physics_process(false)
	g.set_process(false)
	g.start_game(true, match_data)
	for p in [g.p1, g.p2]:
		p.set_process(false)
	# Screenshake code looks up "the" camera through this scene-wide group, which with several
	# games alive can be another game's camera (and its shake tweens). Nothing here renders.
	if g.camera.is_in_group("Camera"):
		g.camera.remove_from_group("Camera")
	return g


# Each seat's candidate rules: "5" (Candidates.gd with the server's set-5 switches), "4"
# (Candidates4.gd, frozen) or "2" (Candidates2.gd, frozen, for A/B arenas).
var candidate_sets = ["4", "4"]
var set5 = {}            # set 5's switches (YomiServer --yomi-set5)
var set6 = {}            # set 6's switches (YomiServer --yomi-set6), for set-6 seats
var characters = ["", ""]
# Set 6 is set 5's generator with its panel-set layer on (Candidates.gd LAYERS); a later set adds a layer there.
const GENERATORS = {"6": "/Candidates.gd", "5": "/Candidates.gd", "4": "/Candidates4.gd", "2": "/Candidates2.gd"}
# Legality (Legality.gd; YomiServer --yomi-legality, --yomi-legality-check): set before start().
var legality_mode = "copy"
var legality_check = ""
var legality_counts = {"calls": 0, "mismatches": 0, "recent": []}


func _make_gens(g, max_candidates):
	var dir = get_script().resource_path.get_base_dir()
	# One legality node per game, shared by both seats' generators.
	var legality = load(dir + "/Legality.gd").new()
	add_child(legality)
	legality.setup(g, legality_mode, legality_check, legality_counts)
	var out = []
	for i in range(2):
		var p = [g.p1, g.p2][i]
		var c = load(dir + GENERATORS[candidate_sets[i]]).new()
		add_child(c)
		# Set 2 keeps the budget it was played with (30); the match's budget is set 4's and 5's.
		if candidate_sets[i] in ["5", "6"]:
			c.setup(p, max_candidates, set5, characters[i], g, set6 if candidate_sets[i] == "6" else {})
		else:
			c.setup(p, 30 if candidate_sets[i] == "2" else max_candidates)
		c.legality = legality
		out.append(c)
	return out


func _make_ghost(max_candidates):
	ghost = _new_game()
	Global.current_game = real
	ghost_gens = _make_gens(ghost, max_candidates)
	initial_states = [ghost.p1.state_machine.state.name, ghost.p2.state_machine.state.name]
	for i in range(2):
		var sm = [ghost.p1, ghost.p2][i].state_machine
		sm.connect("state_exited", self, "_on_ghost_state_exited", [i])
		sm.connect("state_changed", self, "_on_ghost_state_changed", [i])


func done():
	return real.should_game_end()


func candidate_count(seat):
	return cands[seat].size()


# --- turn flow ---------------------------------------------------------------------------------

# Ticks until a player must act or the match ends.
func _advance(g):
	var n = 0
	if trace != null:
		trace.append(_trace_snap(g))
	while not g.is_waiting_on_player() and not g.should_game_end() and n < TICK_CAP:
		g.tick()
		n += 1
		if trace != null:
			trace.append(_trace_snap(g))
	return n


# Debug (trace): a per-tick snapshot of what decides hits and movement.
func _trace_snap(g):
	var out = {"tick": g.current_tick}
	for p in [g.p1, g.p2]:
		var st = p.current_state()
		var pos = p.get_pos()
		var vel = p.get_vel()
		out[p.name] = [st.name, st.current_tick, pos.x, pos.y, str(vel.x), str(vel.y), p.hp, p.hitlag_ticks,
			p.invulnerable, p.on_the_ground, p.is_grounded(), p.combo_count, str(p.hurtbox.x) + "," + str(p.hurtbox.y)]
	var objs = []
	for o in g.objects:
		if is_instance_valid(o) and not o.disabled:
			var opos = o.get_pos()
			var hbs = []
			for st in o.state_machine.states_map.values():
				for c in st.get_children():
					if c is Hitbox and (c.active or c.enabled):
						hbs.append([st.name, c.name, c.active, c.enabled, c.tick, str(c.hit_objects), c.pos_x, c.pos_y])
			objs.append([str(o.obj_name), o.current_state().name if o.current_state() else "", opos.x, opos.y, o.current_state().current_tick, hbs])
	out["objs"] = objs
	out["freeze"] = [g.super_freeze_ticks, g.hit_freeze, g.parry_freeze]
	return out


# game.gd process_tick opens a turn over two frames: the player who became actionable first
# makes the other one actionable (busy_interrupt if it can't really act), then the second
# branch runs for the other player. Both always run before anyone can lock in.
func _open_turn(g):
	_open_turn_branches(g)
	# The action menu's legality pass (ActionButtons.update_buttons, here Candidates.legal_states)
	# runs every move's is_usable(), and some have side effects on the fighter (a Wizard check
	# refreshes its cached position data). The real match runs it when generating candidates;
	# previews and rollouts pass decisions without generating, so run the pass there too (fresh:
	# the state changed since the generators' last epoch, so no cached answer applies).
	if g == ghost:
		for gn in ghost_gens:
			gn.legal_states(true)


func _open_turn_branches(g):
	if g.p1.state_interruptable:
		_open_branch(g, g.p2)
		_open_branch(g, g.p1)
	elif g.p2.state_interruptable:
		_open_branch(g, g.p1)
		_open_branch(g, g.p2)


func _open_branch(g, other):
	other.busy_interrupt = (not other.state_interruptable and not (other.current_state().interruptible_on_opponent_turn or other.feinting or g.negative_on_hit(other)))
	if not other.busy_interrupt:
		other.current_state().on_interrupt()
	other.state_interruptable = true


func _on_ghost_state_exited(state, i):
	dirty[i][state.name] = true


func _on_ghost_state_changed(_stack, i):
	var name = [ghost.p1, ghost.p2][i].state_machine.state.name
	dirty[i][name] = true
	if capture != null and capture[i].size() < 8:
		capture[i].append(name)


func _on_real_state_changed(_stack, i):
	if capture_real != null and capture_real[i].size() < 8:
		capture_real[i].append([real.p1, real.p2][i].state_machine.state.name)


func _begin_turn():
	real_version += 1
	handles.clear()
	ghost_holds = 0
	if real.should_game_end():
		cands = [[], []]
		return
	_open_turn(real)
	var t = OS.get_ticks_usec()
	cands = [gens[0].generate(), gens[1].generate()]

	stats.gen_us += OS.get_ticks_usec() - t
	stats.gens += 1


func _submit(g, c1, c2):
	g.p1.on_action_selected(c1.action, _dup(c1.data), _dup(c1.extra))
	g.p2.on_action_selected(c2.action, _dup(c2.data), _dup(c2.extra))


func _dup(v):
	if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY:
		return v.duplicate(true)
	return v


func step(i1, i2, trace_ = false):
	var moves = [cands[0][i1].action, cands[1][i2].action]
	_play(cands[0][i1], cands[1][i2], trace_)
	var key = "%d,%d" % [i1, i2]
	if monitor.has(key):
		_check_divergence(monitor[key], _detail(real), moves)
	monitor.clear()


# Both seats lock in these inputs ({action, data, extra}), the real match plays the turn, and the
# next turn opens with fresh candidates.
func _play(c1, c2, trace_):
	var t = OS.get_ticks_usec()
	capture_real = [[], []]
	trace = [] if trace_ else null
	# Fighters record an input at the tick that uses it: the current tick + 1 (BaseChar.tick_before).
	var input_tick = real.current_tick + 1
	_submit(real, c1, c2)
	_advance(real)
	for id in [1, 2]:
		var rec = ReplayManager.frames[id].get(input_tick)
		if rec != null:
			replay_frames[id][input_tick] = rec.duplicate(true)
	last_trace = trace
	trace = null
	stats.step_us += OS.get_ticks_usec() - t
	stats.steps += 1
	turn += 1
	_begin_turn()
	# Through the turn opening, as a preview's capture is (_run opens the next turn too).
	entered_real = capture_real
	capture_real = null


# --- raw inputs (step_raw) ---------------------------------------------------------------------

# Plays each seat's exact input ({action, data, extra}, P1 then P2), on or off the candidate grid,
# the way a player's panel submission does (ActionButtons.on_action_submitted ->
# fighter.on_action_selected), then opens the next decision. Returns "" or why the inputs were
# refused (then nothing was played). An input is accepted when a player could have sent it:
#   action  "Continue" or a move of this fighter's action menu that is legal now with this extra
#           (Legality.gd, the same question candidates ask)
#   data    null for Continue and moves without a data scene; otherwise the shape of the move's
#           data scene's get_data(): the same keys, each value of the widget's type (integers
#           where the widget sends integers)
#   extra   exactly the keys the panel sends (DI, reverse, feint, prediction and the character's
#           PlayerExtra keys), each of the panel's type; DI an integer percent vector of length at
#           most 100 (+1 for rounding), non-neutral only with DI enabled; feint only where the
#           move can feint, reverse only where it can be reversed
# Numbers may come as JSON floats; "[Object:null]" (JSON.print's null Object) and {"$d": [[k, v],
# ...]} (the recorder's typed dicts) are read the way a recorded game stores them.
# option_by_id (the request's raw_option_by_id, default off): an option widget's data (e.g. Ninja
# QuickSlash's Attack) is checked against what the panel sends with the recorded id selected, not
# against its default item: the game's ActionUIDataOptionButton sends {id, name: items[id]}, and
# Godot 3's flat OptionButton.items makes that name "No Attack" / null / false / 0 for ids 0-3. On,
# the id must be an enabled item and the name must equal the panel's; off, today's check exactly.
func step_raw(inputs, option_by_id = false):
	if typeof(inputs) != TYPE_ARRAY or inputs.size() != 2:
		return "inputs must be [{action, data, extra} for P1, {action, data, extra} for P2]"
	var cs = []
	for seat in range(2):
		var c = _raw_input(seat, inputs[seat], option_by_id)
		if typeof(c) == TYPE_STRING:
			return "seat %d: %s" % [seat, c]
		cs.append(c)
	_play(cs[0], cs[1], false)
	monitor.clear()
	return ""


func _raw_input(seat, inp, option_by_id = false):
	if typeof(inp) != TYPE_DICTIONARY or typeof(inp.get("action")) != TYPE_STRING:
		return "an input is {action (a string), data, extra}"
	var gen = gens[seat]
	var f = gen.fighter
	var action = inp.action
	var state = null
	if action != "Continue":
		for s in gen.button_states:
			if s.name == action:
				state = s
		if state == null:
			return "%s is not in this fighter's action menu" % action
	var errs = []
	var template = gen.default_candidate().extra  # a fresh epoch: every widget at its default
	var extra = _raw_like(_raw_decode(inp.get("extra")), template, "extra", errs)
	var data = _raw_decode(inp.get("data"))
	if state == null or state.data_ui_scene == null:
		if data != null:
			errs.append("data must be null for %s (it has no data scene)" % action)
	else:
		var node = gen._open_data_node(state)
		var like
		var options = []
		if option_by_id:
			like = _raw_option_like(node, data, options, errs)
		else:
			like = node.get_data()
		if typeof(like) == TYPE_DICTIONARY or typeof(like) == TYPE_ARRAY:
			like = like.duplicate(true)
		data = _raw_like(data, like, "data", errs)
		if errs.empty():
			for k in options:
				if typeof(data) != TYPE_DICTIONARY or typeof(data.get(k)) != TYPE_DICTIONARY:
					continue
				if typeof(like.get(k)) != TYPE_DICTIONARY or not like[k].has("id"):
					continue
				if not _raw_same(data[k].get("name"), like[k].get("name")):
					errs.append("data.%s.name must be what the panel sends for item %d (%s)" % [k, like[k].id, str(like[k].name)])
	if not errs.empty():
		return errs[0]
	var di = extra.DI
	if typeof(di) != TYPE_DICTIONARY:
		return "extra.DI must be {x, y}"
	if abs(di.x) > 100 or abs(di.y) > 100 or di.x * di.x + di.y * di.y > 101 * 101:
		return "extra.DI (%d, %d) is longer than the DI wheel allows" % [di.x, di.y]
	if (di.x != 0 or di.y != 0) and not f.di_enabled:
		return "DI is disabled"
	if typeof(extra.prediction) != TYPE_INT or extra.prediction < -1:
		return "extra.prediction must be -1 or a move category"
	if extra.feint and (state == null or not state.can_feint()):
		return "feint is disabled for %s" % action
	if extra.reverse and (state == null or not (state.flip_allowed() and state.reversible)):
		return "reverse is disabled for %s" % action
	if state != null:
		var legal = gen.legality.legal_states(gen, extra)
		if not state in legal:
			var names = ["Continue"]
			for s in legal:
				names.append(s.name)
			return "%s is not legal now with these extras (legal: %s)" % [action, PoolStringArray(names).join(", ")]
	return {"action": action, "data": data, "extra": extra}


# step_raw's raw_option_by_id: the data scene's get_data() with each option widget (an OptionButton
# with get_data, keyed by its node name as the data scene sends it) set to the item the recorded
# data selects; every selection is put back afterwards, so the panel the candidates read is
# unchanged. Appends why to errs for an id that no enabled item has; `options` gets the keys set.
func _raw_option_like(node, data, options, errs):
	if typeof(data) != TYPE_DICTIONARY:
		return node.get_data()
	var widgets = []
	_raw_option_widgets(node, widgets)
	var saved = []
	for n in widgets:
		var k = str(n.name)
		var v = data.get(k)
		if typeof(v) != TYPE_DICTIONARY or not v.has("id") or k in options:
			continue
		var id = v.id
		if not (typeof(id) in [TYPE_INT, TYPE_REAL]) or float(id) != floor(float(id)):
			errs.append("data.%s.id must be an integer" % k)
			continue
		id = int(id)
		if id < 0 or id >= n.get_item_count() or n.is_item_disabled(id):
			errs.append("data.%s.id %d is not an enabled item of the panel (%d items)" % [k, id, n.get_item_count()])
			continue
		saved.append([n, n.selected])
		n.select(id)
		options.append(k)
	var like = node.get_data()
	if typeof(like) == TYPE_DICTIONARY or typeof(like) == TYPE_ARRAY:
		like = like.duplicate(true)
	saved.invert()
	for s in saved:
		s[0].select(s[1])
	return like


func _raw_option_widgets(n, out):
	for c in n.get_children():
		if c is OptionButton and c.has_method("get_data"):
			out.append(c)
		else:
			_raw_option_widgets(c, out)


# Equal values of one type; numbers compare by value (JSON sends 0 as 0.0).
func _raw_same(a, b):
	if typeof(a) in [TYPE_INT, TYPE_REAL] and typeof(b) in [TYPE_INT, TYPE_REAL]:
		return float(a) == float(b)
	return typeof(a) == typeof(b) and a == b


# Raw-input reading: "[Object:null]" is null, {"$d": [[key, value], ...]} a dict.
func _raw_decode(v):
	if typeof(v) == TYPE_STRING and v == "[Object:null]":
		return null
	if typeof(v) == TYPE_ARRAY:
		var out = []
		for x in v:
			out.append(_raw_decode(x))
		return out
	if typeof(v) == TYPE_DICTIONARY:
		if v.size() == 1 and v.has("$d") and typeof(v["$d"]) == TYPE_ARRAY:
			var d = {}
			for kv in v["$d"]:
				if typeof(kv) == TYPE_ARRAY and kv.size() == 2:
					d[_raw_decode(kv[0])] = _raw_decode(kv[1])
			return d
		var out = {}
		for k in v:
			out[k] = _raw_decode(v[k])
		return out
	return v


# v in the shape of `like` (what the panel or data scene sends): the same keys, each value of its
# type (JSON numbers become the widget's int or float). like == null: any value, whole numbers
# become ints (a widget that sends nothing by default, e.g. Mutant's juke_dir).
func _raw_like(v, like, path, errs):
	var tl = typeof(like)
	var tv = typeof(v)
	if tl == TYPE_NIL:
		if tv == TYPE_REAL and v == floor(v):
			return int(v)
		if tv == TYPE_DICTIONARY:
			var out = {}
			for k in v:
				out[k] = _raw_like(v[k], null, path + "." + str(k), errs)
			return out
		if tv == TYPE_ARRAY:
			var out = []
			for x in v:
				out.append(_raw_like(x, null, path, errs))
			return out
		return v
	if tl == TYPE_DICTIONARY:
		if tv == TYPE_NIL:
			return null  # e.g. Robot's fly_dir when not flying
		if tv != TYPE_DICTIONARY:
			errs.append("%s must be an object with keys %s" % [path, str(like.keys())])
			return v
		for k in like:
			if not v.has(k):
				errs.append("%s lacks %s (expected keys %s)" % [path, str(k), str(like.keys())])
				return v
		var out = {}
		for k in v:
			if not like.has(k):
				errs.append("%s has an unknown key %s (expected keys %s)" % [path, str(k), str(like.keys())])
				return v
			out[k] = _raw_like(v[k], like[k], path + "." + str(k), errs)
		return out
	if tl == TYPE_ARRAY:
		if tv != TYPE_ARRAY:
			errs.append("%s must be a list" % path)
			return v
		var out = []
		for x in v:
			out.append(_raw_like(x, like[0] if like.size() > 0 else null, path, errs))
		return out
	if tl == TYPE_INT or tl == TYPE_REAL:
		if not (tv in [TYPE_INT, TYPE_REAL]):
			errs.append("%s must be a number" % path)
			return v
		if tl == TYPE_INT:
			if float(v) != floor(float(v)):
				errs.append("%s must be an integer" % path)
				return v
			return int(v)
		return float(v)
	if tv != tl:
		errs.append("%s must be of the type the panel sends (like %s)" % [path, str(like)])
	return v


# Writes the real match so far as a game replay file (ReplayManager's format: match_data plus
# frames, version, mod_data), which the game's replay browser lists and plays back exactly. With no
# path, the file's bytes come back base64-encoded instead.
func save_replay(path):
	var inline = path == null
	if inline:
		# Servers started together share one user dir: keep each process's scratch file apart.
		path = "user://yomibench_replay_%d.tmp" % OS.get_process_id()
	var data = match_data.duplicate(true)
	var frames = replay_frames.duplicate(true)
	frames["finished"] = real.game_finished or real.should_game_end()
	data["frames"] = frames
	data["version"] = Global.VERSION
	data["mod_data"] = {}
	Directory.new().make_dir_recursive(path.get_base_dir())
	var f = File.new()
	if f.open(path, File.WRITE) != OK:
		return _error("io", "cannot write " + path)
	f.store_var(data, true)
	f.close()
	var out = {"path": path, "tick": real.current_tick, "hp": [real.p1.hp, real.p2.hp], "version": Global.VERSION}
	if inline:
		# The file's bytes, so a client on another machine (or outside Docker) can keep it.
		if f.open(path, File.READ) != OK:
			return _error("io", "cannot read back " + path)
		out["replay_b64"] = Marshalls.raw_to_base64(f.get_buffer(f.get_len()))
		f.close()
		Directory.new().remove(path)
		out.erase("path")
	return out


# --- previews and handles ----------------------------------------------------------------------

# One preview (the server slices batches across frames; see YomiServer._run_job).
func preview_one(pair, detail, want_cands, trace_ = false):
	var t0 = OS.get_ticks_usec()
	_ghost_from_real()
	var t1 = OS.get_ticks_usec()
	capture = [[], []] if detail else null
	trace = [] if trace_ else null
	var turns = _run(cands, pair, 0)
	var entered = capture
	var traced = trace
	capture = null
	trace = null
	stats.copy_us += t1 - t0
	stats.sim_us += OS.get_ticks_usec() - t1
	stats.previews += 1
	var h = _new_handle(0, pair, 0, turn + turns)
	monitor["%d,%d" % [pair[0], pair[1]]] = _detail(ghost)
	var obs = _handle_obs(h, detail, want_cands)
	if detail:
		obs.detail["entered"] = entered
	if traced != null:
		obs["trace"] = traced
	return obs


# Returns an observation, or {"error": ...}.
func cont(h, pair, min_ticks, detail, want_cands):
	if not handles.has(h) or handles[h].released or not _path_ok(h):
		return _error("stale_handle", "unknown, released or stale handle %d" % h)
	_materialize(h)
	if ghost.should_game_end():
		return _error("match_over", "that handle's match has ended")
	var node_cands = null
	if pair != null:
		node_cands = _cands_at(h)
		for seat in range(2):
			if pair[seat] < 0 or pair[seat] >= node_cands[seat].size():
				return _error("bad_request", "seat %d: handle %d has no candidate %d" % [seat, h, pair[seat]])
	var t = OS.get_ticks_usec()
	var turns = _run(node_cands, pair, min_ticks)
	stats.sim_us += OS.get_ticks_usec() - t
	stats.continues += 1
	var nh = _new_handle(h, pair, min_ticks, handles[h].turn + turns)
	return _handle_obs(nh, detail, want_cands)


func release(ids):
	var n = 0
	# A released handle keeps its recipe: live descendants still replay through it.
	for h in ids:
		if handles.has(h) and not handles[h].released:
			handles[h].released = true
			n += 1
	return n


func _error(code, message):
	return {"error": {"code": code, "message": message}}


func _new_handle(parent, pair, min_ticks, turn_):
	var h = next_handle
	next_handle += 1
	handles[h] = {"id": h, "parent": parent, "pair": pair, "min_ticks": min_ticks, "turn": turn_, "cands": null, "released": false}
	ghost_holds = h
	return h


func _path_ok(h):
	while h != 0:
		if not handles.has(h):
			return false
		h = handles[h].parent
	return true


# Puts the preview game in handle h's state: nothing to do if it already holds h, otherwise
# copy the real match and replay h's path.
func _materialize(h):
	if ghost_holds == h:
		return
	var path = []
	var cur = h
	while cur != 0:
		path.push_front(handles[cur])
		cur = handles[cur].parent
	_ghost_from_real()
	var node_cands = cands
	for i in range(path.size()):
		var rec = path[i]
		_run(node_cands if rec.pair != null else null, rec.pair, rec.min_ticks)
		ghost_holds = rec.id
		if i + 1 < path.size() and path[i + 1].pair != null:
			node_cands = _cands_at(rec.id)
	stats.replays += 1


# Candidates at handle h; the preview game must be holding h.
func _cands_at(h):
	var rec = handles[h]
	if rec.cands == null:
		if ghost.should_game_end():
			rec.cands = [[], []]
		else:
			rec.cands = [ghost_gens[0].generate(), ghost_gens[1].generate()]
	return rec.cands


func _ghost_from_real():
	_copy_real_to(ghost)
	ghost_holds = 0


# Plays `pair` (ids into node_cands; null = both seats' default) on the preview game, then the
# default continuation at each decision until at least min_ticks have passed. Stops at a decision
# point (turn opened) or the match end. Returns the number of decisions passed.
func _run(node_cands, pair, min_ticks):
	var g = ghost
	var start = g.current_tick
	if pair == null:
		_submit(g, ghost_gens[0].default_candidate(), ghost_gens[1].default_candidate())
	else:
		_submit(g, node_cands[0][pair[0]], node_cands[1][pair[1]])
	stats.sim_ticks += _advance(g)
	var turns = 1
	if not g.should_game_end():
		_open_turn(g)
	while not g.should_game_end() and g.current_tick - start < min_ticks:
		_submit(g, ghost_gens[0].default_candidate(), ghost_gens[1].default_candidate())
		stats.sim_ticks += _advance(g)
		turns += 1
		if not g.should_game_end():
			_open_turn(g)
	return turns


func _handle_obs(h, detail, want_cands):
	var t = OS.get_ticks_usec()
	var ended = ghost.should_game_end()
	var obs = {
		"handle": h, "turn": handles[h].turn, "done": ended, "winner": _winner(ghost),
		"actionable": _actionable(ghost), "state": _state(ghost),
	}
	if want_cands:
		obs["candidates"] = _cands_json(_cands_at(h))
	if detail:
		obs["detail"] = _detail(ghost)
	stats.obs_us += OS.get_ticks_usec() - t
	return obs


# --- copying the real match into the preview game ------------------------------------------------

# game.gd copy_to, step for step, plus the sync passes and safe object handling.
# A live projectile/summon of the real game, rebuilt in the preview game `g`: a fresh instance of the same scene,
# registered with `g` under the same name, then given the source's state through the game's own copy_to (the
# object and its physics body). The fresh instance is initialised as a ghost: a non-ghost init builds the property
# list of every state and hitbox, only for copy_to to overwrite it, which is most of an object's copy cost. The
# flag is restored before anything simulates.
func _clone_object_into(g, source):
	var clone = load(source.filename).instance()
	clone.obj_name = source.obj_name
	g.on_object_spawned(clone)
	clone.is_ghost = true
	source.copy_to(clone)
	clone.is_ghost = false
	source.chara.copy_to(clone.chara)
	clone.update_data()
	return clone


func _copy_real_to(g):
	_pt = OS.get_ticks_usec()
	g.current_tick = real.current_tick
	real.p1.chara.copy_to(g.p1.chara)
	real.p2.chara.copy_to(g.p2.chara)
	g.p1.update_data()
	g.p2.update_data()
	_phase("chara")
	var partial = copy_mode == "dirty" and ghost_version == real_version
	var fighters = [[real.p1, g.p1], [real.p2, g.p2]]
	# copy_to enters the copied state with change_state, which first runs the _exit of the
	# preview fighter's current state: in a fresh game the harmless initial state, here whatever
	# the previous preview left. Those exits can act on the other fighter (LassoPull._exit
	# releases a grabbed opponent: the already-copied P1 dropped out of Grabbed). Put both
	# fighters back in their initial state first, without running exits, as a fresh game has them.
	for i in range(2):
		var sm = fighters[i][1].state_machine
		var init_state = sm.states_map[initial_states[i]]
		if sm.state != init_state:
			sm.state.active = false
			sm.state = init_state
			init_state.active = true
	for i in range(2):
		var src = fighters[i][0]
		var dst = fighters[i][1]
		# copy_to appends the source's state stack without clearing it. Start from what a fresh
		# preview game has, its initial state: entering the copied state can look at the previous
		# one (JumpKickLanding does), and a one-entry stack has none.
		dst.state_machine.states_stack = [dst.state_machine.states_map[initial_states[i]]]
		var names
		if partial:
			names = _copy_fighter_partial(src, dst, dirty[i])
		else:
			dirty[i].clear()
			src.copy_to(dst)
			names = src.state_machine.states_map.keys()
		_phase("fighter_copy")
		_sync_fighter(src, dst)
		_phase("sync_fighter")
		for n in names:
			_sync_state(src.state_machine.states_map[n], dst.state_machine.states_map[n])
		_phase("sync_states")
		# copy_to's native chara copy is exact, but it then sets the velocity again from the
		# 5-decimal strings in get_data(); redo the exact copy last.
		src.chara.copy_to(dst.chara)
		# Set by the turn opening (busy_interrupt isn't a copied state variable).
		dst.busy_interrupt = src.busy_interrupt
		dst.state_interruptable = src.state_interruptable
		dst.state_hit_cancellable = src.state_hit_cancellable
		dst.update_data()
		_phase("fighter_tail")
	ghost_version = real_version
	stats.full_copies += 0 if partial else 1
	g.p1.hp = real.p1.hp
	g.p2.hp = real.p2.hp
	_retire_objects(g)
	_phase("retire")
	real.clean_objects()
	g.objs_spawned_count = real.objs_spawned_count
	var copied = []
	for object in real.objects:
		if not is_instance_valid(object) or object.disabled:
			continue  # gone or spent: nothing to carry into the preview
		copied.append([object, _clone_object_into(g, object)])
	_phase("objects")
	# References between objects and fighters (a lasso's target, a projectile's owner) are
	# never copied; point them at the preview game's counterparts.
	# Object states get the same unlisted-variable and hitbox sync as fighter states: e.g.
	# frame_methods (which ticks have _frame_N handlers, a cache paired with the natively copied
	# max_tick; left empty, a re-entered state skips its frame handlers) and queued_state.
	for pair in copied:
		# copy_to inits the fresh object (entering its starting state) and then change_states
		# into the copied state, running the starting state's _exit. Some exits disable their
		# object (Wizard's orb lightning: `_exit(): host.disable()`), so the copy came out
		# disabled while the real one is live. disable() only hides nodes, sets the flag and
		# deactivates hitboxes (restored by the state sync below): put them back.
		var src = pair[0]
		var dst = pair[1]
		if dst.disabled != src.disabled:
			dst.disabled = src.disabled
			dst.visible = src.visible
			for n in ["sprite", "state_machine", "collision_box", "hurtbox"]:
				var a = src.get(n)
				var b = dst.get(n)
				if a is CanvasItem and b is CanvasItem:
					b.visible = a.visible
		for n in src.state_machine.states_map:
			_sync_state(src.state_machine.states_map[n], dst.state_machine.states_map[n])
	_phase("object_states")
	for pair in copied:
		_remap_refs(pair[0], pair[1], g)
		for n in pair[0].state_machine.states_map:
			_remap_refs(pair[0].state_machine.states_map[n], pair[1].state_machine.states_map[n], g)
	for i in range(2):
		var f = fighters[i]
		_remap_refs(f[0], f[1], g)
		_remap_refs(f[0].current_state(), f[1].current_state(), g)
	# game.copy_to targets a fresh game, so game-level state (super/hit/parry freeze, clash tick,
	# turn flags, ...) starts at its defaults there; a reused preview game would keep whatever the
	# previous preview left. Take the real match's values.
	_sync_game(real, g)
	# Every body's cached `data` (what get_pos() and much of the game logic read) is refreshed from
	# its native state by update_data(), which the copy calls. But the real game can end a tick with
	# a body moved natively after its last update_data (Ninja's grappling hook pull: cached x -10,
	# native -14), and the next ticks read the stale cache. Copy the native state and then the
	# cached data exactly as the real bodies have them, last (object init code can also move bodies).
	for i in range(2):
		_copy_body(fighters[i][0], fighters[i][1])
	for pair in copied:
		_copy_body(pair[0], pair[1])
	g.camera.limit_left = real.camera.limit_left
	g.camera.limit_right = real.camera.limit_right
	g.game_finished = real.game_finished
	g.game_end_tick = real.game_end_tick
	_phase("remap")


# Fighter copy_to copies every one of the fighter's states (native.copy_state over each state's
# script variables), which is most of a preview's cost. When the real match hasn't changed since
# the preview game's last full copy, only states it entered or exited since can differ, so its
# states_map is narrowed to those (plus both current states) for the duration of the copy.
# copy_to rebuilds the state stack by looking names up in that map, so it gets an empty source
# stack and the stack is rebuilt from the full map afterwards. Returns the copied state names.
func _copy_fighter_partial(src, dst, touched):
	var full_map = dst.state_machine.states_map
	var names = touched.duplicate()
	names[src.current_state().name] = true
	names[dst.current_state().name] = true
	var reduced = {}
	for n in names:
		if full_map.has(n):
			reduced[n] = full_map[n]
	touched.clear()
	var src_stack = src.state_machine.states_stack
	src.state_machine.states_stack = []
	dst.state_machine.states_map = reduced
	src.copy_to(dst)
	dst.state_machine.states_map = full_map
	src.state_machine.states_stack = src_stack
	for st in src_stack:
		dst.state_machine.states_stack.append(full_map[st.name])
	return reduced.keys()


func _copy_body(src, dst):
	src.chara.copy_to(dst.chara)
	dst.data = _dup(src.data)
	dst.obj_data = dst.data["object_data"] if typeof(dst.data) == TYPE_DICTIONARY and dst.data.has("object_data") else _dup(src.obj_data)


func _sync_game(src, dst):
	for n in _vars(src):
		if GAME_SKIP.has(n):
			continue
		var v = src.get(n)
		if _plain(v):
			dst.set(n, _dup(v))


# copy_to copies the fighter's state variables before change_state, whose enter code then resets
# some of them (e.g. melee_attack_combo_scaling_applied), and skips variables outside that list
# (e.g. touch_of_death). Copy every plain-data fighter variable again, last.
func _sync_fighter(src, dst):
	for n in _vars(src):
		if FIGHTER_SKIP.has(n):
			continue
		var v = src.get(n)
		if _plain(v):
			dst.set(n, _dup(v))


# native.copy_state copies only the variables in the state's property_list, built at init from
# values that were plain scalars then: anything null at init (e.g. queued_state) and every array
# or dictionary (e.g. interrupt_into, edited at runtime) is skipped. Copy those too.
func _sync_state(src, dst):
	_sync_unlisted(src, dst)
	# copy_to copies hitboxes only while they're active, so a hitbox a state switches on by hand
	# (LassoPull's) keeps a reused preview game's stale state (who it already hit, ...).
	var a = src.get_children()
	var b = dst.get_children()
	for i in range(a.size()):
		if a[i] is Hitbox:
			a[i].copy_to(b[i])
			_sync_unlisted(a[i], b[i])


# Plain-data script variables outside the object's native-copy property_list.
func _sync_unlisted(src, dst):
	for n in _unlisted(src):
		var v = src.get(n)
		if _plain(v):
			dst.set(n, _dup(v))


# Script variables of `o` outside its property_list (built once at the object's init, so the
# difference is cached per instance; instance ids are never reused in Godot 3).
func _unlisted(o):
	var id = o.get_instance_id()
	if unlisted_cache.has(id):
		return unlisted_cache[id]
	var listed = {}
	for n in o.property_list:
		listed[n] = true
	var out = []
	for n in _vars(o):
		if not listed.has(n) and n != "property_list":
			out.append(n)
	unlisted_cache[id] = out
	return out


func _remap_refs(src, dst, g):
	# A state's property_list only holds variables that were plain data at init, so object
	# references are always among its unlisted variables.
	var names = _unlisted(src) if src is ObjectState else _vars(src)
	for n in names:
		if n == "objs_map":
			continue
		var v = src.get(n)
		if typeof(v) == TYPE_OBJECT and is_instance_valid(v) and v is BaseObj:
			var key = v.obj_name if v.obj_name else v.name
			if g.objs_map.has(key):
				dst.set(n, g.objs_map[key])


func _vars(o):
	var s = o.get_script()
	if not var_cache.has(s):
		var names = []
		for prop in s.get_script_property_list():
			names.append(prop.name)
		var_cache[s] = names
	return var_cache[s]


# Data that can be copied by value: no objects anywhere inside.
func _plain(v):
	match typeof(v):
		TYPE_OBJECT:
			return false
		TYPE_ARRAY:
			for x in v:
				if not _plain(x):
					return false
		TYPE_DICTIONARY:
			for k in v:
				if not _plain(k) or not _plain(v[k]):
					return false
	return true


# game.copy_to frees the target's objects and then its effects without validity checks. An
# object's particles are also in `effects` and die with the object, so on a reused preview game
# that second free hits freed memory (an access violation in the release build). Free everything
# safely here, and drop the name-map entries of objects that are gone.
func _retire_objects(g):
	for o in g.objects:
		if is_instance_valid(o):
			o.free()
	g.objects = []
	g.active_objects = []
	for fx in g.effects:
		if is_instance_valid(fx):
			fx.free()
	g.effects = []
	# Effects and objects can also drop out of those lists without being freed (an effect that
	# leaves `effects` on its own, a disabled projectile), and pile up over thousands of previews
	# until Godot's memory pools run out. A fresh game has both containers empty.
	for container in [g.fx_node, g.objects_node]:
		for c in container.get_children():
			c.free()
	for k in g.objs_map.keys():
		if k != "P1" and k != "P2":
			g.objs_map.erase(k)


# --- divergence monitoring -----------------------------------------------------------------------

func _check_divergence(predicted, actual, moves):
	var diffs = []
	_diff(predicted, actual, "", diffs)
	div.compared += 1
	if diffs.empty():
		return
	div.diverged += 1
	var entry = {"turn": turn, "moves": moves, "diffs": diffs}
	for d in diffs:
		var field = d.split(":")[0]
		div.fields[field] = div.fields.get(field, 0) + 1
	div.recent.append(entry)
	if div.recent.size() > 20:
		div.recent.pop_front()
	if divergence_log != "":
		var f = File.new()
		if f.open(divergence_log, File.READ_WRITE if f.file_exists(divergence_log) else File.WRITE) == OK:
			f.seek_end()
			f.store_line(JSON.print(entry))
			f.close()


func _diff(a, b, path, out):
	if typeof(a) == TYPE_DICTIONARY and typeof(b) == TYPE_DICTIONARY:
		var keys = {}
		for k in a:
			keys[k] = true
		for k in b:
			keys[k] = true
		for k in keys:
			_diff(a.get(k), b.get(k), "%s.%s" % [path, k], out)
	elif typeof(a) == TYPE_ARRAY and typeof(b) == TYPE_ARRAY and a.size() == b.size():
		for i in range(a.size()):
			_diff(a[i], b[i], "%s[%d]" % [path, i], out)
	elif typeof(a) != typeof(b) or str(a) != str(b):
		out.append("%s: preview %s, step %s" % [path, str(a).substr(0, 60), str(b).substr(0, 60)])


# Debug: copy the real match into the preview game and list script variables that still differ.
func copy_diff():
	_ghost_from_real()
	var out = []
	for pair in [[real.p1, ghost.p1], [real.p2, ghost.p2]]:
		out += _var_diff(pair[0], pair[1], pair[0].name)
		out += _var_diff(pair[0].current_state(), pair[1].current_state(), pair[0].name + "/" + pair[0].current_state().name)
	return out


func _var_diff(a, b, prefix):
	var out = []
	for n in _vars(a):
		var va = a.get(n)
		if not _plain(va) or FIGHTER_SKIP.has(n):
			continue
		var vb = b.get(n)
		if typeof(va) != typeof(vb) or str(va) != str(vb):
			out.append("%s.%s: %s != %s" % [prefix, n, str(va).substr(0, 80), str(vb).substr(0, 80)])
	return out


# --- observations --------------------------------------------------------------------------------

func _winner(g):
	if not g.should_game_end():
		return null
	if g.p1.hp > g.p2.hp:
		return 0
	if g.p2.hp > g.p1.hp:
		return 1
	return null


func _actionable(g):
	if g.should_game_end():
		return [false, false]
	return [not g.p1.busy_interrupt, not g.p2.busy_interrupt]


func _cands_json(cs):
	var out = [[], []]
	for seat in range(2):
		for c in cs[seat]:
			var j = {"id": c.id, "action": c.action, "data": c.data, "extra": c.extra}
			for k in ["factors", "values", "base"]:  # candidate set 4
				if c.has(k):
					j[k] = c[k]
			out[seat].append(j)
	return out


func observation(detail):
	var obs = {
		"turn": turn,
		"done": real.should_game_end(),
		"winner": _winner(real),
		"actionable": _actionable(real),
		"state": _state(real),
		"candidates": _cands_json(cands),
	}
	if detail:
		obs["detail"] = _detail(real)
		obs.detail["entered"] = entered_real
	return obs


# The protocol's state: everything a bot or value network needs from a position. Velocities are
# floats here; `_detail` adds the exact fixed-point strings for agreement checks.
func _state(g):
	var objects = []
	for o in g.objects:
		if is_instance_valid(o) and not o.disabled:
			objects.append(_object(o))
	return {"tick": g.current_tick, "time_left": g.time - g.current_tick, "game_length": g.time,
		"players": [_player(g.p1), _player(g.p2)], "objects": objects}


func _player(p):
	var pos = p.get_pos()
	var vel = p.get_vel()
	var st = p.current_state()
	return {
		"character": str(match_data.selected_characters[p.id].name),
		"hp": p.hp, "max_hp": p.MAX_HEALTH,
		"x": pos.x, "y": pos.y, "vx": float(vel.x), "vy": float(vel.y),
		"facing": p.get_facing_int(), "state": st.state_name, "state_tick": st.current_tick,
		"hitlag": p.hitlag_ticks, "busy_interrupt": p.busy_interrupt, "grounded": p.is_grounded(),
		"stance": str(p.stance),
		"meter": p.supers_available, "meter_partial": p.super_meter,
		"burst": p.bursts_available, "burst_partial": p.burst_meter,
		"combo": p.combo_count, "air_options": p.air_option_bar, "feints": p.feints,
		"state_length": st.anim_length, "state_iasa": st.get("iasa_at"), "hitstun": st.get("hitstun"),
		"raw": _raw(p),
		# The current move as submitted (aim vector, parry timing and height...; null when it holds
		# object references, as hurt states do), DI, moves used this combo (staling), hitboxes.
		"state_data": _dup(st.data) if _plain(st.data) else null,
		"di": _dup(p.current_di), "combo_moves_used": _dup(p.combo_moves_used),
		"hitboxes": _hitboxes(st),
	}


# Every hitbox of a state, active or not: start_tick/active_ticks are state ticks, dir_x is in
# the hitbox's own facing (+1 right, -1 left; only up to date while active), fixed-point strings.
func _hitboxes(st):
	var out = []
	if st == null:
		return out
	for c in st.get_children():
		if c is Hitbox:
			out.append({"active": c.active, "start_tick": c.start_tick, "active_ticks": c.active_ticks,
				"damage": c.damage, "dir_x": c.dir_x, "dir_y": c.dir_y, "knockback": c.knockback,
				"hit_height": c.hit_height, "facing": 1 if c.facing == "Right" else -1})
	return out


# Every plain value the game itself declares as simulation state (BaseObj.state_variables plus
# each character's extras: momentum, ammo, Sadness penalty, air movements, combo proration...), so
# features can be chosen offline from recorded games. Identity and stage constants are left out.
# A fighter's `raw` holds only values that differ from its values at the start of the match (most
# never change); the match's first observation carries the full start values as `raw_start`.
const RAW_SKIP = {"id": true, "name": true, "obj_name": true, "creator_name": true, "match_seed": true,
	"game_tick": true, "current_tick": true, "stage_width": true, "ceiling_height": true, "has_ceiling": true,
	"default_hurtbox": true, "use_platforms": true, "trail_hp": true, "is_color_active": true,
	"can_update_sprite": true, "last_object_hit": true, "gravity": true, "ground_friction": true,
	"air_friction": true, "max_ground_speed": true, "max_air_speed": true, "max_fall_speed": true,
	"damages_own_team": true, "always_parriable": true, "has_projectile_parry_window": true,
	"gravity_enabled": true, "aerial_attack_immune": true, "grounded_attack_immune": true}
const RAW_SKIP_PREFIX = ["style_aura_"]   # cosmetic (style meter visuals)


func _raw(o):
	var full = _raw_full(o)
	if not o.is_in_group("Fighter"):
		return full
	var start = raw_start[o.id - 1]
	var out = {}
	for k in full:
		if not start.has(k) or start[k] != full[k]:
			out[k] = full[k]
	return out


func _raw_full(o):
	var out = {}
	for name in o.state_variables:
		if RAW_SKIP.has(name) or out.has(name) or name.begins_with(RAW_SKIP_PREFIX[0]):
			continue
		var v = o.get(name)
		match typeof(v):
			TYPE_BOOL, TYPE_INT, TYPE_REAL, TYPE_STRING:
				out[name] = v
	return out


func _object(o):
	var pos = o.get_pos()
	var st = o.current_state()
	var d = {
		# An object's id is its owner's player id (1 or 2); obj_name is a per-match spawn counter.
		"name": str(o.obj_name), "kind": o.filename.get_file().get_basename(), "owner": o.id,
		"x": pos.x, "y": pos.y,
		"state": st.state_name if st else "", "state_tick": st.current_tick if st else -1,
		"raw": _raw(o),
		"hitboxes": _hitboxes(st),
	}
	if o.has_method("get_vel"):
		var vel = o.get_vel()
		d["vx"] = float(vel.x)
		d["vy"] = float(vel.y)
	return d


# A fuller snapshot for agreement, determinism and divergence checks: exact velocities.
func _detail(g):
	var s = _state(g)
	for i in range(2):
		var p = [g.p1, g.p2][i]
		s.players[i]["vx"] = str(p.get_vel().x)
		s.players[i]["vy"] = str(p.get_vel().y)
	return s


# --- debug: move listing and probes (Candidates.describe / probe) --------------------------------

func describe(seat):
	return gens[seat].describe()


# Everything each seat can choose now, read off the game's own action menu (ActionSpace.gd).
var action_space_node = null


func action_space():
	if action_space_node == null:
		action_space_node = load(get_script().resource_path.get_base_dir() + "/ActionSpace.gd").new()
		add_child(action_space_node)
		action_space_node.setup(real)
	return action_space_node.describe()


# Candidate set 4: a copy of candidate `spec.candidate` with factors overridden (Candidates.expand),
# appended to this real decision's list for preview and step.
func expand(seat, spec):
	if not gens[seat].has_method("expand"):
		return {"error": "seat %d uses candidate set %s, which has no factors" % [seat, candidate_sets[seat]]}
	var base = spec.get("candidate")
	if not (typeof(base) in [TYPE_INT, TYPE_REAL]) or int(base) != base:
		return {"error": "candidate must be a candidate id"}
	var r = gens[seat].expand(int(base), spec.get("extra_override", {}))
	if r.has("error"):
		return r
	var c = r.candidate
	c["id"] = cands[seat].size()
	c["base"] = int(base) if r.recipe.get("base") == null else r.recipe.base
	r.recipe["base"] = c.base
	gens[seat].recipes[c.id] = r.recipe
	cands[seat].append(c)
	return {"id": c.id, "candidate": c}


# Debug widget probe, appended to this real decision's candidate list.
func probe(seat, spec):
	var r = gens[seat].probe(spec)
	if r.has("error"):
		return r
	var c = r.candidate
	c["id"] = cands[seat].size()
	cands[seat].append(c)
	return {"id": c.id, "candidate": c}


# Debug: copy the real match into the preview game and list every script variable that still
# differs anywhere: fighters, all their states and hitboxes, objects and their states. Object
# references compare by name (the copy remaps them onto the preview game's objects).
func deep_diff(limit):
	_ghost_from_real()
	var out = []
	_deep_vars(real, ghost, "game", out, limit, {"is_ghost": true, "objs_map": true})
	var fs = [[real.p1, ghost.p1], [real.p2, ghost.p2]]
	for pair in fs:
		_deep_obj(pair[0], pair[1], pair[0].name, out, limit)
		_pos_diff(pair[0], pair[1], pair[0].name, out)
	var gobjs = {}
	for o in ghost.objects:
		if is_instance_valid(o):
			gobjs[str(o.obj_name)] = o
	for o in real.objects:
		if not is_instance_valid(o) or o.disabled:
			continue
		var key = str(o.obj_name)
		if not gobjs.has(key):
			out.append("object %s missing in preview" % key)
			continue
		out.append("obj:%s is %s state %s" % [key, o.filename.get_file(), o.current_state().name])
		_deep_obj(o, gobjs[key], "obj:" + key, out, limit)
		_pos_diff(o, gobjs[key], "obj:" + key, out)
	return out


# Cached position (what get_pos() reads) and native position.
func _pos_diff(a, b, prefix, out):
	var pa = [a.get_pos(), a.chara.get_data().object_data.position_x, a.chara.get_data().object_data.position_y]
	var pb = [b.get_pos(), b.chara.get_data().object_data.position_x, b.chara.get_data().object_data.position_y]
	if str(pa) != str(pb):
		out.append("%s POS real %s preview %s" % [prefix, str(pa), str(pb)])


func _deep_obj(a, b, prefix, out, limit):
	_deep_vars(a, b, prefix, out, limit, FIGHTER_SKIP)
	for n in a.state_machine.states_map:
		var sa = a.state_machine.states_map[n]
		var sb = b.state_machine.states_map.get(n)
		if sb == null:
			continue
		_deep_vars(sa, sb, prefix + "/" + n, out, limit, {"property_list": true, "host": true})
		var ca = sa.get_children()
		var cb = sb.get_children()
		for i in range(min(ca.size(), cb.size())):
			if ca[i] is Hitbox:
				_deep_vars(ca[i], cb[i], prefix + "/" + n + "/" + ca[i].name, out, limit, {"property_list": true, "host": true})


func _deep_vars(a, b, prefix, out, limit, skip):
	for n in _vars(a):
		if out.size() >= limit:
			return
		if skip.has(n):
			continue
		var va = _norm(a.get(n))
		var vb = _norm(b.get(n))
		# Per-game nodes (sound players, sprites) differ by identity, not content.
		if typeof(va) == TYPE_STRING and typeof(vb) == TYPE_STRING and va.begins_with("<node") and vb.begins_with("<node"):
			continue
		if typeof(va) != typeof(vb) or str(va) != str(vb):
			out.append("%s.%s: real %s, preview %s" % [prefix, n, str(va).substr(0, 70), str(vb).substr(0, 70)])


func _norm(v):
	match typeof(v):
		TYPE_OBJECT:
			if not is_instance_valid(v):
				return "<freed>"
			if v is BaseObj:
				return "<obj %s>" % str(v.obj_name if v.obj_name else v.name)
			if v is Node:
				return "<node %s>" % str(v.name)
			return "<%s>" % v.get_class()
		TYPE_ARRAY:
			var out = []
			for x in v:
				out.append(_norm(x))
			return out
		TYPE_DICTIONARY:
			var out = {}
			for k in v:
				out[str(k)] = _norm(v[k])
			return out
	return v
