extends Control
# Legal moves for one fighter: candidate set 5 (README "Candidate set 5"): candidate set 4
# (docs/PROTOCOL.md `expand`, README "Candidate set 4"; frozen as Candidates4.gd) plus four Cowboy
# additions, each behind a switch (`set5`). With every switch off this generates set 4's lists and
# expand answers exactly; the additions are marked "Set 5" below; any port must compute them the same way.
#
# Legality comes from Legality.gd (the game's own ActionButtons panel answers). Move data
# is never built by hand: the move's own data_ui_scene is instanced and its widgets are set to
# a few values, and the scene's get_data() produces the dict. Extras come from the character's
# PlayerExtra scene and DI from a DI wheel the same way. So every data dict and extra the server
# applies is one the game's UI could have sent (bad data panics the native fixed-point lib).
#
# A candidate is a recipe: move, widget values, extra set, DI direction, feint, reverse. generate()
# lists base recipes (every discrete variant of every legal move); DI, feint, reverse and the extra
# set are factors that expand() changes on any listed candidate, so a search refines them near its
# support instead of the list carrying every combination.
#
# Widget values (variant 0 is always the widget's default):
#   XY plot   forward, forward-up 45, up, back, forward-down 45, back-up 45, down at full length,
#             then the same 7 at half length; the plot's own limits apply, duplicates collapse
#   8-way     each enabled direction, starting forward, forward-up, forward-down, up, down, back,
#             back-up, back-down, neutral
#   slider    0-100 (dash distance): 100, 0, 60, 20, 80, 40 (one per timing step); else max, min, mid
#   count     max, min, midpoint            checkbox  toggled
#   dropdown  each other enabled item       block height  high, low
# Mode widgets (block height, checkboxes, dropdowns) combine with each other and with every
# variant of the other widgets (low block x each parry timing); the other widgets (XY, 8-way,
# sliders, counts) vary one at a time.

# Generation constants come from candidate_rules_v5.json (set 4's rules plus set 5's; shared with
# ports; hashed into hello's candidate_rules with Candidates4.gd). Names kept from when they were consts.
var XY_ANGLES = []
var XY_LENGTHS = []
var EXTRA_XY_ANGLES = []
var WAY_DIRS = []
var DASH_SLIDER_VALUES = []
var DI_VALUES = []
var HURT_CONTINUE_BASE_DI = []
var CATEGORY_ORDER = []
var MODE_KINDS = []
const DI_FULL = 100

var PARRY_TIMINGS = []
var R5 = {}              # set 5's constants (candidate_rules_v5.json beyond set 4's keys)
const PATH_DEPTH = 4     # set 5's aim / di_fine paths stop at this depth (set5_expand)

var fighter = null
var max_candidates = 0
var button_states = []   # states shown in the action menu, in the UI's creation order
var data_nodes = {}      # state name -> this epoch's freshly instanced data_ui_scene (lazy)
var data_defaults = {}   # state name -> its node's get_data() when opened this epoch
var stub = null
var extra_node = null    # the character's PlayerExtra scene (Wizard hover, Robot fly, ...)
var set5 = {}            # set 5's switches: draw_pairs (A), gun_aim_fine (B), bullet_di (C), draw_base (R)
var character = ""       # this seat's character name (set 5 changes only set5_characters)
var game = null          # the game this fighter plays in (set 5's bullet owner check reads its objects)
var set6 = {}            # set 6 for this seat's character: {pairs, clicks, triples, toggle_only, juke_sections,
                         # pool, pool_dirs, switch}; {} = off
var set7 = {}            # set 7's data-widget switches: {xy_fine, di_fine_all, multi_xy_base, multi_xy}
var panel_slots = {}     # set 6: move name -> its panel slots this decision
var pool_slots = {}      # set 7 L1: move name -> its pool-only triples this decision
var switch_slots = {}    # set 7 switch sets: move name -> its switch sets (explicit) this decision
var aimed_cache = {}     # set 7 L2: move name -> its aimed widgets this decision
var turn_sections = null # set 7 juke_sections: the turn-open panel's JukeDir sections (null: not read yet)

# This decision's lookups (rebuilt by generate()).
var recipes = {}         # candidate id -> recipe (generated and expanded candidates)
var extra_sets = {}      # move name ("Continue") -> its extra sets: [{extra widget index: value as for probe}]
var di_values = []       # DI dicts by factor index: neutral, then each magnitude x the 8 directions
var extra_cache = {}     # "state:k" -> the extra panel's get_extra() with set k applied
var factor_cache = {}    # state name -> _factors(state)
var d5 = {}              # set 5's per-decision state (_new_d5)


class ButtonsStub:
	extends Reference
	# Data scenes peek at the opponent's selected button (Parry pre-fills its timing).
	var opponent_action_buttons = self
	var current_button = null

func setup(fighter_, max_candidates_, set5_ = {}, character_ = "", game_ = null, set6_ = {}):
	fighter = fighter_
	max_candidates = max_candidates_  # 0: no budget, every variant
	set5 = set5_
	set6 = _set6_for(set6_, character_)
	set7 = _set7_for(set6_)
	character = character_
	game = game_
	_load_rules()
	# Data scenes read is_visible_in_tree(), so this container stays "visible" (nothing renders).
	visible = true
	rect_position = Vector2(-10000, -10000)
	stub = ButtonsStub.new()
	for category in fighter.action_cancels:
		for state in fighter.action_cancels[category]:
			if state.show_in_menu and not state in button_states:
				button_states.append(state)
	_new_epoch()


func _load_rules():
	var f = File.new()
	f.open(get_script().resource_path.get_base_dir() + "/candidate_rules_v5.json", File.READ)
	var rules = JSON.parse(f.get_as_text()).result
	f.close()
	XY_ANGLES = rules.xy_angles
	XY_LENGTHS = rules.xy_lengths
	EXTRA_XY_ANGLES = rules.extra_xy_angles
	WAY_DIRS = rules.way_dirs
	DASH_SLIDER_VALUES = rules.dash_slider_values
	DI_VALUES = rules.di_values
	HURT_CONTINUE_BASE_DI = rules.hurt_continue_base_di
	CATEGORY_ORDER = rules.category_order
	MODE_KINDS = rules.mode_kinds
	PARRY_TIMINGS = rules.parry_timings
	R5 = rules


# Purity: every generate() / expand() / default_candidate() starts a new epoch, dropping every
# widget scene and building fresh ones (defaults) on demand, so what a list contains depends only
# on the game state, never on earlier turns, previews, probes or half-restored widgets.
func _new_epoch():
	legal_epoch += 1
	for node in data_nodes.values():
		_drop(node)
	data_nodes = {}
	data_defaults = {}
	if extra_node != null:
		_drop(extra_node)
		extra_node = null
	if di_node != null:
		_drop(di_node)
		di_node = null
	if fighter.player_extra_params_scene:
		extra_node = fighter.player_extra_params_scene.instance()
		extra_node.set_fighter(fighter)
		add_child(extra_node)


func _drop(node):
	remove_child(node)
	node.queue_free()


func _has_data(state):
	return state != null and state.data_ui_scene != null


# Returns this decision's candidates: [{id, action, data, extra, factors, values}, ...], Continue
# first. Order (a port must reproduce it):
#   1. Continue: neutral DI; while hurt (and DI is enabled) then the rules' hurt_continue_base_di
#      (full-strength forward, back, up, down); then Continue with each of its extra sets 1..K-1.
#   2. Every move legal now, in the action menu's order, each with its data variants (see
#      _data_recipes) on extra set 0.
#   3. Moves legal only with an extra set (Cowboy's shift / drift change grounded/aerial input):
#      for each of Continue's extra sets k in order, each newly legal move with its data variants
#      on the same set in that move's own list (_sets_for).
# With max_candidates > 0 the list keeps Continue's variants and every move's first variant, then
# adds variants round-robin over the categories in CATEGORY_ORDER except Continue (each category
# round-robin over its moves in list order) until the budget is spent; kept candidates stay in
# list order. (Continue was in the round-robin before; all its variants are kept anyway, so its
# turn only skipped. It is left out now; the kept lists are the same.)
func generate():
	_new_epoch()
	_activate_extra()
	recipes = {}
	extra_cache = {}
	factor_cache = {}
	di_values = _di_values()
	extra_sets = {}
	panel_slots = {}
	pool_slots = {}
	switch_slots = {}
	aimed_cache = {}
	turn_sections = null
	_new_d5()
	var cont_sets = _sets_for(null)
	var moves = []   # [{state, recipes}]
	var cont = [_recipe(null, {}, 0, null)]
	if fighter.di_enabled and fighter.current_state().is_hurt_state:
		for d in HURT_CONTINUE_BASE_DI:
			var r = _recipe(null, {}, 0, null)
			r.di = int(d)
			cont.append(r)
	for k in range(1, cont_sets.size()):
		cont.append(_recipe(null, {}, k, null))
	moves.append({"state": null, "recipes": cont})
	var listed = {}
	for state in legal_states():
		moves.append({"state": state, "recipes": _data_recipes(state, 0)})
		listed[state.name] = true
	for k in range(1, cont_sets.size()):
		if not _changes_input(cont_sets[k]):
			continue
		_activate_extra()
		var err = _apply_set(cont_sets[k], null)
		var legal_k = legal_states() if err == "" else []
		_activate_extra()
		for state in legal_k:
			if listed.has(state.name):
				continue
			# The same toggles in the move's own list (selecting it may hide or disable them).
			var j = _find_set(_sets_for(state), cont_sets[k])
			if j > 0 and _validate_recipe(_recipe(state, {}, j, null)) == "":
				moves.append({"state": state, "recipes": _data_recipes(state, j)})
				listed[state.name] = true
	if set7.get("multi_xy_base", false):
		# Set 7 L3: each multi-XY move's same-value entries after its variants (before the budget).
		for m in moves:
			if m.state != null:
				_same_value_entries(m.state, m.recipes)
	if _s5("draw_base"):
		# Set 5 R: each move's Draw base entry after its variants (before the budget).
		for m in moves:
			if m.state != null and not m.recipes.empty():
				var r = _draw_entry(m.recipes[0])
				if r != null:
					m.recipes.append(r)
	var keep = _budget(moves)
	var out = []
	for i in range(moves.size()):
		for v in range(moves[i].recipes.size()):
			if keep[i][v]:
				var r = moves[i].recipes[v]
				var c = _build(r)
				c["id"] = out.size()
				recipes[c.id] = r
				out.append(c)
	if d5.applies:
		if _s5("gun_aim_fine"):
			_set_fans(out)
		for c in out:
			_set5_meta(c, recipes[c.id])
	if set7.get("xy_fine", false) or set7.get("di_fine_all", false):
		for c in out:
			_set7_meta(c, recipes[c.id])
	if set7.get("multi_xy", false):
		for c in out:
			_l3_meta(c, recipes[c.id])
	for c in out:
		d5.keys.append(_cand_key(c))
	return out


func _budget(moves):
	var keep = []
	var total = 0
	for m in moves:
		var row = []
		for v in range(m.recipes.size()):
			row.append(max_candidates <= 0 or v == 0 or m.state == null)
			total += 1 if row[v] else 0
		keep.append(row)
	if max_candidates <= 0:
		return keep
	var by_cat = {}
	for cat in CATEGORY_ORDER:
		by_cat[cat] = []
	for i in range(moves.size()):
		if moves[i].state != null:  # Continue's variants are all kept already
			by_cat[_category(moves[i].state)].append(i)
	var next_v = []
	for m in moves:
		next_v.append(1)
	var grew = true
	while total < max_candidates and grew:
		grew = false
		for cat in CATEGORY_ORDER:
			if total >= max_candidates:
				break
			# One variant for this category: from its next move (round-robin) that still has one.
			var ms = by_cat[cat]
			for _n in range(ms.size()):
				var i = ms.pop_front()
				ms.push_back(i)
				if next_v[i] < moves[i].recipes.size():
					keep[i][next_v[i]] = true
					next_v[i] += 1
					total += 1
					grew = true
					break
	return keep


func _recipe(state, widgets, k, data):
	return {"state": state, "widgets": widgets, "extra_set": k, "di": 0, "feint": false, "reverse": false,
		"data": data, "aim": 0, "di_fine": 0, "xy": 0, "di_fine_all": 0, "xy_grid": 0}


# The factors expand() may change on a candidate of this move, for the candidate's `factors`.
func _factors(state):
	var f = {}
	if fighter.di_enabled:
		f["di"] = DI_VALUES.size()
	if state != null and _feintable(state):
		f["feint"] = true
	if state != null and state.flip_allowed() and state.reversible:
		f["reverse"] = true
	var n_sets = _n_extra_sets(state)
	if n_sets > 1:
		f["extra_set"] = n_sets
	return f


func _factors_cached(state):
	var key = _name(state)
	if not factor_cache.has(key):
		factor_cache[key] = _factors(state)
	return factor_cache[key].duplicate()


func _feintable(state):
	_extra_select(state)
	return state.can_feint() and (extra_node == null or extra_node.can_feint)


# The candidate dict for a recipe: data from the move's widgets (rebuilt flipped for reverse),
# extras from the extra panel with the recipe's extra set, DI from the DI wheel.
# Set 5: an aim or di_fine path that doesn't exist returns {"error": why} (expand only).
func _build(r):
	var data = r.data
	if r.state != null and r.xy != 0:
		# Set 7 L2: the widget settings with the xy paths' widget values set last.
		var settings = _xy_settings(r)
		if typeof(settings) == TYPE_STRING:
			return {"error": settings}
		data = _data_for(r.state, settings, r.reverse)
	elif r.state != null and _has_data(r.state) and (r.reverse or data == null):
		data = _data_for(r.state, r.widgets, r.reverse)
	if r.aim != 0:
		data = _aim_data(r, data)
		if typeof(data) == TYPE_STRING:
			return {"error": data}
	var extras
	var layer = _layer_slot(r.state, r.extra_set)
	if layer != null:
		extras = call("_%s_extras" % layer[0], r.state, layer[1])
	else:
		extras = _extras_k(r.state, r.extra_set)
	var di
	if r.di_fine != 0 or r.di_fine_all != 0:
		var v = _di_vec(r)
		if typeof(v) == TYPE_STRING:
			return {"error": v}
		di = {"x": v[0], "y": v[1]}
	else:
		di = di_values[r.di].duplicate()
	var extra = {"DI": di, "reverse": r.reverse, "feint": r.feint, "prediction": -1}
	extra.merge(extras)
	return {"action": "Continue" if r.state == null else r.state.name, "data": data, "extra": extra,
		"factors": _factors_cached(r.state),
		"values": {"di": r.di, "feint": r.feint, "reverse": r.reverse, "extra_set": r.extra_set}}


# A copy of the candidate `base_id` (generated or expanded this decision) with factors overridden:
# {"di": DI factor index (rules di_layout), "feint": bool, "reverse": bool, "extra_set": index}. Every override is
# checked against what a player could select now: DI needs DI enabled, feint and reverse need the
# move's buttons enabled, an extra set must be applicable with the move selected and keep the move
# legal. Returns {"candidate": {...}, "recipe": r} or {"error": message}.
func expand(base_id, override):
	if not recipes.has(base_id):
		return {"error": "candidate %s was not generated or expanded this decision" % str(base_id)}
	if typeof(override) != TYPE_DICTIONARY:
		return {"error": "extra_override must be an object"}
	_new_epoch()  # purity: an expanded candidate depends only on the state and its recipe
	var r = recipes[base_id].duplicate()
	for key in override:
		var on5 = (key == "aim" and _s5("gun_aim_fine")) or (key == "di_fine" and _s5("bullet_di"))
		var on7 = (key == "xy" and set7.get("xy_fine", false)) or (key == "di_fine_all" and set7.get("di_fine_all", false)) \
				or (key == "xy_grid" and set7.get("multi_xy", false))
		if not key in ["di", "feint", "reverse", "extra_set"] and not on5 and not on7:
			return {"error": "unknown factor %s" % str(key)}
	if override.has("di"):
		var d = override.di
		if not (typeof(d) in [TYPE_INT, TYPE_REAL]) or int(d) != d or d < 0 or d >= di_values.size():
			return {"error": "di must be an index into the rules' di_values (below %d)" % di_values.size()}
		if int(d) != 0 and not fighter.di_enabled:
			return {"error": "DI is disabled"}
		r.di = int(d)
	for f in ["feint", "reverse"]:
		if override.has(f):
			if typeof(override[f]) != TYPE_BOOL:
				return {"error": "%s must be true or false" % f}
			r[f] = override[f]
	# Set 5: factor paths (base-3 aim, base-5 di_fine digits, least significant first, none 0).
	for f in ["aim", "di_fine"]:
		if override.has(f):
			var x = override[f]
			if not (typeof(x) in [TYPE_INT, TYPE_REAL]) or int(x) != x or x < 0 or x >= 1e9:
				return {"error": "%s must be a factor path (a non-negative integer)" % f}
			var b = 3 if f == "aim" else 5
			var digits = _path_digits(int(x), b)
			if digits.size() > PATH_DEPTH or 0 in digits:
				return {"error": "%s %d is not a path of at most %d steps (base-%d digits, none 0)" % [f, int(x), PATH_DEPTH, b]}
			if f == "di_fine" and not d5.owns_bullet and int(x) != 0:
				return {"error": "di_fine needs a live bullet of this seat's"}
			r[f] = int(x)
	# Set 7 L2 / L2-DI: xy (packed per-widget paths, checked against the move's aimed widgets when built) and
	# di_fine_all (a base-5 path applied after di_fine).
	for f in ["xy", "di_fine_all"]:
		if override.has(f):
			var x = override[f]
			if not (typeof(x) in [TYPE_INT, TYPE_REAL]) or int(x) != x or x < 0 or x >= (1 << (XY_BITS * XY_SLOTS)):
				return {"error": "%s must be a factor path (a non-negative integer)" % f}
			if f == "di_fine_all":
				var digits = _path_digits(int(x), 5)
				if digits.size() > DI_ALL_DEPTH or 0 in digits:
					return {"error": "di_fine_all %d is not a path of at most %d steps (base-5 digits, none 0)" % [int(x), DI_ALL_DEPTH]}
				if int(x) != 0 and not fighter.di_enabled:
					return {"error": "DI is disabled"}
			r[f] = int(x)
	if override.has("extra_set"):
		# Pool-only values (L1's triples, switch sets) are numbered after every listed value.
		var k = override.extra_set
		var n_sets = _n_all_sets(r.state)
		if not (typeof(k) in [TYPE_INT, TYPE_REAL]) or int(k) != k or k < 0 or k >= n_sets:
			return {"error": "extra_set must be below %d for %s" % [n_sets, _name(r.state)]}
		r.extra_set = int(k)
	if override.has("xy_grid"):
		var v = override.xy_grid
		if not (typeof(v) in [TYPE_INT, TYPE_REAL]) or int(v) != v or v < 0 or v > 4294967295:
			return {"error": "xy_grid must be a packed grid index (a non-negative integer)"}
		var gerr = _apply_xy_grid(r, int(v))
		if gerr != "":
			return {"error": gerr}
	var err = _validate_recipe(r)
	if err != "":
		return {"error": err}
	var c = _build(r)
	if c.has("error"):
		return c
	if d5.applies:
		_set5_meta(c, r)
	_set7_meta(c, r)
	_l3_meta(c, r)
	var key = _cand_key(c)
	if r.aim != 0 or r.di_fine != 0 or r.extra_set >= _sets_for(r.state).size() or r.xy != 0 or r.di_fine_all != 0 \
			or r.xy_grid != 0:
		# Set 5's own expansions are refused when they repeat a listed or expanded candidate.
		var k = d5.keys.find(key)
		if k >= 0:
			return {"error": "same candidate as %d" % k}
	d5.keys.append(key)
	return {"candidate": c, "recipe": r}


# Validate the complete combination with its extras applied, not each factor in isolation.
func _validate_recipe(r):
	var layer = _layer_slot(r.state, r.extra_set)
	if layer != null:
		if layer[0] == "switch":
			# A switch set is valid when the move is legal with the whole set made (its widgets may only take
			# their values together: a toggle and the plot it reveals).
			return _validate_explicit(r, layer[1])
		# A combined set (set 5's Draw pair, set 6's panel set, L1's pool triple) is valid when the move validates
		# with each of its own sets.
		var slot = layer[1]
		for k in slot:
			var rk = r.duplicate()
			rk.extra_set = k
			var e = _validate_recipe(rk)
			if e != "":
				return "combined extra set %d (sets %s): %s" % [r.extra_set, str(slot), e]
		return ""
	if r.di_fine != 0 or r.di_fine_all != 0:
		var v = _di_vec(r)
		if typeof(v) == TYPE_STRING:
			return v
	if r.xy != 0:
		var s = _xy_settings(r)
		if typeof(s) == TYPE_STRING:
			return s
	_activate_extra()
	var err = _apply_extra_set(r.extra_set, r.state)
	if err == "" and r.state != null and not r.state in legal_states():
		err = "%s is not legal with extra set %d" % [_name(r.state), r.extra_set]
	if err == "" and r.feint and (r.state == null or not _feintable(r.state)):
		err = "feint is disabled for %s" % _name(r.state)
	if err == "" and r.reverse and (r.state == null or not (r.state.flip_allowed() and r.state.reversible)):
		err = "reverse is disabled for %s" % _name(r.state)
	_activate_extra()
	return err


func _name(state):
	return "Continue" if state == null else state.name


func _category(state):
	if state == null:
		return "Continue"
	var c = state.get_ui_category()
	return c if c in CATEGORY_ORDER else "Other"


# Candidate 0 of generate(): Continue with neutral DI and the default extras. Used for the
# protocol's default continuation without paying for a full generate().
func default_candidate():
	_new_epoch()
	_activate_extra()
	di_values = _di_values()
	var extra = {"DI": di_values[0].duplicate(), "reverse": false, "feint": false, "prediction": -1}
	extra.merge(_extra_dict(null, 0) if extra_node else {})
	return {"id": 0, "action": "Continue", "data": null, "extra": extra}


func _di_values():
	# The rules' exact DI table, entered on a DI wheel and read back as the menu would send it.
	var out = []
	var dn = _ensure_di()
	var f = fighter.get_opponent_dir()
	for v in DI_VALUES:
		dn.reset()
		if v[0] != 0 or v[1] != 0:
			dn.update_value(Vector2(v[0] * f, v[1]) / 100.0 * dn.panel_radius, false, true)
		var d = dn.get_data().duplicate()
		if int(d.x) != int(v[0] * f) or int(d.y) != int(v[1]):
			push_error("YomiBench: DI wheel read back %s for %s" % [str(d), str(v)])
		out.append(d)
	return out


# --- candidate set 5 (README "Candidate set 5") ---------------------------------------------------
# Four Cowboy additions, each behind a switch: A draw_pairs (Draw + Shift/Rift/Drift/Sight extra
# sets, expand only), B gun_aim_fine (the `aim` factor), C bullet_di (the `di_fine` factor) and
# R draw_base (each move's Draw set as a base entry). A seat of any other character gets set 4's
# lists and answers. Aims and DI values are integer percents of the plot / wheel radius, the values
# those widgets send; the real `probe` reproduces every one.

# This decision's set-5 state (generate() starts it; expand() reads it).
func _new_d5():
	var toward = fighter.get_opponent_dir()
	if toward == 0:
		toward = fighter.get_facing_int()
	d5 = {
		"applies": character in R5.set5_characters,
		"toward": toward,               # toward the opponent (screen x), the facing when level
		"owns_bullet": _owns_bullet(),  # C
		"fans": {},                     # B: gun move name -> {disk, lo, hi}
		"pairs": {},                    # A: move name -> [[Draw set, toggle set], ...]
		"keys": [],                     # every candidate listed or expanded, as _cand_key
	}


# A switch, when set 5 applies to this seat.
func _s5(name):
	return d5.get("applies", false) and set5.get(name, false) == true


func _cand_key(c):
	return JSON.print([c.action, c.data, c.extra])


func _on(e, key):
	var v = e.get(key)
	return typeof(v) == TYPE_BOOL and v


# A Draw set's extras: the draw toggle on, every pair toggle off.
func _is_draw(e):
	if not _on(e, R5.draw_toggle):
		return false
	for t in R5.pair_toggles:
		if _on(e, t):
			return false
	return true


# _extra_dict(state, k) through this decision's cache (as _build reads it).
func _extras_k(state, k):
	var key = ("" if state == null else state.name) + ":" + str(k)
	if not extra_cache.has(key):
		extra_cache[key] = _extra_dict(state, k)
	return extra_cache[key]


# A: a move's pair sets (cached per move): for each pair toggle t in rules order, when the move's
# own sets hold a Draw set and a t set (t on, the draw toggle and the other pair toggles off):
# [the first Draw set, the first t set]. Continue has none. Pair sets are numbered after the move's
# own sets, so set 4's extra_set indices keep their meaning.
func _pair_slots(state):
	if state == null or not _s5("draw_pairs"):
		return []
	if d5.pairs.has(state.name):
		return d5.pairs[state.name]
	var n = _sets_for(state).size()
	var ex = []
	for k in range(n):
		ex.append(_extras_k(state, k))
	var out = []
	if n > 1:
		var d = -1
		for k in range(n):
			if _is_draw(ex[k]):
				d = k
				break
		if d >= 0:
			for t in R5.pair_toggles:
				for k in range(n):
					if not _on(ex[k], t) or _on(ex[k], R5.draw_toggle):
						continue
					var other = false
					for u in R5.pair_toggles:
						if u != t and _on(ex[k], u):
							other = true
					if not other:
						out.append([d, k])
						break
	d5.pairs[state.name] = out
	return out


# --- extra-set layers --------------------------------------------------------------------------------------
# A move's extra_set index runs over its own sets (set 4's _sets_for), then each layer's combined sets, in LAYERS
# order: set 5's Draw pairs, then set 6's panel sets. A layer <key> is two functions: _<key>_slots(state), its
# slots this decision (each a list of the move's own set indices), and _<key>_extras(state, slot), the extras the
# slot gives. Validation is shared (_validate_recipe: valid when the move validates with each own set of the
# slot). A later candidate set adds a layer at the end, so every earlier index keeps its meaning.
const LAYERS = ["pair", "panel"]
# Set 7's pool-only layers, numbered after every listed value and never counted in factors.extra_set (so refinement
# and exploration never walk them): L1's triples, then the switch sets.
const POOL_LAYERS = ["pool", "switch"]


func _n_extra_sets(state):
	var n = _sets_for(state).size()
	for key in LAYERS:
		n += call("_%s_slots" % key, state).size()
	return n


# Every extra_set value of a move, the pool-only ones included.
func _n_all_sets(state):
	var n = _n_extra_sets(state)
	for key in POOL_LAYERS:
		n += call("_%s_slots" % key, state).size()
	return n


# [layer key, slot] for an index past the move's own sets; null for an own set.
func _layer_slot(state, k):
	k -= _sets_for(state).size()
	if k < 0:
		return null
	for key in LAYERS + POOL_LAYERS:
		var slots = call("_%s_slots" % key, state)
		if k < slots.size():
			return [key, slots[k]]
		k -= slots.size()
	return null


# A: a pair set's extras: the Draw set's, with the toggle set's pair_overlay_keys laid over.
func _pair_extras(state, slot):
	var e = _extras_k(state, slot[0]).duplicate(true)
	var single = _extras_k(state, slot[1])
	for key in R5.pair_overlay_keys:
		if single.has(key):
			e[key] = single[key]
	return e


# R: a move's Draw base entry: its first variant with the first extra set, from 1, that is a Draw
# set and validates (default factors); null if none.
func _draw_entry(r0):
	var n = _sets_for(r0.state).size()
	for k in range(1, n):
		if not _is_draw(_extras_k(r0.state, k)):
			continue
		var r = _recipe(r0.state, r0.widgets, k, r0.data)
		if _validate_recipe(r) == "":
			return r
	return null


# C: this seat owns a live bullet object (not disabled; reclaimed husks have left game.objects).
func _owns_bullet():
	if game == null:
		return false
	var me = 1 if fighter == game.p1 else 2
	for o in game.objects:
		if not is_instance_valid(o) or o.disabled or o.id != me:
			continue
		if o.filename.get_file().get_basename() in R5.bullet_objects:
			return true
	return false


func _path_digits(p, b):
	var out = []
	while p > 0:
		out.append(p % b)
		p /= b
	return out


# Python's round(): half to even.
func _round_even(x):
	var f = floor(x)
	var d = x - f
	if d > 0.5 or (d == 0.5 and fmod(f, 2.0) != 0.0):
		f += 1.0
	return int(f)


func _hypot(x, y):
	return sqrt(float(x) * x + float(y) * y)


# Degrees in (-180, 180], counterclockwise from toward the opponent (screen y down).
func _norm_angle(a):
	var m = fmod(a + 180.0, 360.0)
	if m < 0:
		m += 360.0
	a = m - 180.0
	return 180.0 if a == -180.0 else a


func _rel_angle(x, y):
	return _norm_angle(rad2deg(atan2(-float(y), float(x) * d5.toward)))


func _mirror(a):
	return _norm_angle(180.0 - a)


# B: where an aim lives in a move's data: "" the data is {x, y}, "Direction" its Direction is;
# null if neither is an integer vector longer than 1.5.
func _xy_slot(data):
	if typeof(data) != TYPE_DICTIONARY:
		return null
	for slot in ["", "Direction"]:
		var d = data if slot == "" else data.get("Direction")
		if typeof(d) != TYPE_DICTIONARY:
			continue
		var x = d.get("x")
		var y = d.get("y")
		if typeof(x) in [TYPE_INT, TYPE_REAL] and typeof(y) in [TYPE_INT, TYPE_REAL] and int(x) == x and int(y) == y:
			if _hypot(x, y) > 1.5:
				return slot
	return null


# B: one aim step: turn the aim (x, y) by -step (digit 1) or +step (digit 2) in the candidate's
# unmirrored frame, inside the move's fan, at the same length. [x, y] or null.
func _aim_step(x, y, digit, step, fan, reverse):
	var angle = _rel_angle(x, y)
	var length = _hypot(x, y)
	var base = _mirror(angle) if reverse else angle
	var a = _norm_angle(base - step if digit == 1 else base + step)
	if not (fan.disk or (fan.lo - 0.01 <= a and a <= fan.hi + 0.01)):
		return null
	var out = _mirror(a) if reverse else a
	var rad = deg2rad(out)
	return [_round_even(cos(rad) * length * d5.toward), _round_even(-sin(rad) * length)]


# B: each gun move's fan from this decision's unreversed base candidates' aims (any angle when
# both 0 and 180 degrees appear).
func _set_fans(cands):
	var angles = {}
	for c in cands:
		if not c.action in R5.gun_moves or c.extra.get("reverse") == true:
			continue
		var slot = _xy_slot(c.data)
		if slot == null:
			continue
		var d = c.data if slot == "" else c.data.Direction
		if not angles.has(c.action):
			angles[c.action] = []
		angles[c.action].append(_rel_angle(d.x, d.y))
	for name in angles:
		var zero = false
		var half = false
		var lo = INF
		var hi = -INF
		for a in angles[name]:
			zero = zero or abs(a) < 1.0
			half = half or abs(abs(a) - 180.0) < 1.0
			lo = min(lo, a)
			hi = max(hi, a)
		d5.fans[name] = {"disk": zero and half, "lo": lo, "hi": hi}


# B: a gun candidate's data with its aim path applied, or why not.
func _aim_data(r, data):
	var name = _name(r.state)
	if not d5.fans.has(name):
		return "aim is not a factor of %s" % name
	var fan = d5.fans[name]
	data = data.duplicate(true)
	var digits = _path_digits(r.aim, 3)
	for l in range(digits.size()):
		var slot = _xy_slot(data)
		if slot == null:
			return "%s has no aim" % name
		var d = data if slot == "" else data.Direction
		var v = null
		if digits[l] == 1 or digits[l] == 2:
			v = _aim_step(d.x, d.y, digits[l], R5.gun_aim_step / pow(2, l), fan, r.reverse)
		if v == null:
			return "aim %d leaves %s's aims at step %d" % [r.aim, name, l + 1]
		d.x = v[0]
		d.y = v[1]
	return data


# C: one DI step: digits 1/2 turn the DI by -/+ angle_step / 2^level at the same strength, 3/4
# change its strength by -/+ strength_step / 2^level at the same angle (within (0, max], not to the
# same vector or neutral). [x, y] or null (neutral DI has no steps).
func _di_step(x, y, digit, level):
	var m = _hypot(x, y)
	if m < 1.0:
		return null
	var a = atan2(float(y), float(x))
	var scale = pow(2, level)
	if digit == 1 or digit == 2:
		var step = deg2rad(R5.bullet_di_angle_step / scale)
		var s = -step if digit == 1 else step
		return [_round_even(m * cos(a + s)), _round_even(m * sin(a + s))]
	if digit == 3 or digit == 4:
		var ds = R5.bullet_di_strength_step / scale
		var m2 = m - ds if digit == 3 else m + ds
		if not (0.0 < m2 and m2 <= R5.bullet_di_max_strength):
			return null
		var v = [_round_even(m2 * cos(a)), _round_even(m2 * sin(a))]
		if (v[0] == _round_even(x) and v[1] == _round_even(y)) or (v[0] == 0 and v[1] == 0):
			return null
		return v
	return null


# C: a recipe's DI vector as the extra carries it: its table DI, then its di_fine path; or why not.
func _di_vec(r):
	var v = [int(DI_VALUES[r.di][0]) * fighter.get_opponent_dir(), int(DI_VALUES[r.di][1])]
	var digits = _path_digits(r.di_fine, 5)
	for l in range(digits.size()):
		var n = _di_step(v[0], v[1], digits[l], l)
		if n == null:
			return "di_fine %d has no step %d from DI (%d, %d)" % [r.di_fine, l + 1, v[0], v[1]]
		v = n
	digits = _path_digits(r.di_fine_all, 5)
	for l in range(digits.size()):
		var n = _di_step(v[0], v[1], digits[l], l)
		if n == null:
			return "di_fine_all %d has no step %d from DI (%d, %d)" % [r.di_fine_all, l + 1, v[0], v[1]]
		v = n
	return v


# Set 5's per-candidate factors and values: `aim` (B) on a gun candidate with an aim, `di_fine` (C)
# on every candidate of a seat that owns a live bullet; factors list the paths one step longer that
# exist, values the candidate's own path.
func _set5_meta(c, r):
	if _s5("gun_aim_fine") and d5.fans.has(c.action):
		var slot = _xy_slot(c.data)
		if slot != null:
			var depth = _path_digits(r.aim, 3).size()
			var kids = []
			if depth < PATH_DEPTH:
				var d = c.data if slot == "" else c.data.Direction
				for digit in [1, 2]:
					if _aim_step(d.x, d.y, digit, R5.gun_aim_step / pow(2, depth), d5.fans[c.action], r.reverse) != null:
						kids.append(r.aim + digit * int(pow(3, depth)))
			c.factors["aim"] = kids
			c.values["aim"] = r.aim
	if _s5("bullet_di") and d5.owns_bullet:
		var di = c.extra.get("DI", {})
		var x = int(di.get("x", 0))
		var y = int(di.get("y", 0))
		var depth = _path_digits(r.di_fine, 5).size()
		var kids = []
		if depth < PATH_DEPTH:
			for digit in [1, 2, 3, 4]:
				if _di_step(x, y, digit, depth) != null:
					kids.append(r.di_fine + digit * int(pow(5, depth)))
		c.factors["di_fine"] = kids
		c.values["di_fine"] = r.di_fine


# --- legality (Legality.gd) -------------------------------------------------------------------

var legality = null      # this game's Legality node (YomiMatch._make_gens)
var legal_epoch = 0      # bumped by every new epoch: Legality caches its answers within one


# The menu moves (button_states, in menu order) legal now with the extra panel as it stands.
# fresh: the game state may have changed since this epoch began (YomiMatch._open_turn).
func legal_states(fresh = false):
	if fresh:
		legal_epoch += 1
	return legality.legal_states(self, _current_extra())


# --- extra panel -----------------------------------------------------------------------------

# The extra panel (DI, feint, reverse, the move's own widgets) is reused for every candidate, so before
# each one it's brought back to a known baseline: visible, with its options, only while the fighter can act
# (the player sees no extras otherwise), and every widget at its default. The panel's own methods do the work.
# Selecting the move (`state`) sets which widgets apply to it; options and defaults can undo part of that
# selection, so it's applied before and after them.
func _activate_extra(state = null):
	if extra_node == null:
		return
	var can_act = not fighter.busy_interrupt
	extra_node.visible = can_act
	extra_node.reset()
	_extra_select(state)
	if can_act:
		extra_node.show_options()
	_extra_select(state)


func _current_extra():
	return extra_node.get_extra() if extra_node else {}


func _extra_dict(state, k):
	if extra_node == null:
		return {}
	_activate_extra()
	_apply_extra_set(k, state)
	var e = extra_node.get_extra().duplicate(true)
	_activate_extra()
	return e


# Clicks extra set k (of the move's own list) as a player would after selecting the move;
# returns "" or why not.
func _apply_extra_set(k, state):
	return _apply_set(_sets_for(state)[k], state)


func _apply_set(s, state):
	_activate_extra(state)
	if s.empty() or extra_node == null:
		return ""
	var ws = _extra_widgets()
	for ei in s:
		var err = _set_widget(ws[int(ei)], s[ei], true)
		if err != "":
			return "extra widget %s: %s" % [str(ei), err]
		_extra_select(state)
	return ""


# Whether an extra set changes which ground/air moves are legal (input_grounded / input_aerial).
func _changes_input(s):
	if extra_node == null:
		return false
	_activate_extra()
	var before = _current_extra()
	_apply_set(s, null)
	var after = _current_extra()
	_activate_extra()
	for key in ["input_grounded", "input_aerial"]:
		if before.get(key, false) != after.get(key, false):
			return true
	return false


func _find_set(sets, s):
	var key = JSON.print(s)
	for j in range(sets.size()):
		if JSON.print(sets[j]) == key:
			return j
	return -1


# A move's extra sets this decision (cached per move), enumerated with THAT move selected on a
# reset panel, as the game's UI shows the panel when a player picks the move: [{}] then, in widget
# order, each value of each visible enabled non-toggle widget (an XY at full length every 45
# degrees starting forward, else the data widget values above); then for each visible enabled
# toggle, the toggle clicked, followed by the toggle clicked plus each value of each non-toggle
# widget the click revealed (Ninja's release direction, ...). Sets whose resulting extras equal an
# earlier set's, or that a widget refuses, are dropped. Per move, never a union over moves: a set
# is only offered with a move that allows it.
func _sets_for(state):
	var name = _name(state)
	if not extra_sets.has(name):
		extra_sets[name] = _extra_sets(state)
	return extra_sets[name]


func _extra_sets(state):
	var out = [{}]
	if extra_node == null:
		return out
	_activate_extra(state)
	_extra_select(state)
	var seen = {JSON.print(extra_node.get_extra()): true}
	var ws = _extra_widgets()
	var shown = []
	for w in ws:
		shown.append(w.node.is_visible_in_tree() and not w.node.get("disabled") == true)
	for ei in range(ws.size()):
		if ws[ei].kind != "toggle" and shown[ei]:
			for v in _extra_values(ws[ei]):
				_add_extra_set(out, seen, {ei: v}, state)
	for ti in range(ws.size()):
		if ws[ti].kind != "toggle" or not shown[ti]:
			continue
		var on = not ws[ti].node.pressed
		_add_extra_set(out, seen, {ti: on}, state)
		_activate_extra(state)
		_extra_select(state)
		_set_widget(ws[ti], on, true)
		_extra_select(state)
		var revealed = []
		for ei in range(ws.size()):
			if ws[ei].kind != "toggle" and not shown[ei] and ws[ei].node.is_visible_in_tree():
				revealed.append(ei)
		var values = []
		for ei in revealed:
			values.append(_extra_values(ws[ei]))
		_activate_extra(state)
		for n in range(revealed.size()):
			for v in values[n]:
				var s = {}
				s[ti] = on
				s[revealed[n]] = v
				_add_extra_set(out, seen, s, state)
	_activate_extra(state)
	return out


func _add_extra_set(out, seen, s, state):
	_activate_extra()
	var err = _apply_set(s, state)
	var key = JSON.print(extra_node.get_extra())
	_activate_extra()
	if err == "" and not seen.has(key):
		seen[key] = true
		out.append(s)


# Values to try for one extra widget, in the formats _set_widget (probe) takes.
func _extra_values(w):
	var n = w.node
	match w.kind:
		"xy":
			var f = fighter.get_opponent_dir()
			var vals = []
			for a in EXTRA_XY_ANGLES:
				var rad = deg2rad(a)
				vals.append({"x": cos(rad) * f, "y": -sin(rad), "snap": false})
			return vals
		"8way":
			return _widget_values(w)
		"check", "height":
			return [not _save(w)]
		"slider", "count":
			return _widget_values(w)
		"option":
			return _widget_values(w)
	return []


# --- move data from the move's own UI scene ---------------------------------------------------

# A move's data variants as recipes on extra set k, in order: the default; then, for each
# combination of mode-widget values (block height, checkboxes, dropdowns: odometer order, first
# widget slowest, each widget's default first), the other widgets' values one widget at a time,
# interleaved across widgets (first value of each widget, then the second of each, ...).
# Variants whose data equals an earlier one's are dropped.
func _data_recipes(state, k):
	if not _has_data(state):
		return [_recipe(state, {}, k, null)]
	var node = _open_data_node(state)
	var widgets = []
	_find_widgets(node, widgets)
	var modes = []
	var others = []
	for wi in range(widgets.size()):
		if not widgets[wi].node.is_visible_in_tree():
			continue
		if widgets[wi].kind in MODE_KINDS:
			modes.append(wi)
		else:
			others.append(wi)
	var combos = [{}]
	for wi in modes:
		var grown = []
		for c in combos:
			grown.append(c)
			for v in _widget_values(widgets[wi]):
				var c2 = c.duplicate()
				c2[wi] = v
				grown.append(c2)
		combos = grown  # odometer order: the first mode widget changes slowest
	var singles = [{}]
	var lists = []
	for wi in others:
		lists.append(_widget_values(widgets[wi]))
	var i = 0
	var any = true
	while any:
		any = false
		for n in range(others.size()):
			if i < lists[n].size():
				any = true
				var s = {}
				s[others[n]] = lists[n][i]
				singles.append(s)
		i += 1
	var out = []
	var seen = {}
	for c in combos:
		for s in singles:
			var settings = c.duplicate()
			for wi in s:
				settings[wi] = s[wi]
			var data = _data_for(state, settings, false)
			var key = JSON.print(data)
			if seen.has(key):
				continue
			seen[key] = true
			out.append(_recipe(state, settings, k, data))
	return out


func _data_for(state, settings, reverse):
	var node = _open_data_node(state)
	var widgets = []
	_find_widgets(node, widgets)
	return _data_with(node, widgets, settings, reverse)


# The data scene's get_data() with these widget values (and flipped like the reverse button),
# leaving the widgets as they were.
func _data_with(node, widgets, settings, reverse):
	var saved = {}
	for wi in settings:
		saved[wi] = _save(widgets[wi])
		_apply(widgets[wi], settings[wi])
	if reverse:
		node.set_facing(-fighter.get_opponent_dir())
	var d = node.get_data()
	if typeof(d) == TYPE_DICTIONARY or typeof(d) == TYPE_ARRAY:
		d = d.duplicate(true)
	if reverse:
		node.set_facing(fighter.get_opponent_dir())
	var order = saved.keys()
	order.invert()
	for wi in order:
		_restore(widgets[wi], saved[wi])
	# A widget that doesn't restore exactly would leak into the next variant: start that move's
	# scene afresh instead.
	var key = node.state.name
	if (not settings.empty() or reverse) and JSON.print(node.get_data()) != data_defaults.get(key):
		data_nodes.erase(key)
		data_defaults.erase(key)
		_drop(node)
	return d


func _add_variant(out, seen, data):
	var key = JSON.print(data)
	if seen.has(key):
		return
	seen[key] = true
	if typeof(data) == TYPE_DICTIONARY or typeof(data) == TYPE_ARRAY:
		data = data.duplicate(true)
	out.append(data)


func _find_widgets(node, out):
	for c in node.get_children():
		var kind = _kind(c)
		if kind != "":
			out.append({"kind": kind, "node": c})
		else:
			_find_widgets(c, out)


func _kind(n):
	var s = n.get_script()
	var file = s.resource_path.get_file().get_basename() if s else ""
	if file == "XYPlot" or file == "DIWheel":
		return "xy"
	if file == "8Way":
		return "8way"
	if file == "HorizSlider":
		return "slider"
	if file == "CountOption":
		return "count"
	if file == "BlockHeightSelector":
		return "height"
	if n is CheckButton and n.has_method("get_data"):
		return "check"
	if n is OptionButton and n.has_method("get_data"):
		return "option"
	return ""


func _range_node(w):
	if w.kind == "slider":
		return w.node.get_node("Direction")
	return w.node.get_node("HSlider")


# Non-default values to try for one widget, in priority order.
func _widget_values(w):
	var n = w.node
	match w.kind:
		"xy":
			var r = n.panel_radius
			var f = fighter.get_opponent_dir()
			var vals = []
			for length in XY_LENGTHS:
				for a in XY_ANGLES:
					var rad = deg2rad(a)
					vals.append(Vector2(cos(rad) * f, -sin(rad)) * r * length)
			return vals
		"8way":
			var vals = []
			for d in WAY_DIRS:
				var b = n.get_node("%" + n.dir_to_facing(d)) if n.consider_facing else n.get_node("%" + d)
				if not b.disabled:
					vals.append(d)
			return vals
		"slider", "count":
			if w.kind == "count" and str(n.name) == "Melee Parry Timing":
				return PARRY_TIMINGS.duplicate()
			var rn = _range_node(w)
			var lo = rn.min_value
			var hi = rn.max_value
			if w.kind == "slider" and lo == 0 and hi == 100:
				# Dash-style distance: one value in each of the move's timing steps (IASA buckets
				# 0-9, 10-29, 30-49, 50-70, 71-90, 91-100), max and min first.
				return DASH_SLIDER_VALUES.duplicate()
			var mid = lo + (hi - lo) / 2.0
			if rn.step > 0:
				mid = lo + round((mid - lo) / rn.step) * rn.step
			return [hi, lo, mid]
		"check":
			return [not n.pressed]
		"option":
			var vals = []
			for i in range(n.get_item_count()):
				if i != n.selected and not n.is_item_disabled(i):
					vals.append(i)
			return vals
		"height":
			return [true, false]
	return []


func _save(w):
	var n = w.node
	match w.kind:
		"xy":
			return n.value_float
		"8way":
			return n.pressed_button.name if n.pressed_button else null
		"slider", "count":
			return _range_node(w).value
		"check":
			return n.pressed
		"option":
			return n.selected
		"height":
			return n.high.pressed
	return null


func _apply(w, v):
	var n = w.node
	match w.kind:
		"xy":
			n.update_value(v, false, true)
		"8way":
			n.set_dir(v)
		"slider", "count":
			_range_node(w).value = v
		"check":
			n.pressed = v
		"option":
			n.select(v)
		"height":
			n.set_height(v)


func _restore(w, saved):
	var n = w.node
	match w.kind:
		"xy":
			n.update_value(saved, false, true)
		"8way":
			if saved != null:
				n.set_dir(saved, true)
		_:
			_apply(w, saved)


# --- debug: every move and widget, and probes at chosen widget values ---------------------------
#
# `describe()` lists every move in the action menu with its legality now, and every widget its
# data scene, the PlayerExtra scene and the DI wheel show. `probe(spec)` sets those widgets the way
# a player would (the same widget methods mouse and keyboard input end up in) and returns the
# {action, data, extra} the menu would submit. Data still only ever comes from the widgets'
# get_data(); a probe only chooses the widget values. Widget and extra state is restored afterwards
# so probes never change what generate() returns.

const XYPlotScene = preload("res://ui/XYPlot/XYPlot.tscn")
var di_node = null       # a stand-in DI wheel (ActionButtons' DI is a default XYPlot)


func _ensure_di():
	if di_node == null:
		di_node = XYPlotScene.instance()
		add_child(di_node)
	return di_node


func describe():
	_activate_extra()
	var legal = {}
	for s in legal_states():
		legal[s.name] = true
	var moves = []
	for state in button_states:
		var m = {
			"name": state.name, "title": state.title if state.title else state.name,
			"type": _category(state), "grab": state.get("is_grab") == true,
			"air_type": int(state.air_type), "legal": legal.has(state.name), "widgets": [],
			"usable": state.is_usable(), "in_stance": state.allowed_in_stance(),
		}
		if legal.has(state.name):
			m["reversible"] = state.flip_allowed() and state.reversible
			_extra_select(state)
			m["feintable"] = state.can_feint() and (extra_node == null or extra_node.can_feint)
			if _has_data(state):
				var node = _open_data_node(state)
				var ws = []
				_find_widgets(node, ws)
				for w in ws:
					m.widgets.append(_describe_widget(w))
		moves.append(m)
	var extra_ws = []
	if extra_node:
		_activate_extra()
		_extra_select(null)
		for w in _extra_widgets():
			extra_ws.append(_describe_widget(w))
	var preds = [-1]
	if fighter.opponent.state_interruptable and not fighter.opponent.busy_interrupt:
		for state in button_states:
			if not int(state.type) in preds:
				preds.append(int(state.type))
	var out = {"moves": moves, "extra_widgets": extra_ws, "predictions": preds,
		"busy": fighter.busy_interrupt, "hurt": fighter.current_state().is_hurt_state,
		"grounded": fighter.is_grounded(), "facing": fighter.get_facing_int(),
		"opponent_dir": fighter.get_opponent_dir(), "di_enabled": fighter.di_enabled}
	_activate_extra()
	return out


# The menu's sequence when a move button is selected (update_buttons, then its data scene).
func _open_data_node(state):
	if not data_nodes.has(state.name):
		var fresh = state.data_ui_scene.instance()
		fresh.action_buttons = stub
		fresh.state = state
		fresh.fighter = fighter
		add_child(fresh)
		fresh.init()
		data_nodes[state.name] = fresh
	var node = data_nodes[state.name]
	node.show()
	node.set_facing(fighter.get_facing_int())
	node.fighter_update()
	node.set_facing(fighter.get_opponent_dir())
	node.on_button_selected()
	if not data_defaults.has(state.name):
		data_defaults[state.name] = JSON.print(node.get_data())
	return node


# ActionButtons.on_action_selected / extra_updated: the extra panel learns the selected move.
func _extra_select(state):
	if extra_node:
		extra_node.selected_move_will_restart = true
		extra_node.update_selected_move(state)


func _extra_widgets():
	var out = []
	_find_extra_widgets(extra_node, out)
	return out


func _find_extra_widgets(node, out):
	for c in node.get_children():
		var kind = _kind(c)
		if kind == "" and c is BaseButton and c.toggle_mode:
			kind = "toggle"
		if kind != "":
			out.append({"kind": kind, "node": c})
		else:
			_find_extra_widgets(c, out)


func _describe_widget(w):
	var n = w.node
	var d = {"kind": w.kind, "name": str(n.name), "visible": n.is_visible_in_tree()}
	match w.kind:
		"xy":
			d["radius"] = n.panel_radius
			d["panel_type"] = int(n.panel_type)
			d["bottom_half"] = n.bottom_half
			d["always_max"] = n.always_max
			d["min_length"] = n.min_length
			d["limit_angle"] = n.limit_angle
			d["limit_center_degrees"] = n.limit_center_degrees
			d["limit_range_degrees"] = n.limit_range_degrees
			d["limit_symmetrical"] = n.limit_symmetrical
			d["snap_angles"] = n.snap_angles if n.snap else 0
			d["facing"] = n.facing
			d["value"] = n.get_data()
		"8way":
			var dirs = []
			for dir in WAY_DIRS:
				var b = n.get_node("%" + n.dir_to_facing(dir)) if n.consider_facing else n.get_node("%" + dir)
				if not b.disabled:
					dirs.append(dir)
			d["enabled"] = dirs
			d["value"] = n.get_dir()
		"slider", "count":
			if w.kind == "count" and str(n.name) == "Melee Parry Timing":
				d["values"] = PARRY_TIMINGS.duplicate()
			var rn = _range_node(w)
			d["min"] = rn.min_value
			d["max"] = rn.max_value
			d["step"] = rn.step
			d["value"] = rn.value
		"check", "toggle":
			d["pressed"] = n.pressed
			d["disabled"] = n.disabled
		"option":
			var items = []
			for i in range(n.get_item_count()):
				items.append({"text": n.get_item_text(i), "disabled": n.is_item_disabled(i)})
			d["items"] = items
			d["selected"] = n.selected
		"height":
			d["value"] = n.high.pressed
	return d


# spec: {"action": name or "Continue", "widgets": {"<index>": value}, "extra": {"<index>": value},
#        "di": {"x": fx, "y": fy, "snap": bool} (fractions of the wheel radius, screen axes),
#        "feint": bool, "reverse": bool, "prediction": int}
# Widget values: xy {"x", "y", "snap"} (fractions of the radius, screen axes; snap = mouse input,
# no snap = keyboard nudges), 8way a direction name relative to facing (E = toward the opponent),
# slider/count a number, check/toggle/height a bool, option an item index.
# Returns {"candidate": {action, data, extra}} or {"error": message}.
func probe(spec):
	# Purity: a probe starts from fresh widgets, as generate() and expand() do, so earlier probes, previews and expands
	# leave nothing behind in the panel (Robot's FlyDir memory, a Mutant juke left pressed). The fresh panel shows the
	# JukeDir rows the turn-open panel left (the real panel hides rows a frame late), as the panel a probe read before.
	_new_epoch()
	_set_turn_sections()
	_activate_extra()
	var saved_extra = _save_extra()
	var res = _probe(spec)
	_restore_extra(saved_extra)
	_activate_extra()
	return res


func _probe(spec):
	var action = spec.get("action", "Continue")
	var state = null
	if action != "Continue":
		for s in button_states:
			if s.name == action:
				state = s
		if state == null:
			return {"error": "no move %s" % action}
	_extra_select(state)
	# Extra widgets first (as a player toggles them after picking the move); each click re-runs
	# the menu's extra_updated.
	var ewidgets = _extra_widgets() if extra_node else []
	var espec = spec.get("extra", {})
	for k in espec:
		var i = int(k)
		if i < 0 or i >= ewidgets.size():
			return {"error": "no extra widget %s" % k}
		var err = _set_widget(ewidgets[i], espec[k], true)
		if err != "":
			return {"error": "extra widget %s: %s" % [k, err]}
		_extra_select(state)
	if state != null:
		var ok = false
		for s in legal_states():
			if s == state:
				ok = true
		if not ok:
			return {"error": "illegal: %s is not available now" % action}
	var reverse = spec.get("reverse", false) == true
	var feint = spec.get("feint", false) == true
	if reverse and (state == null or not (state.flip_allowed() and state.reversible)):
		return {"error": "reverse is disabled for %s" % action}
	if feint and (state == null or not state.can_feint() or (extra_node != null and not extra_node.can_feint)):
		return {"error": "feint is disabled for %s" % action}
	var data = null
	if _has_data(state):
		var node = _open_data_node(state)
		var ws = []
		_find_widgets(node, ws)
		var saved = []
		for w in ws:
			saved.append(_save(w))
		var wspec = spec.get("widgets", {})
		var err = ""
		for k in wspec:
			var i = int(k)
			if i < 0 or i >= ws.size():
				err = "no widget %s" % k
				break
			err = _set_widget(ws[i], wspec[k], false)
			if err != "":
				err = "widget %s: %s" % [k, err]
				break
		if err == "":
			# The menu's reverse button flips the data scene's facing (ActionButtons._send_ui_action).
			if reverse:
				node.set_facing(-fighter.get_opponent_dir())
			data = node.get_data()
			if typeof(data) == TYPE_DICTIONARY or typeof(data) == TYPE_ARRAY:
				data = data.duplicate(true)
			if reverse:
				node.set_facing(fighter.get_opponent_dir())
		for i in range(ws.size()):
			_restore(ws[i], saved[i])
		if err != "":
			return {"error": err}
	var di = {"x": 0, "y": 0}
	if spec.has("di"):
		var dn = _ensure_di()
		dn.reset()
		var v = spec.di
		var p = Vector2(float(v.get("x", 0)), float(v.get("y", 0))).limit_length(1.0) * dn.panel_radius
		dn.update_value(p, false, v.get("snap", false) != true)
		di = dn.get_data()
		if (di.x != 0 or di.y != 0) and not fighter.di_enabled:
			return {"error": "DI is disabled"}  # as step_raw refuses it (YomiMatch._raw_input)
	var extra = {"DI": di, "reverse": reverse, "feint": feint, "prediction": int(spec.get("prediction", -1))}
	if extra_node:
		extra.merge(extra_node.get_extra())
	return {"candidate": {"action": action, "data": data, "extra": extra}}


# Sets a widget as the player would; returns "" or why the player couldn't.
func _set_widget(w, v, click):
	var n = w.node
	if not n.is_visible_in_tree():
		return "not visible"
	match w.kind:
		"xy":
			if typeof(v) != TYPE_DICTIONARY:
				return "xy value must be {x, y}"
			var p = Vector2(float(v.get("x", 0)), float(v.get("y", 0))).limit_length(1.0) * n.panel_radius
			n.update_value(p, false, v.get("snap", false) != true)
			if click:
				n.emit_signal("data_changed")
		"8way":
			if not v in WAY_DIRS:
				return "unknown direction %s" % str(v)
			var b = n.get_node("%" + n.dir_to_facing(v)) if n.consider_facing else n.get_node("%" + v)
			if b.disabled:
				return "direction %s is disabled" % v
			n.set_dir(v)
		"slider", "count":
			_range_node(w).value = float(v)
		"check", "height":
			if n.get("disabled") == true:
				return "disabled"
			_apply(w, v == true)
		"toggle":
			if n.disabled:
				return "disabled"
			if n.pressed != (v == true):
				n.pressed = v == true    # emits toggled, as a click does
				n.emit_signal("pressed")
		"option":
			var i = int(v)
			if i < 0 or i >= n.get_item_count() or n.is_item_disabled(i):
				return "no enabled item %s" % str(v)
			n.select(i)
			n.emit_signal("item_selected", i)
	return ""


func _save_extra():
	if extra_node == null:
		return null
	var ws = _extra_widgets()
	var vals = []
	for w in ws:
		match w.kind:
			"toggle":
				vals.append(w.node.pressed)
			_:
				vals.append(_save(w))
	var vars = {}
	for p in extra_node.get_script().get_script_property_list():
		var v = extra_node.get(p.name)
		if typeof(v) != TYPE_OBJECT:
			vars[p.name] = v.duplicate(true) if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY else v
	return {"ws": ws, "vals": vals, "vars": vars}


func _restore_extra(saved):
	if saved == null:
		return
	for i in range(saved.ws.size()):
		var w = saved.ws[i]
		var v = saved.vals[i]
		match w.kind:
			"toggle":
				w.node.set_pressed_no_signal(v)
			"8way":
				if v != null and (w.node.pressed_button == null or w.node.pressed_button.name != v):
					if w.node.pressed_button:
						w.node.pressed_button.set_pressed_no_signal(false)
					w.node._on_button_pressed_no_signal(w.node.get_node("%" + v))
			_:
				_restore(w, v)
	for k in saved.vars:
		extra_node.set(k, saved.vars[k])


# --- set 6: panel sets --------------------------------------------------------------------------------------
# Two (or three) of a move's own single changes made together, the toggles and toggle-plus-direction
# combinations a player makes that set 5 can't (Robot's directional flight, Mutant's juke with a direction, Ninja's
# release, ...), and each single change made as a player makes it. Rules: candidate_rules_v6.json (`set6`,
# `panel_pairs`). Every panel set is built on a fresh panel (the scene's, as a decision starts) and the working
# panel is put back after, so set 4's and set 5's extras never see its clicks. Never base entries.
const SET6_CHARS = ["Ninja", "Cowboy", "Wizard", "Mutant", "Robot"]


# This seat's set-6 options from the server's switches (pairs_<c>, clicks_<c>, toggle_only_<c>, triples).
func _set6_for(switches, character_):
	if typeof(switches) != TYPE_DICTIONARY or switches.empty() or not character_ in SET6_CHARS:
		return {}
	var c = character_.to_lower()
	var pairs = bool(switches.get("pairs_" + c, false))
	var clicks = bool(switches.get("clicks_" + c, false))
	if not pairs and not clicks:
		return {}
	return {"pairs": pairs, "clicks": clicks, "triples": pairs and bool(switches.get("triples", false)),
		"toggle_only": bool(switches.get("toggle_only_" + c, false)),
		# set 7 (YomiServer SET6_EXTRA_KEYS): these ride on the seat's panel sets, as in the simulator
		"juke_sections": bool(switches.get("juke_sections", false)), "pool": bool(switches.get("l1_pool", false)),
		"pool_dirs": bool(switches.get("l1_panel_dirs", false)), "switch": bool(switches.get("switch_sets", false))}


# Set 7's data-widget switches (any character of a set-6 seat).
func _set7_for(switches):
	var out = {}
	for k in ["xy_fine", "di_fine_all", "multi_xy_base", "multi_xy"]:
		out[k] = typeof(switches) == TYPE_DICTIONARY and bool(switches.get(k, false))
	return out


# A move's panel slots (cached per move): single clicks [k] first, then pairs [i, j] (i < j, on different
# widgets), then triples, each kept when every widget takes its value and its fresh-panel extras are new: not the
# reset panel's, a Draw pair's or an earlier panel set's. Continue has none.
func _panel_slots(state):
	if state == null or set6.empty():
		return []
	if panel_slots.has(state.name):
		return panel_slots[state.name]
	var sets = _sets_for(state)
	var n = sets.size()
	var own = []
	for k in range(n):
		own.append(_fresh_set4(sets[k], state))
	var seen = [_extras_key(own[0])]
	for slot in _pair_slots(state):
		var e = own[slot[0]].duplicate(true)
		for key in R5.pair_overlay_keys:
			if own[slot[1]].has(key):
				e[key] = own[slot[1]][key]
		seen.append(_extras_key(e))
	var kinds = _widget_kinds()
	var cands = []
	if set6.clicks:
		for k in range(1, n):
			cands.append([k])
	for i in range(1, n):
		for j in range(i + 1, n):
			if set6.pairs and _disjoint(sets[i], sets[j]) and (not set6.toggle_only or _has_toggle(sets[i], kinds)
					or _has_toggle(sets[j], kinds)):
				cands.append([i, j])
	if set6.triples:
		for i in range(1, n):
			for j in range(i + 1, n):
				for k in range(j + 1, n):
					if _disjoint(sets[i], sets[j]) and _disjoint(sets[i], sets[k]) and _disjoint(sets[j], sets[k]) \
							and (not set6.toggle_only or _has_toggle(sets[i], kinds) or _has_toggle(sets[j], kinds)
							or _has_toggle(sets[k], kinds)):
						cands.append([i, j, k])
	var out = []
	for slot in cands:
		var e = _fresh_apply(_combine(sets, slot, kinds), state)
		if e == null:
			continue
		var key = _extras_key(e)
		if not key in seen:
			seen.append(key)
			out.append(slot)
	panel_slots[state.name] = out
	return out


# A panel set's extras: its sets made together on a fresh panel (as set 4 applies them if a widget refuses).
func _panel_extras(state, slot):
	var combined = _combine(_sets_for(state), slot, _widget_kinds())
	var e = _fresh_apply(combined, state)
	return e if e != null else _fresh_set4(combined, state)


# `s` made on a fresh panel as a player makes it (the server's probe): the panel activated, the move selected,
# then each widget clicked with the move selected again after each. Its extras, or null if a widget refuses.
func _fresh_apply(s, state):
	var saved = _swap_in_fresh_panel(true)
	_activate_extra()
	_extra_select(state)
	var ok = true
	var ws = _extra_widgets()
	for ei in s:
		if _set_widget(ws[int(ei)], s[ei], true) != "":
			ok = false
			break
		_extra_select(state)
	var e = extra_node.get_extra().duplicate(true)
	_swap_back_panel(saved)
	return e if ok else null


# `s` applied as set 4 applies its own sets (_apply_set), on a fresh panel: its extras.
func _fresh_set4(s, state):
	var saved = _swap_in_fresh_panel()
	_activate_extra()
	_apply_set(s, state)
	var e = extra_node.get_extra().duplicate(true)
	_swap_back_panel(saved)
	return e


# probe_style: a panel for a set made as the probe makes it; with set 7's juke_sections it reads the turn-open JukeDir
# sections (_set_turn_sections).
func _swap_in_fresh_panel(probe_style = false):
	var saved = extra_node
	extra_node = fighter.player_extra_params_scene.instance()
	extra_node.set_fighter(fighter)
	add_child(extra_node)
	if probe_style and set6.get("juke_sections", false):
		_set_turn_sections()
	return saved


func _swap_back_panel(saved):
	_drop(extra_node)
	extra_node = saved


# The slot's sets as one set: every toggle click first (a player clicks a toggle, then picks its direction; the
# panel resets a direction clicked before its toggle), then the values, each in slot order.
func _combine(sets, slot, kinds):
	var toggles = {}
	var values = {}
	for k in slot:
		for ei in sets[k]:
			if kinds[int(ei)] == "toggle":
				toggles[ei] = sets[k][ei]
			else:
				values[ei] = sets[k][ei]
	for ei in values:
		toggles[ei] = values[ei]
	return toggles


func _widget_kinds():
	var out = []
	for w in _extra_widgets():
		out.append(w.kind)
	return out


func _disjoint(a, b):
	for ei in a:
		if b.has(ei):
			return false
	return true


func _has_toggle(s, kinds):
	for ei in s:
		if kinds[int(ei)] == "toggle":
			return true
	return false


func _extras_key(e):
	return JSON.print(e, "", true)


# --- set 7 (candidate rules v6 + SET6_EXTRA_KEYS; the simulator is the reference) ------------------------------------
# Switches outside set 6's 16, each its own rules hash (YomiServer.set6_hash): juke_sections (L0), l1_pool and
# l1_panel_dirs (L1's pool-only triples and toggle_pool), xy_fine and di_fine_all (L2: the `xy` and `di_fine_all`
# factors), multi_xy_base and multi_xy (L3: same-value base entries and the `xy_grid` factor), switch_sets (pool-only
# sets for panel values set 6 never builds). Set 7 as played is set 6 with the first six on; d1bd8211465b adds
# switch_sets. Nothing here runs with every one of them off.

# L2: xy angle step at depth 0 (degrees), length step (share of the radius); depth k divides each by 2^k. Where the
# neighbouring grid angle on a side is more than XY_GAP away, the first step to that side is the gap's bisector.
const XY_ANGLE_STEP = 15.0
const XY_LENGTH_STEP = 0.25
const XY_GAP = 45.0
const XY_DEPTH = 3
const DI_ALL_DEPTH = 3
const XY_BITS = 7        # bits per aimed widget in a packed xy value (a depth-3 base-5 path is at most 124)
const XY_SLOTS = 4       # at most this many aimed widgets per move
const XY_GRID_SLOTS = 8  # L3: the most XY slots an xy_grid value packs (4 bits each)


# Python float semantics (as the reference computes): floored %, math.radians / math.degrees (one multiplication
# by a constant computed in double).
func _fmod_floor(a, b):
	var m = fmod(a, b)
	if m == 0.0:
		return 0.0
	if (b < 0.0) != (m < 0.0):
		return m + b
	return m


func _norm7(a):
	a = _fmod_floor(a + 180.0, 360.0) - 180.0
	return 180.0 if a == -180.0 else a


func _radians(x):
	return x * (PI / 180.0)


func _degrees(x):
	return x * (180.0 / PI)


func _rel_angle_to(x, y, toward):
	return _norm7(_degrees(atan2(-float(y), float(x) * toward)))


# Dictionaries compare by reference in Godot 3; extras values (fly_dir {x, y}, ...) compare by content.
func _same(a, b):
	return typeof(a) == typeof(b) and JSON.print(a) == JSON.print(b)


# --- L1: pool-only triples and the toggle pool ---------------------------------------------------------------

# A move's pool-only triples (cached per move), numbered after its panel sets: three of its single changes, i<j<k,
# pairwise on different widgets, at least one a toggle click, made together as a panel set is, kept when the fresh
# panel's extras are new (not the reset panel's, a Draw pair's, a listed panel set's or an earlier triple's). Empty
# unless l1_pool is on, and when set 6 already lists triples.
func _pool_slots(state):
	if state == null or set6.empty() or not set6.pool or set6.triples:
		return []
	if pool_slots.has(state.name):
		return pool_slots[state.name]
	var sets = _sets_for(state)
	var n = sets.size()
	var kinds = _widget_kinds()
	var seen = [_extras_key(_fresh_set4(sets[0], state))]
	for slot in _pair_slots(state):
		seen.append(_extras_key(_overlay(_fresh_set4(sets[slot[0]], state), _fresh_set4(sets[slot[1]], state))))
	for slot in _panel_slots(state):
		var e = _fresh_apply(_combine(sets, slot, kinds), state)
		if e != null:
			seen.append(_extras_key(e))
	var out = []
	for i in range(1, n):
		for j in range(i + 1, n):
			if not _disjoint(sets[i], sets[j]):
				continue
			for k in range(j + 1, n):
				if not (_disjoint(sets[i], sets[k]) and _disjoint(sets[j], sets[k])):
					continue
				if not (_has_toggle(sets[i], kinds) or _has_toggle(sets[j], kinds) or _has_toggle(sets[k], kinds)):
					continue
				var slot = [i, j, k]
				var e = _fresh_apply(_combine(sets, slot, kinds), state)
				if e == null:
					continue
				var key = _extras_key(e)
				if not key in seen:
					seen.append(key)
					out.append(slot)
	pool_slots[state.name] = out
	return out


func _pool_extras(state, slot):
	return _panel_extras(state, slot)


# The Draw pair rule (_pair_extras) on two given extras.
func _overlay(a, b):
	var e = a.duplicate(true)
	for key in R5.pair_overlay_keys:
		if b.has(key):
			e[key] = b[key]
	return e


# A move's extra_set values that change a toggle (a boolean extra) from the reset panel's, in extra_set order: its own
# sets, Draw pairs, panel sets, pool triples and switch sets (Continue: its switch sets only, its own are base entries).
# With l1_panel_dirs, also its single non-boolean panel changes that take effect (fly_dir only with flight on).
func _toggle_sets(state):
	var sw = not set6.empty() and set6.switch
	if state == null and not sw:
		return []
	var n = _sets_for(state).size()
	var p5 = _pair_slots(state)
	var panel = _panel_slots(state)
	var pool = _pool_slots(state)
	var pooled = n + p5.size() + panel.size() + pool.size()
	var switches = _switch_slots(state)
	var total = pooled + switches.size()
	var pool_dirs = not set6.empty() and set6.pool_dirs
	var reset = _extras_k(state, 0)
	var out = []
	var from = max(pooled, 1) if state == null else 1
	for k in range(from, total):
		var e
		if k >= pooled:
			e = _switch_extras(state, switches[k - pooled])
		elif k < n:
			e = _extras_k(state, k)
		elif k - n < p5.size():
			e = _pair_extras(state, p5[k - n])
		else:
			var i = k - n - p5.size()
			e = _panel_extras(state, panel[i] if i < panel.size() else pool[i - panel.size()])
		if _flips(e, reset) or (pool_dirs and k < n and _changes_dir(e, reset)):
			out.append(k)
	return out


func _flips(e, reset):
	for k in e:
		if typeof(e[k]) == TYPE_BOOL:
			if not reset.has(k):
				if e[k]:
					return true
			elif not _same(reset[k], e[k]):
				return true
	return false


func _changes_dir(e, reset):
	var fly_on = e.has("fly_enabled") and e.fly_enabled == true
	for k in e:
		if typeof(e[k]) != TYPE_BOOL and (k != "fly_dir" or fly_on) and reset.has(k) and not _same(reset[k], e[k]):
			return true
	return false


# The seat's toggle pool at this decision: one entry per listed move other than Continue (its first listed candidate),
# in list order, with its toggle_sets values and whether it can feint / reverse; with switch_sets, Continue's entry
# (its switch sets) first where it has any. Every value it names is an extra_override `expand` accepts for any
# candidate of that move. Needs l1_pool on for this seat. {"moves": [...]} or {"error": why}.
func toggle_pool():
	if set6.empty() or not set6.pool:
		return {"error": "toggle_pool needs set 6 with l1_pool on for this seat's character"}
	_new_epoch()  # purity: like expand, from fresh widgets
	var seen = {}
	var out = []
	var cont_done = false
	var ids = recipes.keys()
	ids.sort()
	for id in ids:
		var r = recipes[id]
		var expanded = r.get("base") != null
		if r.state == null and not expanded and not cont_done and set6.switch:
			cont_done = true
			var cs = _toggle_sets(null)
			_activate_extra()
			if not cs.empty():
				out.append({"candidate": id, "action": "Continue", "extra_sets": cs, "feint": false, "reverse": false})
			continue
		if r.state == null or expanded or seen.has(r.state.name):
			continue
		seen[r.state.name] = true
		var ks = _toggle_sets(r.state)
		_activate_extra()
		out.append({"candidate": id, "action": r.state.name, "extra_sets": ks, "feint": _feintable(r.state),
			"reverse": r.state.flip_allowed() and r.state.reversible})
	return {"moves": out}


# --- switch sets ---------------------------------------------------------------------------------------------

# A move's switch sets (cached per move), explicit, numbered after its pool triples: 1. every extra XY plot the move
# shows, or that a toggle click reveals, at half length on the 8 extra_xy_angles (set 4 lists full length only), the
# revealing toggle clicked first; 2. Continue only: two of its own single changes on different widgets, made together
# (set 6's panel sets skip Continue). Kept when every widget takes its value and the fresh panel's extras are new (not
# the reset panel's, an own set's, a Draw pair's, a listed panel set's, a pool triple's or an earlier switch set's).
func _switch_slots(state):
	if set6.empty() or not set6.switch:
		return []
	var name = _name(state)
	if switch_slots.has(name):
		return switch_slots[name]
	var sets = _sets_for(state)
	var n = sets.size()
	var kinds = _widget_kinds()
	var seen = []
	for k in range(n):
		seen.append(_extras_key(_fresh_set4(sets[k], state)))
	for k in range(1, n):
		var e = _fresh_apply(sets[k], state)
		if e != null:
			seen.append(_extras_key(e))
	for slot in _pair_slots(state):
		seen.append(_extras_key(_overlay(_fresh_set4(sets[slot[0]], state), _fresh_set4(sets[slot[1]], state))))
	for slot in _panel_slots(state) + _pool_slots(state):
		var e = _fresh_apply(_combine(sets, slot, kinds), state)
		if e != null:
			seen.append(_extras_key(e))
	var cands = []
	# 1. half-length XY values, on a fresh panel with the move selected (as _fresh_apply makes them)
	var half = []
	for a in EXTRA_XY_ANGLES:
		var rad = deg2rad(a)
		half.append({"x": cos(rad) * fighter.get_opponent_dir() * 0.5, "y": -sin(rad) * 0.5, "snap": false})
	var saved = _swap_in_fresh_panel(true)
	_activate_extra()
	_extra_select(state)
	var ws = _extra_widgets()
	var shown = []
	for w in ws:
		shown.append(w.node.is_visible_in_tree() and not w.node.get("disabled") == true)
	for ei in range(ws.size()):
		if kinds[ei] == "xy" and shown[ei]:
			for v in half:
				var s = {}
				s[ei] = v
				cands.append(s)
	for ti in range(ws.size()):
		if kinds[ti] != "toggle" or not shown[ti]:
			continue
		# each toggle from the move-selected fresh panel (one toggle's click doesn't carry to the next)
		_drop(extra_node)
		extra_node = null
		_swap_in_fresh_panel(true)
		_activate_extra()
		_extra_select(state)
		ws = _extra_widgets()
		var on = not ws[ti].node.pressed
		if _set_widget(ws[ti], on, true) != "":
			continue
		_extra_select(state)
		for ei in range(ws.size()):
			if kinds[ei] == "xy" and not shown[ei] and ws[ei].node.is_visible_in_tree():
				for v in half:
					var s = {}
					s[ti] = on
					s[ei] = v
					cands.append(s)
	_swap_back_panel(saved)
	# 2. Continue's pairs
	if state == null:
		for i in range(1, n):
			for j in range(i + 1, n):
				if _disjoint(sets[i], sets[j]):
					cands.append(_combine(sets, [i, j], kinds))
	var out = []
	for s in cands:
		var e = _fresh_apply(s, state)
		if e == null:
			continue
		var key = _extras_key(e)
		if not key in seen:
			seen.append(key)
			out.append(s)
	switch_slots[name] = out
	return out


# Debug (live compare): the extra-panel widget settings behind extra_set k of candidate base_id's move, in the order
# they are applied, as probe takes them ([[widget index, value], ...]): a Draw pair's two sets in order, a panel set or
# pool triple combined (toggles first), a switch set as it is. {"widgets": [...]} or {"error": why}.
func extra_set_widgets(base_id, k):
	if not recipes.has(base_id):
		return {"error": "candidate %s was not generated or expanded this decision" % str(base_id)}
	var state = recipes[base_id].state
	var sets = _sets_for(state)
	var s
	if k < sets.size():
		s = sets[k]
	else:
		var layer = _layer_slot(state, k)
		if layer == null:
			return {"error": "extra_set %d out of range" % k}
		if layer[0] == "switch":
			s = layer[1]
		elif layer[0] == "pair":
			var out = []
			for u in layer[1]:
				for ei in sets[u]:
					out.append([int(ei), sets[u][ei]])
			return {"widgets": out}
		else:
			s = _combine(sets, layer[1], _widget_kinds())
	var out = []
	for ei in s:
		out.append([int(ei), s[ei]])
	return {"widgets": out}


# A switch set's extras: the fresh panel's with the set made (as _fresh_apply), or as set 4 applies it.
func _switch_extras(state, s):
	var e = _fresh_apply(s, state)
	return e if e != null else _fresh_set4(s, state)


# _validate_recipe with an explicit extra set in place of the recipe's own.
func _validate_explicit(r, s):
	_activate_extra()
	var err = _apply_set(s, r.state)
	if err == "" and r.state != null and not r.state in legal_states():
		err = "%s is not legal with extra set %d" % [_name(r.state), r.extra_set]
	if err == "" and r.feint and (r.state == null or not _feintable(r.state)):
		err = "feint is disabled for %s" % _name(r.state)
	if err == "" and r.reverse and (r.state == null or not (r.state.flip_allowed() and r.state.reversible)):
		err = "reverse is disabled for %s" % _name(r.state)
	_activate_extra()
	return err


# --- juke_sections (L0) ----------------------------------------------------------------------------------------
# The real panel hides an 8Way's empty rows deferred (try_hide_sections), so a panel made within one frame shows the
# rows the previous frame left: at a turn's open, the rows the panel shows with Continue selected. A Mutant JukeDir
# direction in a hidden row reads as neutral. With juke_sections on, probe-style panels read those rows.

func _juke_dir(node):
	if node == null:
		return null
	var out = []
	_find_extra_widgets(node, out)
	for w in out:
		if w.kind == "8way" and str(w.node.name) == "JukeDir":
			return w.node
	return null


const WAY_ROWS = [["NW", "N", "NE"], ["W", "Neutral", "E"], ["SW", "S", "SE"]]


# The turn-open panel's JukeDir rows (cached per decision): a row shows when one of its buttons is enabled.
func _turn_sections():
	if turn_sections == null:
		turn_sections = []
		var saved = _swap_in_fresh_panel()
		_activate_extra()
		var jd = _juke_dir(extra_node)
		if jd != null:
			for row in WAY_ROWS:
				var any = false
				for d in row:
					any = any or not jd.get_node("%" + d).disabled
				turn_sections.append(any)
		_swap_back_panel(saved)
	return turn_sections


func _set_turn_sections():
	var jd = _juke_dir(extra_node)
	if jd == null:
		return
	var rows = _turn_sections()
	for r in range(rows.size()):
		jd.get_node("%" + WAY_ROWS[r][1]).get_parent().visible = rows[r]


# --- L2: fine aim (xy) and fine DI for every seat (di_fine_all) ---------------------------------------------------

# The move's data node as this epoch has it (opened if it isn't yet; an open node is not re-opened).
func _node_now(state):
	return data_nodes[state.name] if data_nodes.has(state.name) else _open_data_node(state)


func _widgets_of(node):
	var ws = []
	_find_widgets(node, ws)
	return ws


# A widget's value as the reference reads it: an XY plot's point, a slider's or count's number.
func _wvalue(w):
	if w.kind == "xy":
		return w.node.value_float
	return _range_node(w).value


# The data a widget sends (an aim step changes it or is no step).
func _wdata(w):
	if w.kind == "xy":
		return JSON.print(w.node.get_data())
	return str(int(_range_node(w).value))


# A move's aimed widgets (cached per move): each visible non-mode data widget that is an XY plot, or a slider or count
# that can take more values than the base list gives it, with those values as the widget takes them (`grid`: an XY
# plot's angles from toward the opponent, counterclockwise; a slider's values, its default included). At most XY_SLOTS.
func _aimed(state):
	if aimed_cache.has(state.name):
		return aimed_cache[state.name]
	var out = []
	if _has_data(state):
		var node = _node_now(state)
		var ws = _widgets_of(node)
		var opp = fighter.get_opponent_dir()
		for wi in range(ws.size()):
			var w = ws[wi]
			if not w.node.is_visible_in_tree() or w.kind in MODE_KINDS or out.size() >= XY_SLOTS:
				continue
			if w.kind == "xy":
				var grid = []
				var saved = _snap(w)
				for v in _widget_values(w):
					_apply(w, v)
					var p = w.node.value_float
					_unsnap(w, saved)
					if _hypot(p.x, p.y) > 1e-9:
						var a = _rel_angle_to(p.x, p.y, opp)
						var dup = false
						for g in grid:
							dup = dup or abs(_norm7(g - a)) < 0.5
						if not dup:
							grid.append(a)
				out.append({"child": wi, "grid": grid, "wrap": not w.node.limit_angle})
			elif w.kind in ["slider", "count"]:
				var rn = _range_node(w)
				var grid = []
				var saved = rn.value
				for v in _widget_values(w):
					rn.value = v
					if not rn.value in grid:
						grid.append(rn.value)
					rn.value = saved
				if not rn.value in grid:
					grid.append(rn.value)
				var span = rn.max_value - rn.min_value
				var n = floor(span / rn.step) + 1.0 if rn.step > 0 else floor(span) + 1.0
				if grid.size() < n:
					out.append({"child": wi, "grid": grid, "wrap": false})
	aimed_cache[state.name] = out
	return out


func _xy_slot_path(xy, j):
	return (xy >> (XY_BITS * j)) & ((1 << XY_BITS) - 1)


# A packed xy value is valid: every slot below `slots` a base-5 path of at most XY_DEPTH digits, none 0, nothing above.
func _xy_valid(xy, slots):
	if slots < XY_SLOTS and xy >> (XY_BITS * slots) != 0:
		return false
	for j in range(min(slots, XY_SLOTS)):
		var d = _path_digits(_xy_slot_path(xy, j), 5)
		if d.size() > XY_DEPTH or 0 in d:
			return false
	return true


# From angle `a` (on the grid, within 0.5 degrees), the distance to the next grid angle to side `sign` (+1
# counterclockwise), if any; around the circle when `wrap`, else measured from a cut in the middle of the grid's
# widest gap (outside the plot's angle limit).
func _grid_gap(a, side, grid, wrap):
	var on = false
	for g in grid:
		on = on or abs(_norm7(g - a)) < 0.5
	if not on:
		return null
	var cut = 0.0
	if not wrap:
		var s = []
		for g in grid:
			s.append(_fmod_floor(g, 360.0))
		s.sort()
		var best = -1.0
		var at = 0.0
		for k in range(s.size()):
			var nxt = s[k + 1] if k + 1 < s.size() else s[0] + 360.0
			if nxt - s[k] > best:
				best = nxt - s[k]
				at = s[k] + best / 2.0
		cut = at
	var m = null
	for g in grid:
		var d
		if wrap:
			d = _fmod_floor(side * (g - a), 360.0)
		else:
			d = side * (_fmod_floor(g - cut, 360.0) - _fmod_floor(a - cut, 360.0))
		if d > 0.5 and d < 360.0:
			m = d if m == null else min(m, d)
	return m


# One fine-aim step on an aimed widget at its current value, through the widget's own methods: true when the data it
# sends changed. XY: digits 1/2 turn by -/+ XY_ANGLE_STEP / 2^level at the same length (gap fill at level 0), 3/4
# change the length by -/+ XY_LENGTH_STEP / 2^level of the radius (not on an always_max plot; within (0, radius]).
# Slider or count: 1/2 move half way to the next grid value below / above (or the widget's limit).
func _aim_step7(w, a, digit, level):
	var before = _wdata(w)
	if w.kind == "xy":
		var n = w.node
		var p = n.value_float
		var px = float(p.x)
		var py = float(p.y)
		var length = _hypot(px, py)
		if length < 1e-9:
			return false
		var sc = float(1 << level)
		var opp = fighter.get_opponent_dir()
		var np
		if digit == 1 or digit == 2:
			var side = -1.0 if digit == 1 else 1.0
			var ang = _rel_angle_to(px, py, opp)
			var step = XY_ANGLE_STEP / sc
			if level == 0:
				var gap = _grid_gap(ang, side, a.grid, a.wrap)
				if gap != null and gap > XY_GAP:
					step = gap / 2.0
			var rad = _radians(ang + side * step)
			np = Vector2(cos(rad) * length * opp, -sin(rad) * length)
		elif digit == 3 or digit == 4:
			if n.always_max:
				return false
			var dl = XY_LENGTH_STEP * n.panel_radius / sc
			var l2 = length - dl if digit == 3 else length + dl
			if not (l2 > 1e-6 and l2 <= n.panel_radius + 1e-6):
				return false
			np = Vector2(px / length * l2, py / length * l2)
		else:
			return false
		n.update_value(np, false, true)
	elif w.kind in ["slider", "count"]:
		var rn = _range_node(w)
		var v = rn.value
		var bound = null
		if digit == 1:
			for g in a.grid:
				if g < v - 1e-9 and (bound == null or g > bound):
					bound = g
			if bound == null and rn.min_value < v:
				bound = rn.min_value
		elif digit == 2:
			for g in a.grid:
				if g > v + 1e-9 and (bound == null or g < bound):
					bound = g
			if bound == null and rn.max_value > v:
				bound = rn.max_value
		if bound == null:
			return false
		rn.value = v + (bound - v) / 2.0
	else:
		return false
	return _wdata(w) != before


# Aimed widget `a` of move `state` as recipe `r`'s settings leave it, its xy path `path` applied, then `extra` (a digit
# at depth `level`, or 0). [true, its value] or [false]; the widget is put back.
func _aimed_run(state, a, r, path, extra = 0, level = 0):
	var node = _node_now(state)
	var w = _widgets_of(node)[a.child]
	var saved = _snap(w)
	if r.widgets.has(a.child):
		_apply(w, r.widgets[a.child])
	var ok = true
	var digits = _path_digits(path, 5)
	for l in range(digits.size()):
		if not _aim_step7(w, a, digits[l], l):
			ok = false
			break
	if ok and extra != 0:
		ok = _aim_step7(w, a, extra, level)
	var v = _wvalue(w)
	_unsnap(w, saved)
	return [true, v] if ok else [false]


# A widget's exact state, to put it back after a trial (the reference steps a copy of the widget): an XY plot's script
# variables, a slider's or count's value.
func _snap(w):
	if w.kind != "xy":
		return _range_node(w).value
	var vars = {}
	for p in w.node.get_script().get_script_property_list():
		var v = w.node.get(p.name)
		if typeof(v) != TYPE_OBJECT:
			vars[p.name] = v.duplicate(true) if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY else v
	return vars


func _unsnap(w, saved):
	if w.kind != "xy":
		_range_node(w).value = saved
		return
	for k in saved:
		w.node.set(k, saved[k])


# Recipe r's widget settings with its xy paths' final widget values set last, or why not.
func _xy_settings(r):
	var name = _name(r.state)
	if r.state == null or not _has_data(r.state):
		return "xy is not a factor of %s" % name
	var aimed = _aimed(r.state)
	if not _xy_valid(r.xy, aimed.size()):
		return "xy %d is not a path of at most %d steps on each of %s's %d aimed widgets" % [r.xy, XY_DEPTH, name, aimed.size()]
	var settings = r.widgets.duplicate()
	for j in range(aimed.size()):
		var p = _xy_slot_path(r.xy, j)
		if p == 0:
			continue
		var res = _aimed_run(r.state, aimed[j], r, p)
		if not res[0]:
			return "xy %d has no step on %s's widget %d" % [r.xy, name, j]
		settings.erase(aimed[j].child)
		settings[aimed[j].child] = res[1]
	return settings


# Set 7's per-candidate factors and values: `xy` on a candidate of a move with aimed widgets (the paths one step longer
# that exist), and `di_fine_all` on every candidate of a seat with DI enabled (not where set 5's di_fine steers it;
# children only from a non-neutral DI).
func _set7_meta(c, r):
	if not (set7.get("xy_fine", false) or set7.get("di_fine_all", false)):
		return
	if r.state != null and set7.xy_fine and _has_data(r.state):
		var aimed = _aimed(r.state)
		if not aimed.empty():
			var kids = []
			for j in range(aimed.size()):
				var p = _xy_slot_path(r.xy, j)
				var depth = _path_digits(p, 5).size()
				if depth >= XY_DEPTH:
					continue
				if not _aimed_run(r.state, aimed[j], r, p)[0]:
					continue
				for digit in [1, 2, 3, 4]:
					if _aimed_run(r.state, aimed[j], r, p, digit, depth)[0]:
						kids.append(r.xy + ((digit * int(pow(5, depth))) << (XY_BITS * j)))
			c.factors["xy"] = kids
			c.values["xy"] = r.xy
	var bullet = _s5("bullet_di") and d5.owns_bullet
	if set7.di_fine_all and not bullet and fighter.di_enabled:
		var di = c.extra.get("DI", {})
		var x = int(di.get("x", 0))
		var y = int(di.get("y", 0))
		var depth = _path_digits(r.di_fine_all, 5).size()
		var kids = []
		if (x != 0 or y != 0) and depth < DI_ALL_DEPTH:
			for digit in [1, 2, 3, 4]:
				if _di_step(x, y, digit, depth) != null:
					kids.append(r.di_fine_all + digit * int(pow(5, depth)))
		c.factors["di_fine_all"] = kids
		c.values["di_fine_all"] = r.di_fine_all


# --- L3: moves with two or more XY widgets ---------------------------------------------------------------------

# A move's XY slots ([widget index, its grid values]) when it shows two or more XY data widgets; else [].
func _multi_xy(state):
	if not _has_data(state):
		return []
	var node = _open_data_node(state)
	var ws = _widgets_of(node)
	var out = []
	for wi in range(ws.size()):
		if ws[wi].kind == "xy" and ws[wi].node.is_visible_in_tree():
			out.append([wi, _widget_values(ws[wi])])
	return out if out.size() >= 2 else []


# multi_xy_base: a multi-XY move's same-value entries after its variants: for each grid value g of its first XY widget
# (grid order), every XY widget set to g (skipped when a widget's grid lacks g); data repeats dropped.
func _same_value_entries(state, recs):
	var slots = _multi_xy(state)
	if slots.empty() or recs.empty():
		return
	var k = recs[0].extra_set
	var seen = {}
	for r in recs:
		if r.data != null:
			seen[JSON.print(r.data)] = true
	for g in slots[0][1]:
		var all = true
		for s in slots:
			all = all and g in s[1]
		if not all:
			continue
		var settings = {}
		for s in slots:
			settings[s[0]] = g
		var data = _data_for(state, settings, false)
		var key = JSON.print(data)
		if seen.has(key):
			continue
		seen[key] = true
		recs.append(_recipe(state, settings, k, data))


func _xy_grid_slot(v, j):
	return (v >> (4 * j)) & 15


# Whether XY widget `wi` still reads its value once `settings` are applied in order (the widgets are put back).
func _xy_at_default(node, settings, wi):
	var ws = _widgets_of(node)
	if ws[wi].kind != "xy":
		return false
	var before = ws[wi].node.value_float
	var saved = []
	for j in settings:
		saved.append([j, _save(ws[j])])
		_apply(ws[j], settings[j])
	var same = ws[wi].node.value_float == before
	saved.invert()
	for s in saved:
		_restore(ws[s[0]], s[1])
	return same


# The slots of a multi-XY move a recipe leaves at their defaults (no xy_grid index, no xy path on that widget, and the
# widget reads its value after the recipe's settings).
func _xy_default_slots(r, slots):
	var refined = []
	if r.xy != 0 and (set7.xy_fine or set7.di_fine_all):
		var aimed = _aimed(r.state)
		for j in range(aimed.size()):
			if _xy_slot_path(r.xy, j) != 0:
				refined.append(aimed[j].child)
	var node = _open_data_node(r.state)
	var out = []
	for j in range(slots.size()):
		var wi = slots[j][0]
		out.append(_xy_grid_slot(r.xy_grid, j) == 0 and not wi in refined and _xy_at_default(node, r.widgets, wi))
	return out


# multi_xy: r with xy_grid value v (absolute). Each slot that changes must be one the recipe leaves at its default; its
# widget is set to that grid value. "" or why not.
func _apply_xy_grid(r, v):
	var name = _name(r.state)
	var slots = [] if r.state == null else _multi_xy(r.state)
	if slots.empty():
		return "xy_grid is not a factor of %s (it needs two or more XY widgets)" % name
	var n = min(slots.size(), XY_GRID_SLOTS)
	if n < XY_GRID_SLOTS and v >> (4 * n) != 0:
		return "xy_grid %d names a slot %s doesn't have (%d XY widgets)" % [v, name, n]
	var at_default = _xy_default_slots(r, slots)
	r.widgets = r.widgets.duplicate()  # the recipe is a shallow copy of the base's
	for j in range(n):
		var old = _xy_grid_slot(r.xy_grid, j)
		var nv = _xy_grid_slot(v, j)
		if old == nv:
			continue
		if not at_default[j]:
			return "xy_grid: %s's XY widget %d is not at its default" % [name, j]
		if nv < 1 or nv > slots[j][1].size():
			return "xy_grid: %s's XY widget %d has %d grid values, not %d" % [name, j, slots[j][1].size(), nv]
		r.widgets[slots[j][0]] = slots[j][1][nv - 1]
	r.xy_grid = v
	r.data = null
	return ""


# multi_xy's per-candidate factor and value: on a candidate of a multi-XY move, `xy_grid` lists, for each slot still at
# its default, every grid index (at most 15), and the candidate's own value.
func _l3_meta(c, r):
	if not set7.get("multi_xy", false) or r.state == null:
		return
	var slots = _multi_xy(r.state)
	if slots.empty():
		return
	var at_default = _xy_default_slots(r, slots)
	var kids = []
	for j in range(min(slots.size(), XY_GRID_SLOTS)):
		if at_default[j]:
			for gi in range(1, min(slots[j][1].size(), 15) + 1):
				kids.append(r.xy_grid | (gi << (4 * j)))
	c.factors["xy_grid"] = kids
	c.values["xy_grid"] = r.xy_grid
