extends "res://ui/UILayer.gd"
# The main menu gets "Multiplayer (vs bot)" as its first multiplayer entry, right under Singleplayer (above the Ranked
# mod's "Multiplayer (Ranked)"), a Button in the menu's container. Pressing it starts a local multiplayer session against a YomiBench bot (VsBot.gd):
# the bot server and a headless copy of this game on this PC, joined over 127.0.0.1 only. It is refused while
# an online session or a Steam lobby is open. "Back to main menu" stops both and reloads the menu.

const VsBot = preload("res://YomiBenchVsBot/VsBot.gd")
const AFTER = ["MultiplayerButton", "DirectConnectButton", "SteamMultiplayerButton"]


var vsbot_button = null


func _ready():
	var anchor = null
	for n in AFTER:  # the last of the game's multiplayer entries in the menu
		var b = get_node_or_null("%" + n)
		if b != null and (anchor == null or (b.get_parent() == anchor.get_parent() and b.get_index() > anchor.get_index())):
			anchor = b
	if anchor == null:
		return
	var b = Button.new()
	b.name = "VsBotButton"
	b.text = "Multiplayer (vs bot)"
	b.flat = anchor.flat
	b.mouse_default_cursor_shape = anchor.mouse_default_cursor_shape
	b.rect_min_size = anchor.rect_min_size
	anchor.get_parent().add_child_below_node(anchor, b)
	b.connect("pressed", self, "_on_vsbot_pressed")  # the game wires menu sounds to every button itself
	# First of the multiplayer entries: right under Singleplayer, so above "Multiplayer (Ranked)" when the Ranked mod
	# is on (it puts its button under Singleplayer a few frames later, so ours moves back up whenever the menu's
	# buttons change) and above "Multiplayer (Steam)" without it.
	vsbot_button = b
	_place_vsbot_button()
	var box = b.get_parent()
	if box is Container:
		box.connect("sort_children", self, "_place_vsbot_button", [], CONNECT_DEFERRED)
	if VsBot.autoplay_schedule() != "" and VsBot.instance(get_tree()) == null:
		call_deferred("_autoplay")  # headless test hook (--yomibench-vsbot-autoplay=...)


func _place_vsbot_button():
	if not is_instance_valid(vsbot_button):
		return
	var single = get_node_or_null("%SingleplayerButton")
	var box = vsbot_button.get_parent()
	if single == null or single.get_parent() != box:
		return  # no Singleplayer entry beside ours: it stays after the multiplayer entries
	var want = single.get_index() + 1
	if vsbot_button.get_index() < single.get_index():
		want -= 1  # moving ours down from above Singleplayer: its index shifts up by one
	if vsbot_button.get_index() != want:
		box.move_child(vsbot_button, want)


func _autoplay():
	while not Global.mods_loaded:
		yield(get_tree(), "idle_frame")
	_on_vsbot_pressed()


func _on_vsbot_pressed():
	# Not while the game still loads its mods in the background (the character select's thread).
	if VsBot.instance(get_tree()) != null or not Global.mods_loaded:
		return
	Global.frame_advance = false
	var v = VsBot.new()
	get_tree().root.add_child(v)  # outlives the menu scene's reloads
	if v.launch():  # refuses (and says why) while an online session or a Steam lobby is open
		hide_main_menu()
