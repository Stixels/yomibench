"""step_raw's raw_option_by_id flag (YomiMatch.step_raw): off by default, sent only when asked for, and read by the mod."""
import re
from pathlib import Path

import pytest

from yomibench.engine import Engine, EngineError, InProcess

MOD = Path(__file__).resolve().parents[1] / "mod" / "YomiBench"
INPUTS = [{"action": "QuickSlash", "data": {"Attack": {"id": 2, "name": False}}, "extra": {}},
          {"action": "Continue", "data": None, "extra": {}}]


class Recorder:
    def __init__(self):
        self.msgs = []

    def handle(self, msg):
        self.msgs.append(msg)
        return {"error": {"code": "bad_request", "message": "recorded"}}


def _sent(**kw):
    r = Recorder()
    with pytest.raises(EngineError):
        Engine(InProcess(r)).step_raw(INPUTS, **kw)
    return r.msgs[0]


def test_off_by_default_sends_todays_request():
    assert _sent() == {"cmd": "step_raw", "inputs": INPUTS}
    assert _sent(raw_option_by_id=False) == {"cmd": "step_raw", "inputs": INPUTS}


def test_on_adds_the_field():
    assert _sent(raw_option_by_id=True) == {"cmd": "step_raw", "inputs": INPUTS, "raw_option_by_id": True}


def test_the_mod_reads_it_and_defaults_off():
    server = (MOD / "YomiServer.gd").read_text(encoding="utf-8")
    assert 'match_.step_raw(msg.get("inputs"), msg.get("raw_option_by_id") == true)' in server
    match = (MOD / "YomiMatch.gd").read_text(encoding="utf-8")
    assert re.search(r"^func step_raw\(inputs, option_by_id = false\):", match, re.M)
    assert re.search(r"^func _raw_input\(seat, inp, option_by_id = false\):", match, re.M)
    # Off: the template is the data scene's get_data() at its defaults, as before the flag.
    assert re.search(r"if option_by_id:\n\t+like = _raw_option_like\(node, data, options, errs\)\n\t+else:\n\t+like = node.get_data\(\)",
                     match)
