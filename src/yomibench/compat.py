"""Which self-play records can train together (torch-free: the training cache uses it too)."""

from __future__ import annotations

import json
from typing import Any

from yomibench.valuenet import FEATURES


def record_keys(header: dict[str, Any]) -> tuple[str, str]:
    """(game key, candidate set) of a record. Records train together only if their game keys agree:
    the game version, the match settings, and the engine's state version (the protocol `version`,
    which fixes what a state holds), and the current encoder FEATURES version. Raw records
    are encoded by that one feature version at load time. The engine itself is not part
    of it, so real-game and simulator records of the same game mix, as a
    generation window across the switch needs. Candidate sets are kept apart separately (see
    check_compatible): they change what the search could pick, not what a position is worth."""
    eng = header.get("engine", {})
    game = json.dumps({"game_version": eng.get("game_version"), "state_version": eng.get("version"),
                       "features_version": FEATURES,
                       "match_settings": eng.get("match_settings")}, sort_keys=True)
    return game, str(eng.get("candidate_set"))


def check_keys(keys: list[tuple[str, str]], allow_mixed_candidate_sets: bool = False,
               allow_match_settings: list[dict[str, Any]] | None = None) -> None:
    """Refuse records of different games, and of different candidate sets unless
    `allow_mixed_candidate_sets`, given each record's record_keys. `allow_match_settings`
    (match-settings variants, see match_variant): records of different match settings may still
    train together if every one matches a variant and their game keys agree on everything else."""
    games: dict[str, int] = {}
    sets: dict[str, int] = {}
    for game, cset in keys:
        games[game] = games.get(game, 0) + 1
        sets[cset] = sets.get(cset, 0) + 1
    if len(games) > 1 and allow_match_settings:
        _check_variants(games, allow_match_settings)
    elif len(games) > 1:
        raise SystemExit("records from different game versions, state versions or match settings; "
                         "train on one at a time:\n" + "\n".join(f"  {n} x {k[:200]}" for k, n in games.items()))
    if len(sets) > 1 and not allow_mixed_candidate_sets:
        raise SystemExit("records from different candidate sets ("
                         + ", ".join(f"{n} x set {k}" for k, n in sorted(sets.items()))
                         + "); pass --allow-mixed-candidate-sets to train on them together")


def _check_variants(games: dict[str, int], variants: list[dict[str, Any]]) -> None:
    """check_keys with an allow-list: each game key matches a variant, and the keys agree once the
    settings the variants name are set aside (raw comparison, as without an allow-list)."""
    varying = _varying(variants)
    outside = {k: n for k, n in games.items() if match_variant((k, ""), variants) is None}
    if outside:
        raise SystemExit("records whose match settings are not in --allow-match-settings ("
                         + "; ".join(variant_label(v) for v in variants) + "); train on one at a time:\n"
                         + "\n".join(f"  {n} x {k[:200]}" for k, n in outside.items()))
    rest: dict[str, int] = {}
    for k, n in games.items():
        g = json.loads(k)
        g["match_settings"] = {s: v for s, v in (g.get("match_settings") or {}).items() if s not in varying}
        r = json.dumps(g, sort_keys=True)
        rest[r] = rest.get(r, 0) + n
    if len(rest) > 1:
        raise SystemExit("records from different game versions, state versions or match settings (beyond the "
                         "--allow-match-settings variants, which only vary " + ", ".join(sorted(varying))
                         + "); train on one at a time:\n" + "\n".join(f"  {n} x {k[:200]}" for k, n in games.items()))


# --- match-settings allow-list (a window mixing approved settings, e.g. ranked at ceilings 500 and 800) ---
# train --allow-match-settings '[{"ceiling_height": 800}, {"ceiling_height": 500, "has_ceiling": true}]':
# each variant names the settings it fixes. A setting any variant names "varies"; in a variant that
# doesn't name it, it takes its standard value (SIM_SETTINGS_STANDARD). A record matches a variant
# when its varying settings equal the variant's by value; every other setting (and the game,
# state and feature versions) must still be identical across the window. Partial variants are thus
# relative to the window's own settings (the ranked preset in production); full dicts pin everything.


def parse_allow_match_settings(value: Any) -> list[dict[str, Any]]:
    """The --allow-match-settings value checked: a non-empty list of non-empty JSON objects."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) and v for v in value):
        raise ValueError("--allow-match-settings must be a non-empty JSON list of non-empty objects, e.g. "
                         '[{"ceiling_height": 800}, {"ceiling_height": 500, "has_ceiling": true}]')
    return [dict(v) for v in value]


def _varying(variants: list[dict[str, Any]]) -> list[str]:
    return sorted({k for v in variants for k in v})


def variant_label(variant: dict[str, Any]) -> str:
    """A variant's name in reports: the settings it names, e.g. "ceiling_height=500,has_ceiling=true"."""
    return ",".join(f"{k}={json.dumps(v)}" for k, v in sorted(variant.items()))


def variant_settings(base: dict[str, Any], variants: list[dict[str, Any]], i: int) -> dict[str, Any]:
    """`base` settings with variant `i`'s varying settings (the record filter's view of a variant)."""
    return {**base, **{k: variants[i].get(k, SIM_SETTINGS_STANDARD.get(k)) for k in _varying(variants)}}


def match_variant(keys: tuple[str, str], variants: list[dict[str, Any]]) -> int | None:
    """The index of the first variant whose varying settings a record (its record_keys) has, or None."""
    have = json.loads(keys[0]).get("match_settings") or {}
    varying = _varying(variants)
    got = [_canon(have.get(k, SIM_SETTINGS_STANDARD.get(k))) for k in varying]
    for i, v in enumerate(variants):
        if got == [_canon(v.get(k, SIM_SETTINGS_STANDARD.get(k))) for k in varying]:
            return i
    return None


# --- record filters (a training window pinned to one game and candidate set) -------------------
# After the fleet switches settings or candidate rules (the ranked preset, set 5), the generation
# folders hold both the old records and the new ones. The trainer then passes a filter (train
# --record-filter) and only the records matching it are trained on, validated or evaluated; the
# rest are excluded, counted and logged, never mixed.

# The match settings a simulator applies, with their standard values: the
# only ones a record can play off-standard. A record whose header lacks one (or all: records from
# before hello reported settings) played its standard value.
SIM_SETTINGS_STANDARD = {"stage_width": 800, "char_distance": 125, "char_height": 0, "game_length": 3600,
                         "clashing_enabled": True, "starting_meter": "0", "global_gravity_modifier": "1",
                         "has_ceiling": True, "ceiling_height": 800}


def _canon(v: Any) -> Any:
    """A setting's value for comparison: numbers and decimal strings by value (1.5 == "1.5" == "1.50")."""
    from decimal import Decimal, InvalidOperation

    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, str)):
        try:
            return Decimal(str(v)).normalize()
        except InvalidOperation:
            return v
    return v


def filter_reason(keys: tuple[str, str], record_filter: dict[str, Any], any_candidate_set: bool = False,
                  allow_match_settings: list[dict[str, Any]] | None = None) -> str | None:
    """Why a record with these record_keys is outside `record_filter` ({"candidate_set": "5",
    "match_settings": {overrides of the standard settings}}), or None if it is inside: each sim
    setting must have the filter's value (its standard value where the filter names none; settings
    outside the sim's list are left to check_keys), or, with `allow_match_settings`, the filter's
    settings with one variant's varying settings (variant_settings: the reason given is then against
    the variant the record's varying settings match, if any); and the candidate set must be the
    filter's unless `any_candidate_set`."""
    game, cset = keys
    have = json.loads(game).get("match_settings") or {}
    want = record_filter.get("match_settings") or {}
    why = _settings_reason(have, want)
    if why is not None and allow_match_settings:
        reasons = [_settings_reason(have, variant_settings(want, allow_match_settings, i))
                   for i in range(len(allow_match_settings))]
        i = match_variant(keys, allow_match_settings)
        # inside if any variant fits; outside, the reason against the variant its ceiling (etc.) matches
        why = None if None in reasons else (why if i is None else reasons[i])
    if why is not None:
        return why
    if not any_candidate_set and record_filter.get("candidate_set") is not None \
            and str(cset) not in allowed_candidate_sets(record_filter):
        return f"candidate set {cset}"
    return None


def allowed_candidate_sets(record_filter: dict[str, Any]) -> list[str]:
    """The candidate sets a filter keeps: its `candidate_set`, plus every set in its optional
    `candidate_sets` allow-list (a window spanning a candidate-set swap, e.g. ["5", "6"]; train must
    then also get --allow-mixed-candidate-sets). Without `candidate_sets`: the one set, as before."""
    own = str(record_filter.get("candidate_set"))
    extra = [str(s) for s in record_filter.get("candidate_sets") or []]
    return [own, *[s for s in extra if s != own]]


def _settings_reason(have: dict[str, Any], want: dict[str, Any]) -> str | None:
    for k in sorted(set(SIM_SETTINGS_STANDARD) | set(want)):
        expected, got = want.get(k, SIM_SETTINGS_STANDARD.get(k)), have.get(k, SIM_SETTINGS_STANDARD.get(k))
        if _canon(got) != _canon(expected):
            return f"{k} {got!r}"
    return None


def apply_record_filter(chunks: dict[str, list[Any]], keys_of: Any, record_filter: dict[str, Any] | None,
                        log: Any = print, allow_match_settings: list[dict[str, Any]] | None = None
                        ) -> dict[str, Any] | None:
    """Drop the files outside `record_filter` from each split of `chunks` (in place; `keys_of(path)`
    gives a file's record_keys) and log what was kept and excluded, with counts. Files under one of
    the filter's `any_candidate_set_dirs` need only match its settings (R-fit 1's set-4 ranked games,
    trained with --allow-mixed-candidate-sets). With `allow_match_settings` (filter_reason), the
    summary also counts the kept records by variant (`kept_by_match_settings`; "other": the filter's
    own settings, matching no variant). Returns the summary (None without a filter)."""
    if not record_filter:
        return None
    from pathlib import Path

    loose = [Path(d).resolve() for d in record_filter.get("any_candidate_set_dirs") or []]
    kept: dict[str, int] = {}
    excluded: dict[str, int] = {}
    loose_kept = 0
    by_variant: dict[str, int] = {}
    by_set: dict[str, int] = {}
    many_sets = bool(record_filter.get("candidate_sets"))
    labels = [variant_label(v) for v in allow_match_settings or []]
    for split, files in chunks.items():
        keep = []
        for f in files:
            any_set = bool(loose) and any(d == Path(f).resolve().parent or d in Path(f).resolve().parents
                                          for d in loose)
            keys = tuple(keys_of(f))
            why = filter_reason(keys, record_filter, any_set, allow_match_settings)
            if why is None:
                keep.append(f)
                loose_kept += any_set
                if many_sets:
                    by_set[str(keys[1])] = by_set.get(str(keys[1]), 0) + 1
                if allow_match_settings:
                    i = match_variant(keys, allow_match_settings)
                    label = "other" if i is None else labels[i]
                    by_variant[label] = by_variant.get(label, 0) + 1
            else:
                excluded[why] = excluded.get(why, 0) + 1
        kept[split] = len(keep)
        files[:] = keep
    summary = {"filter": {k: v for k, v in record_filter.items()}, "kept": kept, "kept_total": sum(kept.values()),
               "excluded_total": sum(excluded.values()), "excluded": dict(sorted(excluded.items())),
               "kept_any_candidate_set": loose_kept}
    if allow_match_settings:
        summary["allow_match_settings"] = [dict(v) for v in allow_match_settings]
        summary["kept_by_match_settings"] = {k: by_variant.get(k, 0) for k in [*labels, "other"]
                                             if k in labels or by_variant.get(k)}
    if many_sets:
        summary["kept_by_candidate_set"] = {s: by_set.get(s, 0) for s in allowed_candidate_sets(record_filter)}
    log(f"record filter (candidate set {record_filter.get('candidate_set')}, settings "
        f"{json.dumps(record_filter.get('match_settings') or {}, sort_keys=True)}"
        + (f", any candidate set in {len(loose)} folder(s)" if loose else "") + "): "
        f"kept {summary['kept_total']} records (" + ", ".join(f"{k} {n}" for k, n in kept.items()) + ")"
        + (f", {loose_kept} of them from the any-candidate-set folders" if loose else "")
        + f"; excluded {summary['excluded_total']}"
        + (" (" + ", ".join(f"{n} x {k}" for k, n in summary["excluded"].items()) + ")" if excluded else "")
        + ("; kept by match settings: " + ", ".join(f"{k} {n}" for k, n in summary["kept_by_match_settings"].items())
           if allow_match_settings else "")
        + ("; kept by candidate set: " + ", ".join(f"{k} {n}" for k, n in summary["kept_by_candidate_set"].items())
           if many_sets else ""))
    return summary


def excludes_sampled_starts(record_filter: dict[str, Any] | None, flag: bool = False) -> bool:
    """Whether a window leaves sampled-start records out: train --exclude-sampled-starts, or the record
    filter's "exclude_sampled_starts": true."""
    return bool(flag or (record_filter or {}).get("exclude_sampled_starts"))


def drop_sampled_starts(chunks: dict[str, list[Any]], sampled_of: Any, log: Any = print) -> dict[str, Any]:
    """Drop sampled-start records (yomibench.sampled_starts: rows beginning mid-game; `sampled_of(path)`
    says which) from every split of `chunks`, in place, e.g. for a holdout that must hold only games from
    the opening. Returns {"excluded": {split: n}, "excluded_total": n}, logged."""
    excluded: dict[str, int] = {}
    for split, files in chunks.items():
        keep = [f for f in files if not sampled_of(f)]
        excluded[split] = len(files) - len(keep)
        files[:] = keep
    out = {"excluded": excluded, "excluded_total": sum(excluded.values())}
    log("sampled starts excluded: " + ", ".join(f"{k} {n}" for k, n in excluded.items()))
    return out


# --- held-out human tests ------------------------------------------------------------------------
# Human-vs-bot games (yomibench.human_play) are a held-out human test: they
# never enter training or a gate's game set. Their records say so in the header (`arm`), and the
# folder they're written to carries a marker file, so the guards below catch them by either route.

HUMAN_PLAY_ARM = "human-play"
HELD_OUT_ARMS = frozenset({HUMAN_PLAY_ARM})
HELD_OUT_MARKER = ".held-out.json"


def held_out_arm(header: dict[str, Any]) -> str | None:
    """The held-out arm a record's header names (None: an ordinary record)."""
    arm = header.get("arm")
    return arm if arm in HELD_OUT_ARMS or header.get("train") is False else None


def mark_held_out(folder: Any, arm: str) -> None:
    """Mark `folder` (and everything under it) as held out: never a training folder."""
    from pathlib import Path

    path = Path(folder) / HELD_OUT_MARKER
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"arm": arm, "train": False,
                                    "why": "a held-out human test: never trained on, never a gate game"}) + "\n",
                        encoding="utf-8")


def held_out_folder(folder: Any) -> str | None:
    """The marker file guarding `folder` (in it or in a folder above it), or None."""
    from pathlib import Path

    p = Path(folder).resolve()
    for d in (p, *p.parents):
        if (d / HELD_OUT_MARKER).exists():
            return str(d / HELD_OUT_MARKER)
    return None


def refuse_held_out(headers: list[tuple[Any, dict[str, Any]]]) -> None:
    """Refuse (SystemExit) if any (path, header) is a held-out record."""
    bad = [(p, held_out_arm(h)) for p, h in headers if held_out_arm(h)]
    if bad:
        raise SystemExit(f"{len(bad)} held-out record(s) (e.g. {bad[0][0]}: arm {bad[0][1]}); human-play games are a "
                         "held-out human test and are never trained on: move them out of the training folders")


def held_out_gz(data: bytes) -> str | None:
    """held_out_arm of a gzipped record given as bytes (only its first line is decompressed)."""
    import gzip
    import io

    try:
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
            return held_out_arm(json.loads(f.readline()))
    except (OSError, ValueError, EOFError):
        return None
