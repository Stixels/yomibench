"""Quick-turn menus picked by a policy net (cheap_cap_policy `policy40+gumbel8`, `policy40+gumbel8+parryx6`).

Off by default. Human play uses it through yomibench.anytime (human_play --quick-policy PATH
--quick-policy-sha256 SHA): its emergency quick turn, the production search's double-oracle warm start and
the two-decision stages' child menus. The code is public; a policy file ships only with a bot build.
 A quick (playout-capped) decision searches a short menu of each seat's candidates; today
`playout.cap_candidates(48, "options-shuffle")` picks it without looking at the position. Here a pinned
policy net (yomibench.policy.PolicyNet, trained on full-turn search mixes; pure numpy inference) scores the
seat's base candidates and the menu is (`policy_menu`, salt `seed:turn:seat`):

  guard    every Continue variant, and for every move (a run of one action, as cap_candidates) its variant
           with the highest perturbed score, so every move stays offered (as options-shuffle's guard);
  policy   Gumbel top-k: perturbed score = policy logit + Gumbel(0, 1) noise from a generator seeded by
           sha256("seed:turn:seat:policy-menu"); the highest perturbed scores join until the menu holds K
           (= 40);
  random   G (= 8) slots drawn uniformly without replacement from the rest (same generator), until the
           menu holds K + G (= 48, the cheap cap).

Kept candidates stay in list order; a list of at most K + G candidates is kept whole; the guard may exceed
the cap (as options-shuffle's does). The G random slots are the exploration guard against a near-zero
prior: every candidate keeps a chance on every quick turn. Full turns never use the net.

`+parryx6`: the policy menu above, then exactly options-shuffle+parryx6's addition on top
(`playout.add_extra_count_options`): for every move with a count widget (e.g. ParryHigh's timing), its
first 6 count options per value class in the options-shuffle-counts order, each by its seeded variant,
added beyond the cap only if no candidate already in the menu shows that option. So an option the policy
(or a random slot) already picked is never added twice. The menu grows by at most 6 per value class of every
move with a count widget (ParryHigh's two heights: 12; a list with several count moves or classes can add more).

The net is pinned by file and sha256 (`QuickPolicy.load`, or `install` for a process-wide one; --quick-policy
PATH --quick-policy-sha256 SHA): a missing file or a sha mismatch refuses to start. Records name it in the header's playout_cap
(`quick_policy`: {"sha256", "file"}). Same seed, same file -> the same menu (numpy float32 forward pass
and a seeded generator; a different BLAS could in principle reorder near-ties of the perturbed scores).
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import numpy as np

QUICK_POLICIES = ("policy40+gumbel8", "policy40+gumbel8+parryx6")
_NAME = re.compile(r"^policy(\d+)\+gumbel(\d+)(?:\+parryx(\d+))?$")


def is_quick_policy(policy: str | None) -> bool:
    return bool(policy) and policy in QUICK_POLICIES


def parse(policy: str) -> tuple[int, int, int]:
    """(K policy slots, G random slots, T parry timings per class or 0) of a quick-policy name."""
    m = _NAME.match(policy or "")
    if not m or policy not in QUICK_POLICIES:
        raise ValueError(f"not a quick policy: {policy!r} (one of {QUICK_POLICIES})")
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def check_cap(policy: str, cap: int) -> None:
    k, g, _ = parse(policy)
    if cap != k + g:
        raise ValueError(f"cheap_cap_policy {policy} needs --cheap-cap {k + g} (K + G), not {cap}")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rng(salt: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{salt}:policy-menu".encode()).hexdigest()[:16], 16))


def _moves(cands: list[Any]) -> list[list[int]]:
    moves: list[list[int]] = []
    for i, c in enumerate(cands):
        if moves and cands[moves[-1][0]].action == c.action:
            moves[-1].append(i)
        else:
            moves.append([i])
    return moves


def policy_menu(cands: list[Any], scores: Any, k: int, g: int, salt: str) -> list[Any]:
    """One seat's menu (module docstring): guard, Gumbel top-k to `k`, `g` uniform random slots."""
    cands = list(cands)
    n, cap = len(cands), k + g
    if n <= cap:
        return cands
    rng = _rng(salt)
    pert = np.asarray(scores, dtype=np.float64) + rng.gumbel(size=n)
    keep: set[int] = set()
    for m in _moves(cands):
        if cands[m[0]].action == "Continue":
            keep.update(m)
        else:
            keep.add(max(m, key=lambda i: pert[i]))
    for i in np.argsort(-pert, kind="stable"):
        if len(keep) >= k:
            break
        keep.add(int(i))
    for i in rng.permutation(n).tolist():
        if len(keep) >= cap:
            break
        keep.add(i)
    return [c for i, c in enumerate(cands) if i in keep]


class QuickPolicy:
    """A pinned policy net for quick menus. `net` needs `scores(state, seat, candidates) -> array`."""

    def __init__(self, net: Any, sha256: str, file: str):
        self.net, self.sha256, self.file = net, sha256, file

    @staticmethod
    def load(path: str | Path, sha256: str) -> "QuickPolicy":
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"--quick-policy {path}: no such file (refusing to start)")
        data = p.read_bytes()  # hash and load the same bytes: no window between the check and the load
        got = hashlib.sha256(data).hexdigest()
        if got != (sha256 or "").strip().lower():
            raise SystemExit(f"--quick-policy {path}: sha256 {got} != expected {sha256!r} (refusing to start)")
        from yomibench.policy import PolicyNet

        return QuickPolicy(PolicyNet.load(io.BytesIO(data)), got, p.name)

    def header(self) -> dict[str, str]:
        return {"sha256": self.sha256, "file": self.file}

    def scores(self, state: dict[str, Any], seat: int, candidates: list[Any]) -> Any:
        """The net's score for each of `seat`'s candidates (higher: likelier)."""
        return self.net.scores(state, seat, list(candidates))

    def menus(self, state: dict[str, Any], candidates: Any, cap: int, policy: str, salt: str) -> list[list[Any]]:
        """Both seats' quick menus (salt `salt:seat`, salt = "seed:turn")."""
        return [m for m, _ in self.scored_menus(state, candidates, cap, policy, salt)]

    def scored_menus(self, state: dict[str, Any], candidates: Any, cap: int, policy: str, salt: str,
                     need_scores: bool = False) -> list[tuple[list[Any], Any]]:
        """`menus`, each with its candidates' scores (in menu order; None for a list kept whole without
        scoring it, unless `need_scores`). The menus are exactly `menus`'."""
        from yomibench.playout import add_extra_count_options

        check_cap(policy, cap)
        k, g, t = parse(policy)
        out = []
        for seat, cands in enumerate(candidates):
            cands = list(cands)
            s = f"{salt}:{seat}"
            if len(cands) <= cap:
                out.append((cands, self.net.scores(state, seat, cands) if need_scores and cands else None))
                continue
            sc = self.net.scores(state, seat, cands)
            menu = policy_menu(cands, sc, k, g, s)
            if t:
                menu = add_extra_count_options(cands, menu, s, t)
            at = {c.id: i for i, c in enumerate(cands)}
            out.append((menu, np.asarray([sc[at[c.id]] for c in menu])))
        return out


ACTIVE: QuickPolicy | None = None


def install(path: str | Path, sha256: str) -> QuickPolicy:
    """Load and pin the process's quick policy (selfplay --quick-policy / --quick-policy-sha256)."""
    global ACTIVE
    ACTIVE = QuickPolicy.load(path, sha256)
    return ACTIVE


def active(policy: str) -> QuickPolicy:
    if ACTIVE is None:
        raise ValueError(f"cheap_cap_policy {policy} needs a pinned policy file (--quick-policy PATH "
                         "--quick-policy-sha256 SHA)")
    return ACTIVE
