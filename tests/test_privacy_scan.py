"""tools/privacy_scan.py: this repository passes its generic rules; the rules and a deny file catch what they say."""
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("privacy_scan", ROOT / "tools" / "privacy_scan.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def test_this_repository_is_clean():
    if not (ROOT / ".git").exists():
        return  # an sdist or copy without git: nothing to list
    assert ps.scan_repo(ROOT, ps.rules()) == []


def test_generic_rules_and_a_deny_file(tmp_path):
    deny = tmp_path / "deny.txt"
    deny.write_text("# private\n(?i)secretbox\n", encoding="utf-8")
    rs = ps.rules(deny)
    text = "\n".join(["hub at 100.101.2.3:8765", "mail me@somewhere.com", r"C:\Users\someone\x",
                      "/home/someone/x", "on SecretBox tonight", "fine line", "x@users.noreply.github.com"])
    whys = [h["why"] for h in ps.scan_text(rs, "notes.md", text)]
    assert whys == ["tailnet IP", "email", "local path", "home path", "deny-list term"]


def test_commit_people_are_checked(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("ok\n", encoding="utf-8")
    git("add", "a.txt")
    git("-c", "user.name=x", "-c", "user.email=x@users.noreply.github.com", "commit", "-q", "-m", "one")
    git("-c", "user.name=y", "-c", "user.email=y@mailhost.com", "commit", "-q", "--allow-empty", "-m", "two")
    hits = ps.scan_commits(tmp_path, ps.rules(), "HEAD")
    assert {h["why"] for h in hits} == {"email"} and len({h["path"] for h in hits}) == 1  # commit two only


def test_arena_results_are_allowed_data_but_never_name_a_host(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "arena" / "results").mkdir(parents=True)
    (tmp_path / "arena" / "results" / "ranked.jsonl").write_text('{"a":"gen-1","actual":{"git_sha":"x"}}\n',
                                                                   encoding="utf-8")
    (tmp_path / "other.jsonl").write_text("{}\n", encoding="utf-8")
    git("add", ".")
    hits = ps.scan_repo(tmp_path, ps.rules())
    assert [(h["path"], h["why"]) for h in hits] == [("other.jsonl", "asset, data or binary file")]
    (tmp_path / "arena" / "results" / "ranked.jsonl").write_text('{"a":"gen-1","actual":{"host":"box"}}\n',
                                                                   encoding="utf-8")
    hits = ps.scan_repo(tmp_path, ps.rules())
    assert ("arena/results/ranked.jsonl", "host or machine name in a results row") in [(h["path"], h["why"])
                                                                                       for h in hits]
