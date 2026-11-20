"""The encoder identity (cache.ENCODER) is pinned, and guarded: the code that decides encoded rows is fingerprinted
with comments, docstrings and formatting removed, so a wording edit passes and a code edit fails here.

When this fails, the encoding inputs changed (cache.encoder_inputs). Then either
- rows can change: bump cache.ENCODER (every cache rebuilds) and update FINGERPRINT, or
- rows can't change (a rename, a refactor): prove it by encoding a fixed set of records with the old and the new
  code and comparing every row byte for byte, then update FINGERPRINT only.
"""
import ast
import hashlib
import textwrap

from yomibench import cache

FINGERPRINT = "d1bea9d79c533ea0"  # the same for the trainer code that built the live caches


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _dump(node) -> str:
    """A version-stable dump: node types and fields, skipping empty and None fields (newer Pythons add some)."""
    if isinstance(node, ast.AST):
        fields = []
        for name in node._fields:
            v = getattr(node, name, None)
            if v is None or v == []:
                continue
            fields.append(f"{name}={_dump(v)}")
        return f"{type(node).__name__}({','.join(fields)})"
    if isinstance(node, list):
        return "[" + ",".join(_dump(x) for x in node) + "]"
    return repr(node)


def fingerprint() -> str:
    parts = []
    for src in cache.encoder_inputs():
        try:
            parts.append(_dump(_strip_docstrings(ast.parse(textwrap.dedent(src)))))
        except SyntaxError:  # the "format N" / "features N" lines
            parts.append(src)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def test_encoder_identity_is_pinned():
    assert cache.encoder_hash() == cache.ENCODER == "853f7784eeb12ad1"


def test_encoding_inputs_are_unchanged():
    assert fingerprint() == FINGERPRINT, (
        "encoding inputs changed (cache.encoder_inputs): re-run an encode parity check, then bump cache.ENCODER "
        "(forces a re-encode) or, if rows are byte-identical, update FINGERPRINT in this test")


def test_docstrings_and_comments_dont_count():
    src = 'def f(x):\n    """doc"""\n    # note\n    return x + 1\n'
    other = 'def f(x):\n    """other words"""\n    return  x+1  # different note\n'
    assert _dump(_strip_docstrings(ast.parse(src))) == _dump(_strip_docstrings(ast.parse(other)))
    assert _dump(_strip_docstrings(ast.parse(src))) != _dump(_strip_docstrings(ast.parse(src.replace("+ 1", "+ 2"))))
