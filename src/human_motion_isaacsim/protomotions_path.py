from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def _explicit_protomotions_root() -> Path | None:
    """Return the ProtoMotions root from PROTOMOTIONS_ROOT or PROTO_MOTIONS_ROOT env vars, if set."""
    configured = os.environ.get("PROTOMOTIONS_ROOT") or os.environ.get("PROTO_MOTIONS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return None


def _repo_root() -> Path:
    """Return the repository root based on this source file's location."""
    return Path(__file__).resolve().parents[2]


def _repo_local_protomotions_root() -> Path:
    """Return the path to the vendored ProtoMotions checkout under third_party/."""
    return (_repo_root() / "third_party" / "ProtoMotions").resolve()


def _legacy_protomotions_root() -> Path:
    """Return the legacy ~/code/ProtoMotions path used before vendoring."""
    return (Path.home() / "code" / "ProtoMotions").resolve()


def _is_protomotions_repo(root: Path) -> bool:
    """Check whether the given directory contains a protomotions Python package."""
    return (root / "protomotions" / "__init__.py").exists()


def resolve_protomotions_root() -> Path:
    """Find the ProtoMotions root by checking imports, env vars, and well-known paths."""
    spec = find_spec("protomotions")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent.parent

    explicit_root = _explicit_protomotions_root()
    if explicit_root is not None:
        if _is_protomotions_repo(explicit_root):
            return explicit_root
        raise FileNotFoundError(
            "ProtoMotions checkout not found. Set PROTOMOTIONS_ROOT or PROTO_MOTIONS_ROOT "
            f"to a repo containing protomotions/__init__.py. Checked: {explicit_root}"
        )

    checked_roots = []
    for root in (_repo_local_protomotions_root(), _legacy_protomotions_root()):
        checked_roots.append(str(root))
        if _is_protomotions_repo(root):
            return root

    raise FileNotFoundError(
        "ProtoMotions checkout not found. Checked: "
        + ", ".join(checked_roots)
        + ". Initialize third_party/ProtoMotions or set PROTOMOTIONS_ROOT / PROTO_MOTIONS_ROOT."
    )


def ensure_protomotions_importable() -> Path:
    """Ensure the protomotions package is importable, adding its root to sys.path if needed."""
    spec = find_spec("protomotions")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent.parent

    root = resolve_protomotions_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    spec = find_spec("protomotions")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Unable to import protomotions from {root}")

    return root
