from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def _configured_protomotions_root() -> Path:
    configured = os.environ.get("PROTOMOTIONS_ROOT") or os.environ.get("PROTO_MOTIONS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "code" / "ProtoMotions").resolve()


def resolve_protomotions_root() -> Path:
    spec = find_spec("protomotions")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent.parent

    root = _configured_protomotions_root()
    if (root / "protomotions" / "__init__.py").exists():
        return root

    raise FileNotFoundError(
        "ProtoMotions checkout not found. Set PROTOMOTIONS_ROOT or PROTO_MOTIONS_ROOT "
        f"to a repo containing protomotions/__init__.py. Checked: {root}"
    )


def ensure_protomotions_importable() -> Path:
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
