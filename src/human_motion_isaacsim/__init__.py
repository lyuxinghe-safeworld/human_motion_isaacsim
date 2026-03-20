__all__ = [
    "MotionMetadata",
    "MotionRunResult",
    "MotionController",
    "init",
    "run",
    "list_models",
    "load_motion_metadata",
    "viewport_capture",
]


def __getattr__(name: str):
    if name == "MotionMetadata":
        from human_motion_isaacsim.motion_file import MotionMetadata

        return MotionMetadata
    if name == "MotionRunResult":
        from human_motion_isaacsim.result import MotionRunResult

        return MotionRunResult
    if name == "MotionController":
        from human_motion_isaacsim.motion_runner import MotionController

        return MotionController
    if name in {"init", "run"}:
        from human_motion_isaacsim import _api

        return getattr(_api, name)
    if name == "list_models":
        from human_motion_isaacsim._registry import list_models

        return list_models
    if name == "load_motion_metadata":
        from human_motion_isaacsim.motion_file import load_motion_metadata

        return load_motion_metadata
    if name == "viewport_capture":
        import importlib

        return importlib.import_module("human_motion_isaacsim.viewport_capture")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
