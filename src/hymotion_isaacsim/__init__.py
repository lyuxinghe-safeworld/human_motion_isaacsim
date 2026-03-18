__all__ = [
    "MotionMetadata",
    "MotionRunResult",
    "ProtoMotionIsaacSimController",
    "load_motion_metadata",
    "recording",
]


def __getattr__(name: str):
    if name == "MotionMetadata":
        from hymotion_isaacsim.motion_file import MotionMetadata

        return MotionMetadata
    if name == "MotionRunResult":
        from hymotion_isaacsim.result import MotionRunResult

        return MotionRunResult
    if name == "ProtoMotionIsaacSimController":
        from hymotion_isaacsim.runtime import ProtoMotionIsaacSimController

        return ProtoMotionIsaacSimController
    if name == "load_motion_metadata":
        from hymotion_isaacsim.motion_file import load_motion_metadata

        return load_motion_metadata
    if name == "recording":
        import importlib

        return importlib.import_module("hymotion_isaacsim.recording")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
