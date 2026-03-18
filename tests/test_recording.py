import asyncio
from pathlib import Path

import pytest


def test_capture_active_viewport_to_file_flushes_async_capture(tmp_path, monkeypatch):
    import hymotion_isaacsim.recording as recording

    output_path = tmp_path / "frame.png"
    calls = []

    class _FakeCaptureHelper:
        async def wait_for_result(self):
            calls.append("wait_for_result")

    class _FakeViewportUtility:
        @staticmethod
        def get_active_viewport():
            calls.append("get_active_viewport")
            return "viewport"

        @staticmethod
        def capture_viewport_to_file(viewport, file_path):
            calls.append(("capture_viewport_to_file", viewport, file_path))
            return _FakeCaptureHelper()

    class _FakeKitApp:
        @staticmethod
        def get_app():
            class _FakeAppInterface:
                async def next_update_async(self):
                    calls.append("next_update_async")

            return _FakeAppInterface()

    class _FakeRendererCapture:
        @staticmethod
        def acquire_renderer_capture_interface():
            class _FakeCaptureInterface:
                def wait_async_capture(self):
                    calls.append("wait_async_capture")

            return _FakeCaptureInterface()

    class _FakeSimulationApp:
        def run_coroutine(self, coro):
            calls.append("run_coroutine")
            return asyncio.run(coro)

    def fake_import_module(name):
        mapping = {
            "omni.kit.viewport.utility": _FakeViewportUtility,
            "omni.kit.app": _FakeKitApp,
            "omni.renderer_capture": _FakeRendererCapture,
        }
        return mapping[name]

    monkeypatch.setattr(recording.importlib, "import_module", fake_import_module)

    recording.capture_active_viewport_to_file(
        output_path,
        simulation_app=_FakeSimulationApp(),
        flush_updates=2,
    )

    assert calls == [
        "run_coroutine",
        "get_active_viewport",
        ("capture_viewport_to_file", "viewport", str(output_path)),
        "wait_for_result",
        "wait_async_capture",
        "next_update_async",
        "wait_async_capture",
        "next_update_async",
    ]


def test_capture_active_viewport_to_file_requires_simulation_app(tmp_path):
    from hymotion_isaacsim.recording import capture_active_viewport_to_file

    with pytest.raises(ValueError, match="simulation_app"):
        capture_active_viewport_to_file(Path(tmp_path / "frame.png"), simulation_app=None)
