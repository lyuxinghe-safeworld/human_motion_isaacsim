def test_package_exports_controller_symbol():
    import hymotion_isaacsim

    assert hasattr(hymotion_isaacsim, "ProtoMotionIsaacSimController")


def test_env_docs_reference_known_good_versions():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "isaacsim==5.1.0.0" in text
    assert "isaaclab==2.3.0" in text
    assert "Python 3.11" in text


def test_motion_run_result_defaults():
    from pathlib import Path

    from hymotion_isaacsim import MotionRunResult

    result = MotionRunResult(success=True, motion_file=Path("walk.motion"))

    assert result.success is True
    assert result.motion_file == Path("walk.motion")
    assert result.video_output is None
    assert result.num_steps == 0
    assert result.duration_seconds == 0.0
    assert result.error_message is None


def test_env_readme_contains_smoke_command():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "scripts/smoke_run_motion.py" in text


def test_env_readme_documents_vnc_monitor_flow():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "TurboVNC" in text
    assert "DISPLAY=:1" in text
    assert "scripts/test_isaacsim_monitor.py" in text
