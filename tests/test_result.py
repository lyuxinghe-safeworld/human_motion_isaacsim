def test_package_exports_controller_symbol():
    import human_motion_isaacsim

    assert hasattr(human_motion_isaacsim, "ProtoMotionIsaacSimController")


def test_env_docs_reference_known_good_versions():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "isaacsim==5.1.0.0" in text
    assert "isaaclab==2.3.0" in text
    assert "Python 3.11" in text


def test_motion_run_result_defaults():
    from pathlib import Path

    from human_motion_isaacsim import MotionRunResult

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


def test_docs_reference_run_custom_scene_wrapper_script():
    from pathlib import Path

    readme_text = Path("README.md").read_text()
    env_readme_text = Path("env/README.md").read_text()

    assert "scripts/run_custom_scene.sh" in readme_text
    assert "scripts/run_custom_scene.sh" in env_readme_text


def test_env_readme_documents_vnc_monitor_flow():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "TurboVNC" in text
    assert "DISPLAY=:1" in text
    assert "scripts/test_isaacsim_monitor.py" in text


def test_install_docs_use_uv_consistently():
    from pathlib import Path

    readme_text = Path("README.md").read_text()
    env_readme_text = Path("env/README.md").read_text()
    install_text = Path("env/install.sh").read_text()

    assert "uv" in readme_text.lower()
    assert "uv" in env_readme_text.lower()
    assert "uv venv env/.venv --python 3.11" in install_text


def test_repo_includes_protomotions_submodule_configuration():
    from pathlib import Path

    text = Path(".gitmodules").read_text()

    assert "third_party/ProtoMotions" in text
    assert "git@github.com:NVlabs/ProtoMotions.git" in text


def test_install_script_builds_unified_protomotions_env():
    from pathlib import Path

    text = Path("env/install.sh").read_text()

    assert "torchvision==0.22.0" in text
    assert 'git -C "$PROTO_ROOT" lfs pull' in text
    assert 'uv pip install --python env/.venv/bin/python -e "$PROTO_ROOT"' in text
    assert 'uv pip install --python env/.venv/bin/python -e .' in text


def test_env_readme_mentions_git_lfs_for_protomotions_assets():
    from pathlib import Path

    text = Path("env/README.md").read_text()

    assert "git lfs" in text.lower()
