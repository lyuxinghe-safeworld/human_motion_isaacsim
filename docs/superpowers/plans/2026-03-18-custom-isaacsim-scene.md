# Custom Isaac Sim Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure Isaac Sim environment (no Isaac Lab) with ground plane, static objects, and SMPL humanoid controlled by ProtoMotions, supporting both standalone and ProtoMotions-controlled modes.

**Architecture:** Entry script creates `SimulationApp` + `World` + `Articulation` + static objects using `omni.isaac.core`. A custom `IsaacSimSimulator` adapter subclasses ProtoMotions' `Simulator` base class and wraps the already-created world/articulation. In ProtoMotions mode, the standard `Env` + `Agent` inference loop drives the humanoid. In standalone mode, the humanoid stands in rest pose.

**Tech Stack:** `omni.isaac.core` (World, Articulation, GroundPlane, rigid body prims), `omni.isaac.sensor` (ContactSensor), ProtoMotions (`Simulator` base class, `Env`, `Agent`, `SceneLib`, `MotionLib`), PyTorch.

**Spec:** `docs/superpowers/specs/2026-03-18-custom-isaacsim-scene-design.md`

---

### File Structure

| File | Responsibility |
|---|---|
| `src/hymotion_isaacsim/custom_scene.py` | `populate_scene(world)` — adds hardcoded static objects to the USD stage |
| `src/hymotion_isaacsim/isaacsim_simulator.py` | `IsaacSimSimulator` — adapter subclassing ProtoMotions' `Simulator`, wrapping `omni.isaac.core.World` + `Articulation` |
| `scripts/run_custom_scene.py` | Entry script — CLI, scene setup, two-mode dispatch (standalone vs ProtoMotions) |
| `tests/test_custom_scene.py` | Unit tests for `populate_scene` |
| `tests/test_isaacsim_simulator.py` | Unit tests for the simulator adapter (mocked articulation) |

---

### Task 1: Scene Builder (`custom_scene.py`)

**Files:**
- Create: `src/hymotion_isaacsim/custom_scene.py`
- Create: `tests/test_custom_scene.py`

- [ ] **Step 1: Write the failing test**

`tests/test_custom_scene.py`:
```python
from __future__ import annotations

import pytest


def test_populate_scene_adds_three_objects():
    """populate_scene should add box, cylinder, sphere to the world."""
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS, populate_scene

    assert len(SCENE_OBJECTS) == 3

    added = []

    class _FakeWorld:
        class scene:
            @staticmethod
            def add(obj):
                added.append(obj)

    populate_scene(_FakeWorld())
    assert len(added) == 3


def test_scene_objects_have_expected_prim_paths():
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS

    paths = [obj["prim_path"] for obj in SCENE_OBJECTS]
    assert all(p.startswith("/World/custom_scene/") for p in paths)


def test_scene_objects_are_static():
    """All scene objects should have fixed_base=True (static)."""
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS

    for obj in SCENE_OBJECTS:
        assert obj.get("fixed_base", False) is True, f"{obj['prim_path']} is not static"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_custom_scene.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `custom_scene.py`**

`src/hymotion_isaacsim/custom_scene.py`:
```python
from __future__ import annotations

from typing import Any

SCENE_OBJECTS = [
    {
        "type": "cube",
        "prim_path": "/World/custom_scene/box",
        "size": 1.0,
        "position": (2.0, 1.0, 0.5),
        "color": (0.8, 0.2, 0.2),
        "fixed_base": True,
    },
    {
        "type": "cylinder",
        "prim_path": "/World/custom_scene/cylinder",
        "radius": 0.3,
        "height": 1.5,
        "position": (-1.0, 2.0, 0.75),
        "color": (0.2, 0.8, 0.2),
        "fixed_base": True,
    },
    {
        "type": "sphere",
        "prim_path": "/World/custom_scene/sphere",
        "radius": 0.5,
        "position": (1.0, -1.5, 0.5),
        "color": (0.2, 0.2, 0.8),
        "fixed_base": True,
    },
]


def populate_scene(world: Any) -> None:
    """Add hardcoded static objects to the Isaac Sim world.

    Uses omni.isaac.core object prims. Each object is a rigid body with a
    collider and fixed_base=True (static — won't fall or move).
    """
    import numpy as np
    from omni.isaac.core.objects import FixedCuboid, FixedCylinder, FixedSphere

    _BUILDERS = {
        "cube": lambda obj: FixedCuboid(
            prim_path=obj["prim_path"],
            size=obj["size"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
        "cylinder": lambda obj: FixedCylinder(
            prim_path=obj["prim_path"],
            radius=obj["radius"],
            height=obj["height"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
        "sphere": lambda obj: FixedSphere(
            prim_path=obj["prim_path"],
            radius=obj["radius"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
    }

    for obj in SCENE_OBJECTS:
        prim = _BUILDERS[obj["type"]](obj)
        world.scene.add(prim)
```

Note: `FixedCuboid`/`FixedCylinder`/`FixedSphere` from `omni.isaac.core.objects` create static rigid bodies with colliders that won't fall or move. If the Isaac Sim version does not have these classes, fall back to `DynamicCuboid` etc. with `mass=0`. The test uses a mock world so it validates the structure, not the Isaac Sim API.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_custom_scene.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hymotion_isaacsim/custom_scene.py tests/test_custom_scene.py
git commit -m "feat: add custom_scene module with hardcoded static objects"
```

---

### Task 2: Simulator Adapter — Constructor and Body Ordering (`isaacsim_simulator.py`, part 1)

**Files:**
- Create: `src/hymotion_isaacsim/isaacsim_simulator.py`
- Create: `tests/test_isaacsim_simulator.py`

**Reference files (read, don't modify):**
- `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/base_simulator/simulator.py` — base class
- `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/base_simulator/config.py` — `SimBodyOrdering`
- `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/base_simulator/simulator_state.py` — `RootOnlyState`, `RobotState`, `ResetState`, `StateConversion`, `DataConversionMapping`
- `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/isaaclab/simulator.py` — reference implementation

This task creates the adapter class shell with constructor and `_get_sim_body_ordering()`. All other abstract methods are stubbed with `raise NotImplementedError` — they'll be filled in subsequent tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_isaacsim_simulator.py`:
```python
from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_fake_articulation(body_names=None, dof_names=None):
    """Create a mock articulation with expected SMPL body/DOF names."""
    from hymotion_isaacsim.binding import EXPECTED_SMPL_BODY_NAMES, EXPECTED_SMPL_JOINT_NAMES

    art = MagicMock()
    art.body_names = list(body_names or EXPECTED_SMPL_BODY_NAMES)
    art.dof_names = list(dof_names or EXPECTED_SMPL_JOINT_NAMES)
    art.num_bodies = len(art.body_names)
    art.num_dof = len(art.dof_names)
    return art


def _make_minimal_configs():
    """Create minimal configs matching what the Simulator base class expects."""
    sim_config = SimpleNamespace(
        num_envs=1,
        headless=True,
        sim=SimpleNamespace(fps=60, decimation=4),
        control=SimpleNamespace(
            control_type="BUILT_IN_PD",
        ),
        domain_randomization=SimpleNamespace(),
        w_last=False,
    )
    robot_config = SimpleNamespace(
        asset=SimpleNamespace(
            asset_root="/tmp/assets",
            asset_file_name="mjcf/smpl.xml",
            usd_asset_file_name="usd/smpl/smpl.usda",
        ),
    )
    return sim_config, robot_config


class TestIsaacSimSimulatorConstruction:
    def test_adapter_stores_world_and_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        world = MagicMock()
        sim_app = MagicMock()
        sim_config, robot_config = _make_minimal_configs()

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = world
        adapter._articulation = art
        adapter._simulation_app = sim_app

        assert adapter._world is world
        assert adapter._articulation is art
        assert adapter._simulation_app is sim_app

    def test_get_sim_body_ordering_returns_articulation_names(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation(
            body_names=["Pelvis", "L_Hip", "R_Hip"],
            dof_names=["L_Hip_x", "L_Hip_y", "R_Hip_x"],
        )

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art

        ordering = adapter._get_sim_body_ordering()
        assert ordering.body_names == ["Pelvis", "L_Hip", "R_Hip"]
        assert ordering.dof_names == ["L_Hip_x", "L_Hip_y", "R_Hip_x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py -v`
Expected: FAIL — `ImportError` (module doesn't exist)

- [ ] **Step 3: Implement adapter shell**

`src/hymotion_isaacsim/isaacsim_simulator.py`:
```python
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from protomotions.simulator.base_simulator.config import SimBodyOrdering
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)


class IsaacSimSimulator(Simulator):
    """ProtoMotions Simulator adapter backed by omni.isaac.core World + Articulation.

    This adapter wraps an already-created World, SimulationApp, and humanoid
    Articulation. It does NOT create them — the entry script does.
    """

    def __init__(
        self,
        *,
        world: Any,
        articulation: Any,
        simulation_app: Any,
        config: Any,
        robot_config: Any,
        scene_lib: Any,
        device: torch.device | str,
        terrain: Any = None,
    ) -> None:
        self._world = world
        self._articulation = articulation
        self._simulation_app = simulation_app

        super().__init__(
            config=config,
            robot_config=robot_config,
            terrain=terrain,
            device=device,
            scene_lib=scene_lib,
        )

    # ── Initialization ──────────────────────────────────────────────

    def _create_simulation(self) -> None:
        # No-op: world and articulation already exist.
        pass

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        return SimBodyOrdering(
            body_names=list(self._articulation.body_names),
            dof_names=list(self._articulation.dof_names),
        )

    def _init_camera(self) -> None:
        raise NotImplementedError("Task 3")

    # ── State getters ───────────────────────────────────────────────

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        raise NotImplementedError("Task 3")

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        raise NotImplementedError("Task 3")

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        raise NotImplementedError("Task 3")

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        raise NotImplementedError("Task 3")

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Task 3")

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError("Task 4")

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        raise NotImplementedError("Task 3")

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        raise NotImplementedError("Task 3")

    # ── State setters ───────────────────────────────────────────────

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError("Task 3")

    # ── Control ─────────────────────────────────────────────────────

    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        raise NotImplementedError("Task 3")

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        raise NotImplementedError("Task 3")

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError("Task 3")

    # ── Physics ─────────────────────────────────────────────────────

    def _physics_step(self) -> None:
        raise NotImplementedError("Task 3")

    # ── Visualization ───────────────────────────────────────────────

    def _write_viewport_to_file(self, file_name: str) -> None:
        raise NotImplementedError("Task 5")

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, Any]] = None
    ) -> None:
        raise NotImplementedError("Task 5")

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        super().close()
        if self._simulation_app is not None and hasattr(self._simulation_app, "close"):
            self._simulation_app.close()
```

**Important implementation note:** The `super().__init__()` call invokes `Simulator.__init__()` which does significant setup (parsing configs, allocating buffers). During implementation, read the base class `__init__` carefully (lines 113–200 of `protomotions/simulator/base_simulator/simulator.py`) and ensure the config objects passed in have all required fields.

The `config` (simulator config) must have at minimum: `num_envs`, `headless`, `sim.fps`, `sim.decimation`, `control.control_type`, `domain_randomization`, `w_last`, `experiment_name`.

The `robot_config` must have at minimum: `control.control_type`, `kinematic_info.num_bodies`, `kinematic_info.num_dofs`, `kinematic_info.dof_names`, `kinematic_info.body_names`, `number_of_actions`, `asset.*`.

Both come from the checkpoint's `resolved_configs_inference.pt`, so they should already have these fields. But the test helper `_make_minimal_configs()` is intentionally incomplete — it's used only with `__new__` (bypassing `__init__`). For a real constructor call, use the full checkpoint configs.

If the base `__init__` calls any abstract methods during construction, those methods must not be `NotImplementedError` stubs — move their implementation to this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hymotion_isaacsim/isaacsim_simulator.py tests/test_isaacsim_simulator.py
git commit -m "feat: add IsaacSimSimulator adapter shell with body ordering"
```

---

### Task 3: Simulator Adapter — State Read/Write, Control, and Physics

**Files:**
- Modify: `src/hymotion_isaacsim/isaacsim_simulator.py`
- Modify: `tests/test_isaacsim_simulator.py`

**Reference:** `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/isaaclab/simulator.py` — follow the same patterns for state reading/writing, adapting from Isaac Lab's `self._robot.data.*` to `omni.isaac.core`'s `articulation.*` APIs.

This task implements all the core abstract methods: state getters, state setter, control application, physics stepping, camera init, and the empty object state returns.

- [ ] **Step 1: Write failing tests for state getters**

Add to `tests/test_isaacsim_simulator.py`:
```python
class TestStateGetters:
    def test_get_simulator_root_state_returns_root_only_state(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        # Mock articulation to return known values
        art.get_world_poses.return_value = (
            torch.tensor([[1.0, 2.0, 3.0]]),  # positions
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # quaternions (wxyz)
        )
        art.get_velocities.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter.config = SimpleNamespace(w_last=False)

        state = adapter._get_simulator_root_state()
        assert state.root_pos.shape == (1, 3)
        assert state.root_rot.shape == (1, 4)
        assert state.root_vel.shape == (1, 3)
        assert state.root_ang_vel.shape == (1, 3)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_dof_state_returns_robot_state(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        num_dof = 69  # SMPL: 23 joints × 3 DOF each
        art = _make_fake_articulation()
        art.get_joint_positions.return_value = torch.zeros((1, num_dof))
        art.get_joint_velocities.return_value = torch.zeros((1, num_dof))

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1

        state = adapter._get_simulator_dof_state()
        assert state.dof_pos.shape == (1, num_dof)
        assert state.dof_vel.shape == (1, num_dof)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_object_root_state_returns_empty(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter.scene_lib = MagicMock()
        adapter.scene_lib.num_scenes.return_value = 0

        state = adapter._get_simulator_object_root_state()
        assert state.root_pos is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestStateGetters -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement state getters**

Replace the `NotImplementedError` stubs in `isaacsim_simulator.py` for:
- `_get_simulator_root_state`: call `articulation.get_world_poses()` for position + quaternion (wxyz), `articulation.get_velocities()` for linear + angular velocity. Return `RootOnlyState` with `state_conversion=StateConversion.SIMULATOR`.
- `_get_simulator_dof_state`: call `articulation.get_joint_positions()` and `articulation.get_joint_velocities()`. Return `RobotState` with `state_conversion=StateConversion.SIMULATOR`.
- `_get_simulator_bodies_state`: call `articulation.get_world_poses()` per body or use the rigid body view API. Return `RobotState` with `rigid_body_pos`, `rigid_body_rot`, `rigid_body_vel`, `rigid_body_ang_vel`.
- `_get_simulator_dof_forces`: call `articulation.get_measured_joint_forces()`. Return `RobotState` with `dof_forces`.
- `_get_simulator_dof_limits_for_verification`: call `articulation.get_dof_limits()`. Return `(lower, upper)` tensors.
- `_get_simulator_object_root_state`: return `ObjectState` with zero tensors (no SceneLib objects).
- `_get_simulator_object_contact_buf`: return `ObjectState` with zero tensors.
- `_init_camera`: set a sensible default camera view.

**Key:** All state getters must handle `env_ids` parameter — when not None, index into the batch dimension. Since `num_envs=1`, this is typically `None` or `tensor([0])`, but implement it correctly for consistency with the base class contract.

- [ ] **Step 4: Run state getter tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestStateGetters -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for state setter and control**

Add to `tests/test_isaacsim_simulator.py`:
```python
class TestStateSetterAndControl:
    def test_set_simulator_env_state_writes_to_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1

        new_states = ResetState(
            root_pos=torch.zeros((1, 3)),
            root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_vel=torch.zeros((1, 3)),
            root_ang_vel=torch.zeros((1, 3)),
            dof_pos=torch.zeros((1, 69)),
            dof_vel=torch.zeros((1, 69)),
            state_conversion=StateConversion.SIMULATOR,
        )

        adapter._set_simulator_env_state(new_states, env_ids=torch.tensor([0]))
        art.set_world_poses.assert_called_once()
        art.set_joint_positions.assert_called_once()
        art.set_joint_velocities.assert_called_once()

    def test_apply_simulator_pd_targets_calls_apply_action(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1

        targets = torch.zeros((1, 69))
        adapter._apply_simulator_pd_targets(targets)
        art.apply_action.assert_called_once()

    def test_apply_simulator_torques_calls_apply_action(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1

        torques = torch.zeros((1, 69))
        adapter._apply_simulator_torques(torques)
        art.apply_action.assert_called_once()

    def test_apply_root_velocity_impulse_sets_velocity(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1

        lin_vel = torch.tensor([[1.0, 0.0, 0.0]])
        ang_vel = torch.tensor([[0.0, 0.0, 1.0]])
        env_ids = torch.tensor([0])
        adapter._apply_root_velocity_impulse(lin_vel, ang_vel, env_ids)
        art.set_velocities.assert_called_once()

    def test_physics_step_calls_world_step_decimation_times(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation()
        world = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = world
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter.decimation = 4
        adapter.headless = True
        adapter._common_actions = torch.zeros((1, 69))
        # Mock _apply_control to avoid base class dependencies
        adapter._apply_control = MagicMock()

        adapter._physics_step()

        assert world.step.call_count == 4
        assert adapter._apply_control.call_count == 4

    def test_close_calls_super_and_simulation_app(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        sim_app = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._simulation_app = sim_app
        adapter._simulation_running = True

        adapter.close()

        sim_app.close.assert_called_once()
        assert adapter._simulation_running is False
```

- [ ] **Step 6: Implement state setter, control, and physics step**

Replace the `NotImplementedError` stubs for:
- `_set_simulator_env_state`: write root state via `articulation.set_world_poses()`, write joint state via `articulation.set_joint_positions()` + `articulation.set_joint_velocities()`, write joint targets via `articulation.set_joint_position_targets()`.
- `_apply_simulator_pd_targets`: call `articulation.apply_action(ArticulationAction(joint_positions=targets))`.
- `_apply_simulator_torques`: call `articulation.apply_action(ArticulationAction(joint_efforts=torques))`.
- `_apply_root_velocity_impulse`: set root velocity via `articulation.set_velocities()`.
- `_physics_step`: loop `self.decimation` times — call inherited `_apply_control()`, then `self._world.step(render=False)`. After the loop, optionally call `self._world.render()` if not headless.

**Key for `_physics_step`:** Unlike IsaacLab which has `_scene.write_data_to_sim()` / `_scene.update()`, `omni.isaac.core` handles this implicitly through `world.step()`. The articulation reads/writes are synced automatically.

- [ ] **Step 7: Run all tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/hymotion_isaacsim/isaacsim_simulator.py tests/test_isaacsim_simulator.py
git commit -m "feat: implement state read/write, control, and physics stepping in IsaacSimSimulator"
```

---

### Task 4: Simulator Adapter — Contact Sensors

**Files:**
- Modify: `src/hymotion_isaacsim/isaacsim_simulator.py`
- Modify: `tests/test_isaacsim_simulator.py`

**Reference:** `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/isaaclab/simulator.py` — see how `_get_simulator_bodies_contact_buf` is implemented with Isaac Lab's contact sensors.

This task implements contact sensor setup and the `_get_simulator_bodies_contact_buf` method.

- [ ] **Step 1: Write failing test**

Add to `tests/test_isaacsim_simulator.py`:
```python
class TestContactSensors:
    def test_get_simulator_bodies_contact_buf_returns_correct_shape(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        num_bodies = 24  # SMPL body count
        art = _make_fake_articulation()

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter._num_bodies = num_bodies
        adapter._contact_sensors = {}  # will be populated during scene setup

        buf = adapter._get_simulator_bodies_contact_buf()
        assert buf.shape == (1, num_bodies, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestContactSensors -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement contact sensor support**

In `isaacsim_simulator.py`:
- Add a `setup_contact_sensors(body_prim_paths: list[str])` method that creates an `omni.isaac.sensor.ContactSensor` for each humanoid body prim path and stores them in `self._contact_sensors`.
- Implement `_get_simulator_bodies_contact_buf`: iterate over contact sensors, read force data from each, stack into `(num_envs, num_bodies, 3)` tensor. If no contact sensors are set up, return zeros.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestContactSensors -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hymotion_isaacsim/isaacsim_simulator.py tests/test_isaacsim_simulator.py
git commit -m "feat: add contact sensor support to IsaacSimSimulator"
```

---

### Task 5: Simulator Adapter — Viewport Capture and Markers

**Files:**
- Modify: `src/hymotion_isaacsim/isaacsim_simulator.py`
- Modify: `tests/test_isaacsim_simulator.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_isaacsim_simulator.py`:
```python
class TestViewportAndMarkers:
    def test_write_viewport_to_file_calls_capture_api(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._viewport_api = MagicMock()

        # Should not raise NotImplementedError
        adapter._write_viewport_to_file("/tmp/test_frame.png")
        # Verify the capture API was actually called (exact method depends on Isaac Sim version)
        assert len(adapter._viewport_api.method_calls) > 0, "Expected viewport API to be called"

    def test_update_simulator_markers_does_not_raise(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        # Should be a no-op, not raise
        adapter._update_simulator_markers(None)
        adapter._update_simulator_markers({})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestViewportAndMarkers -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement viewport capture and markers**

In `isaacsim_simulator.py`:
- `_write_viewport_to_file`: use `omni.kit.viewport.utility` to get the active viewport, then call the viewport capture API to save a frame to the given path. The exact API depends on Isaac Sim version — check for `capture_viewport_to_file()` or the `ViewportCaptureDelegate` pattern.
- `_update_simulator_markers`: implement as no-op for now. Visualization markers are optional for the initial pipeline. If markers_state is None or empty, return immediately.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/test_isaacsim_simulator.py::TestViewportAndMarkers -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hymotion_isaacsim/isaacsim_simulator.py tests/test_isaacsim_simulator.py
git commit -m "feat: add viewport capture and marker stubs to IsaacSimSimulator"
```

---

### Task 6: Entry Script — Standalone Mode

**Files:**
- Create: `scripts/run_custom_scene.py`

**Reference files:**
- `scripts/smoke_run_motion.py` — existing entry script pattern
- `src/hymotion_isaacsim/checkpoint.py` — `load_tracker_assets()`
- `src/hymotion_isaacsim/protomotions_path.py` — `ensure_protomotions_importable()`

This task creates the entry script with standalone mode only (no `--motion-file`). ProtoMotions mode is added in Task 7.

- [ ] **Step 1: Implement standalone entry script**

`scripts/run_custom_scene.py`:
```python
#!/usr/bin/env python3
"""Run a custom Isaac Sim scene with optional ProtoMotions humanoid control.

Standalone mode (no --motion-file):
  Humanoid stands in rest pose in a scene with static objects.

ProtoMotions mode (--motion-file provided):
  ProtoMotions agent controls the humanoid, tracking the given motion file.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a custom Isaac Sim scene with optional ProtoMotions control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ProtoMotions tracker checkpoint (needed for humanoid asset path)",
    )
    parser.add_argument(
        "--motion-file", type=str, default="",
        help="Path to .motion file. If omitted, runs standalone (rest pose).",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument(
        "--video-output", type=str, default="",
        help="Output MP4 path. If omitted, no video saved.",
    )
    return parser.parse_args()


def build_scene(checkpoint_path: str, headless: bool):
    """Create the Isaac Sim world with ground plane, humanoid, and static objects.

    Returns (simulation_app, world, articulation, tracker_assets).
    """
    from hymotion_isaacsim.protomotions_path import ensure_protomotions_importable
    from hymotion_isaacsim.checkpoint import load_tracker_assets

    ensure_protomotions_importable()
    tracker_assets = load_tracker_assets(checkpoint_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import GroundPlane
    from hymotion_isaacsim.custom_scene import populate_scene

    # Physics dt from checkpoint config
    fps = getattr(tracker_assets.simulator_config.sim, "fps", 60)
    world = World(physics_dt=1.0 / fps, rendering_dt=1.0 / fps)

    # Ground plane
    world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=100.0))

    # Humanoid USD
    asset_root = tracker_assets.robot_config.asset.asset_root
    usd_file = tracker_assets.robot_config.asset.usd_asset_file_name
    humanoid_usd_path = str(Path(asset_root) / usd_file)
    add_reference_to_stage(humanoid_usd_path, "/World/Humanoid")
    articulation = world.scene.add(
        Articulation(prim_path="/World/Humanoid", name="humanoid")
    )

    # Static objects
    populate_scene(world)

    world.reset()

    return simulation_app, world, articulation, tracker_assets


def run_standalone(world, simulation_app, headless: bool):
    """Run the scene with the humanoid in rest pose."""
    print("Running standalone mode (humanoid in rest pose). Press Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            world.step(render=not headless)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


def main():
    args = parse_args()
    simulation_app, world, articulation, tracker_assets = build_scene(
        args.checkpoint, args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        # ProtoMotions mode — Task 7
        raise NotImplementedError("ProtoMotions mode not yet implemented")


if __name__ == "__main__":
    main()
```

**Implementation notes:**
- The exact Isaac Sim import path may be `from isaacsim import SimulationApp` or `from omni.isaac.kit import SimulationApp` depending on the installed version. Check which is available.
- `add_reference_to_stage` loads a USD file as a reference at the given prim path. The humanoid USD path is constructed from `robot_config.asset.asset_root` + `robot_config.asset.usd_asset_file_name`.
- The `Articulation` wrapper must be added to `world.scene` so that `world.reset()` initializes it for physics.

- [ ] **Step 2: Manual smoke test — standalone mode**

Run (requires Isaac Sim + VNC display):
```bash
cd /home/lyuxinghe/code/hymotion_isaacsim
python scripts/run_custom_scene.py --checkpoint <path-to-checkpoint>
```

Expected: Isaac Sim window shows a flat ground plane with 3 colored objects and the SMPL humanoid standing in rest pose. No Isaac Lab environment hierarchy.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_custom_scene.py
git commit -m "feat: add run_custom_scene.py entry script with standalone mode"
```

---

### Task 7: Entry Script — ProtoMotions Mode

**Files:**
- Modify: `scripts/run_custom_scene.py`

**Reference:**
- `src/hymotion_isaacsim/protomotions_runtime.py` — existing ProtoMotions integration (lines 41–158 for `build_standalone_runner`, lines 160–244 for `run_standalone_motion`)
- `/home/lyuxinghe/code/ProtoMotions/protomotions/components/scene_lib.py` — `SceneLib.empty()` factory
- `/home/lyuxinghe/code/ProtoMotions/protomotions/components/motion_lib.py` — `MotionLib` constructor

- [ ] **Step 1: Implement ProtoMotions mode**

Replace the `raise NotImplementedError` in `main()` with the full ProtoMotions integration:

```python
def run_protomotions(
    world,
    articulation,
    simulation_app,
    tracker_assets,
    motion_file: str,
    checkpoint_path: str,
    headless: bool,
    video_output: str | None,
):
    """Run ProtoMotions agent to control the humanoid."""
    from copy import deepcopy
    from dataclasses import asdict

    from lightning.fabric import Fabric
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    from protomotions.envs.managers import base_manager as base_manager_module

    from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
    from hymotion_isaacsim.motion_file import load_motion_metadata
    from hymotion_isaacsim.recording import compile_video, frame_path_for_step

    # Disable torch.compile warmup
    base_manager_module.TORCH_COMPILE_AVAILABLE = False

    # Fabric (single device, no DDP)
    fabric = Fabric(
        **asdict(
            FabricConfig(
                devices=1, num_nodes=1, strategy="auto", loggers=[], callbacks=[],
            )
        )
    )
    fabric.launch()

    # Configs from checkpoint
    robot_config = deepcopy(tracker_assets.robot_config)
    simulator_config = deepcopy(tracker_assets.simulator_config)
    env_config = deepcopy(tracker_assets.env_config)
    agent_config = deepcopy(tracker_assets.agent_config)
    motion_lib_config = deepcopy(tracker_assets.motion_lib_config)

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    simulator_config.num_envs = 1
    simulator_config.headless = headless
    motion_lib_config.motion_file = str(Path(motion_file).resolve())

    # Motion metadata for step count
    motion_metadata = load_motion_metadata(motion_file)
    max_steps = int(motion_metadata.duration_seconds * 30)

    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = max(env_config.max_episode_length, max_steps + 100)

    # Empty scene lib, motion lib from file
    scene_lib = SceneLib.empty(num_envs=1, device=str(fabric.device))
    motion_lib = MotionLib(motion_lib_config, device=str(fabric.device))

    # Our custom simulator adapter
    simulator = IsaacSimSimulator(
        world=world,
        articulation=articulation,
        simulation_app=simulation_app,
        config=simulator_config,
        robot_config=robot_config,
        scene_lib=scene_lib,
        device=fabric.device,
        terrain=None,
    )
    simulator._initialize_with_markers({})

    # Env
    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=None,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Agent
    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=Path(checkpoint_path).resolve().parent,
    )
    agent.setup()
    agent.load(str(Path(checkpoint_path).resolve()), load_env=False)

    # Video setup
    video_path = Path(video_output) if video_output else None
    frames_dir = None
    if video_path:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = video_path.with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Inference loop
    agent.eval()
    done_indices = None
    try:
        for step in range(max_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            obs, rewards, dones, terminated, extras = env.step(actions)

            if frames_dir is not None:
                frame_path = frame_path_for_step(frames_dir, step)
                simulator._write_viewport_to_file(str(frame_path))

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        if video_path and frames_dir:
            frame_paths = sorted(frames_dir.glob("*.png"))
            compile_video(frame_paths, video_path, fps=30)
            print(f"Video saved to {video_path}")

    finally:
        simulation_app.close()
```

Then update `main()`:
```python
def main():
    args = parse_args()
    simulation_app, world, articulation, tracker_assets = build_scene(
        args.checkpoint, args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        run_protomotions(
            world=world,
            articulation=articulation,
            simulation_app=simulation_app,
            tracker_assets=tracker_assets,
            motion_file=args.motion_file,
            checkpoint_path=args.checkpoint,
            headless=args.headless,
            video_output=args.video_output or None,
        )
```

**Implementation notes:**
- The `Env` constructor may differ from what's shown here. During implementation, read the base `Env.__init__` signature and the specific `EnvClass` from the checkpoint to determine exact kwargs. The existing `protomotions_runtime.py:130-139` is the reference.
- `_initialize_with_markers({})` triggers the two-phase init: `_create_simulation()` (no-op) then `_finalize_setup()` (builds conversion tensors). This must be called before the Env is created.
- Passing `terrain=None` to the Env may require handling. Check if the Env accesses `self.terrain` and whether None is safe. If not, create a minimal terrain stub.

- [ ] **Step 2: Manual smoke test — ProtoMotions mode**

Run (requires Isaac Sim + VNC + checkpoint + motion file):
```bash
cd /home/lyuxinghe/code/hymotion_isaacsim
python scripts/run_custom_scene.py \
  --checkpoint <path-to-checkpoint> \
  --motion-file <path-to-motion-file> \
  --video-output /tmp/custom_scene_test.mp4
```

Expected: Isaac Sim window shows the humanoid performing the motion from the .motion file in a scene with 3 static objects on a flat ground. Video file is saved.

- [ ] **Step 3: Compare with existing pipeline**

Run the same motion through the old pipeline:
```bash
python scripts/smoke_run_motion.py \
  --checkpoint <path-to-checkpoint> \
  --motion-file <path-to-motion-file> \
  --video-output /tmp/old_pipeline_test.mp4
```

Compare the two videos visually. The humanoid motion should be the same; the scene (ground, objects, camera angle) will differ.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_custom_scene.py
git commit -m "feat: add ProtoMotions mode to run_custom_scene.py"
```

---

### Task 8: Integration Validation

**Files:** No new files — this is a validation-only task.

- [ ] **Step 1: Run all unit tests**

```bash
cd /home/lyuxinghe/code/hymotion_isaacsim && python -m pytest tests/ -v
```

Expected: All tests pass, including existing tests (no regressions).

- [ ] **Step 2: Run standalone smoke test**

```bash
python scripts/run_custom_scene.py --checkpoint <path> --headless
```

Expected: Runs without error, exits cleanly.

- [ ] **Step 3: Run ProtoMotions smoke test**

```bash
python scripts/run_custom_scene.py \
  --checkpoint <path> \
  --motion-file <path> \
  --headless \
  --video-output /tmp/integration_test.mp4
```

Expected: Runs without error, produces video file.

- [ ] **Step 4: Verify no Isaac Lab dependencies in custom pipeline**

```bash
grep -r "isaaclab\|isaac_lab\|InteractiveScene" scripts/run_custom_scene.py src/hymotion_isaacsim/custom_scene.py src/hymotion_isaacsim/isaacsim_simulator.py
```

Expected: No matches — the custom pipeline should have zero Isaac Lab imports.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: integration fixes for custom Isaac Sim scene pipeline"
```
