from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from human_motion_isaacsim.checkpoint import TrackerAssets


def _teardown_helper(helper: Any) -> None:
    for method_name in ("teardown", "destroy", "shutdown", "close"):
        method = getattr(helper, method_name, None)
        if callable(method):
            method()
            return


@dataclass(slots=True)
class PackageState:
    model_name: str | None = None
    tracker_assets: TrackerAssets | None = None
    world: Any | None = None
    articulation: Any | None = None
    body_rigid_view: Any | None = None
    headless: bool = True
    reference_markers: bool = False
    simulation_app: Any | None = None
    owned_helpers: list[Any] = field(default_factory=list)

    def teardown(self) -> None:
        first_error: Exception | None = None
        try:
            for helper in self.owned_helpers:
                if helper is self.simulation_app:
                    continue
                try:
                    _teardown_helper(helper)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            self.owned_helpers.clear()
            self.model_name = None
            self.tracker_assets = None
            self.world = None
            self.articulation = None
            self.body_rigid_view = None
            self.headless = True
            self.reference_markers = False

        if first_error is not None:
            raise first_error


PACKAGE_STATE = PackageState()
