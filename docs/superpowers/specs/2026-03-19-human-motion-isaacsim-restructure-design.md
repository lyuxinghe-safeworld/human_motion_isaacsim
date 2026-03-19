# Human Motion Isaac Sim Restructure Design

## Goal

Rename the local Isaac Sim integration repo and package from the old `hymotion_*` naming to `human_motion_isaacsim`, vendor upstream ProtoMotions as a git submodule under `third_party/ProtoMotions`, align install/docs with the actual Python `venv` workflow, and keep the custom-scene inference path working against a `.motion` input and generated MP4 output.

## Scope

This design covers:

- repo identity and local package rename
- ProtoMotions submodule integration
- runtime path-resolution updates
- install and README updates
- verification through tests and one real `run_custom_scene.py` video generation

This design does not refactor ProtoMotions itself beyond consuming it from a new in-repo location.

## Current Constraints

1. The local package lives under `src/hymotion_isaacsim` and all imports, tests, and scripts use that name.
2. ProtoMotions is discovered from an external checkout through `PROTOMOTIONS_ROOT`, `PROTO_MOTIONS_ROOT`, or `~/code/ProtoMotions`.
3. The docs mention `uv`, but the desired project workflow is Python `venv`.
4. The user wants the checkout itself renamed from `/home/lyuxinghe/code/protomotions_isaacsim` to `/home/lyuxinghe/code/human_motion_isaacsim`.

## Chosen Approach

### 1. Repo and package naming

- Rename the checkout directory to `/home/lyuxinghe/code/human_motion_isaacsim`.
- Rename the Python package directory from `src/hymotion_isaacsim` to `src/human_motion_isaacsim`.
- Rename the Python distribution in `pyproject.toml` to `human-motion-isaacsim`.
- Rewrite imports in package code, tests, and scripts to `human_motion_isaacsim`.

### 2. ProtoMotions submodule layout

- Add `third_party/ProtoMotions` as a git submodule.
- Use `git@github.com:NVlabs/ProtoMotions.git` as the submodule URL.
- Treat `third_party/ProtoMotions` as the default in-repo source for:
  - Python imports
  - requirements installation
  - checkpoint examples and pretrained asset paths

### 3. Runtime path resolution

Update ProtoMotions discovery so the lookup order is:

1. explicit `PROTOMOTIONS_ROOT`
2. explicit `PROTO_MOTIONS_ROOT`
3. repo-local `third_party/ProtoMotions`
4. legacy fallback `~/code/ProtoMotions`

This keeps backward compatibility for users with an external checkout while making the submodule the normal path.

### 4. Installation model

Switch the repo docs and bootstrap script to a Python `venv` workflow:

- create `env/.venv` with `python3.11 -m venv`
- install pinned dependencies with `env/.venv/bin/pip`
- install ProtoMotions requirements from `third_party/ProtoMotions/requirements_isaaclab.txt`
- install this repo editable

The install script should fail clearly if `third_party/ProtoMotions` is missing and no explicit ProtoMotions override path is provided.

### 5. Documentation

Update the top-level README and `env/README.md` to reflect:

- the new repo path
- the new package name
- the submodule-backed default layout
- Python `venv` instead of `uv`
- checkpoint and example paths rooted at `third_party/ProtoMotions`

### 6. Verification

Verification should include:

- unit tests updated for the new import path
- environment install in the renamed checkout
- a real `scripts/run_custom_scene.py` invocation producing an MP4 from:
  `/home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion`

## Risk Handling

### Git and filesystem risk

Renaming the checkout directory changes the working path mid-task. To reduce risk:

- finish local doc writes before the move
- rename the directory once
- switch all subsequent commands to the new absolute path

### Submodule usability risk

Submodules are commonly cloned incorrectly. The docs should explicitly use:

```bash
git clone --recurse-submodules ...
```

and explain:

```bash
git submodule update --init --recursive
```

for existing clones.

### Compatibility risk

Keeping `PROTOMOTIONS_ROOT` and `PROTO_MOTIONS_ROOT` overrides avoids breaking existing local setups that still point to an external checkout.
