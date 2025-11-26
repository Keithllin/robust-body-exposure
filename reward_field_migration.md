# Reward Field Migration Plan (RoBE → `robust-body-exposure-recovering`)

This note explains how to port the unified reward-field stack (and its field-guided policy tooling) from the reference repository `../robust-body-exposure-main` into this branch (`robust-body-exposure-recovering`). It lists every required module, their destinations inside this repo, and the order in which they should be integrated and validated.

## 1. Source → Destination Map

| Component | Source path (reference repo) | Destination inside this repo | Notes |
|-----------|------------------------------|------------------------------|-------|
| `body_info.pkl` | `assistive-gym-fem/assistive_gym/envs/body_info.pkl` | already present at `assistive-gym-fem/assistive_gym/envs/body_info.pkl` | Verify checksum; keep the richer version (with torso radii) if they differ. |
| Reward helpers (`ClothState`, `compute_reward_field`, diffusion, etc.) | `assistive-gym-fem/assistive_gym/envs/reward_functions.py` | add new file `assistive-gym-fem/assistive_gym/envs/reward_functions.py` | Required by both training and visualization scripts. |
| Field-guided heuristic | `assistive-gym-fem/assistive_gym/envs/field_guided_policy.py` | add same path here | Optional but recommended for deterministic baselines / debugging (draws red pick spheres + green arrows). |
| Environment logic with reward-field integration | `assistive-gym-fem/assistive_gym/envs/robe_bm.py` | either (a) copy as `assistive_gym/envs/robe_bm.py`, or (b) merge reward-field sections into the existing `robe_bm_reversible.py` | Choose (a) if you want parity with the mainline RoBE env; choose (b) if this branch must stay “reversible” only. |
| Utility helpers (`get_body_points_from_obs`, `sub_sample_point_clouds`, `get_edge_connectivity`, etc.) | `assistive-gym-fem/assistive_gym/envs/bu_gnn_util.py` | merge missing functions into this repo’s `assistive_gym/envs/bu_gnn_util.py` | This repo already has a trimmed version—confirm every helper used by the reward pipeline exists. |
| Simulation runner that consumes reward fields | `code/run_robe_sim.py` (and friends) | mirror updates in `code/run_robe_sim*.py` here | Needed so CMA/GNN drivers capture `info['reward_field']`, snapshots, etc. |
| Test harness for end-to-end visualization | `scripts/test_field_guided_policy_sim.py` (recent addition) | optional new script under `code/` or `scripts/` | Helps validate the port before touching training scripts. |

## 2. Integration Steps

1. **Sync utilities**
   - Diff the two versions of `assistive_gym/envs/bu_gnn_util.py`. Ensure the following functions match the reference implementation: `get_body_points_from_obs`, `sub_sample_point_clouds`, `get_edge_connectivity`, `randomize_target_limbs`, `scale_action`, `check_grasp_on_cloth`.
   - Confirm `body_info.pkl` is the extended one (8-limb radii + torso metadata). Replace if necessary.

2. **Add reward-field module**
   - Copy `reward_functions.py` verbatim.
   - Update `assistive_gym/envs/__init__.py` to export `compute_reward_field` only if other modules import it via package-relative paths.

3. **Bring in the environment hook**
   - Fast path: copy `robe_bm.py` from the main repo so that `RobustBodyExposureEnv` is available alongside the reversible env already in this branch.
   - Alternate path: open `robe_bm_reversible.py`, locate its `step()` method, and transplant the reward-field block from the main repo (`cloth_state = build_cloth_state(...)` through the `info[...]` assignments). Ensure snapshot helpers (`_record_snapshot`, `_maybe_record`) and attributes (`reward_snapshot_stride`, etc.) are defined in `__init__`.

4. **Field-guided policy (optional but recommended)**
   - Copy `field_guided_policy.py` so you can debug reward fields via the deterministic “force-field” heuristic.
   - If you reuse the same script, remember to update its import root (this repo lives at `/home/keithlin/RCHI/robust-body-exposure-recovering`).

5. **Update simulation drivers**
   - In `code/run_robe_sim*.py`, switch to the new env import (`from assistive_gym.envs.robe_bm import RobustBodyExposureEnv`) or equivalent.
   - Propagate any CLI arguments (`--graph-config`, `--env-var`) exactly as in the reference repo so reward snapshots and `info['reward_field']` are saved when running CMA/GNN loops.

6. **(Optional) Add a minimal validation script**
   - Drop `scripts/test_field_guided_policy_sim.py` (or place it under `code/`) to spin up the env, compute a reward field, and call `compute_field_guided_action(debug_mode=True)`. This script is invaluable for confirming the “red sphere + green arrow” visualization works post-port.

## 3. Validation Checklist

After copying the files, run the following quick tests from the new repo root:

```bash
# 1. Sanity check: instantiate the env and ensure reward fields populate info
python3 code/run_robe_sim.py --model-path <checkpoint> --graph-config 3D --env-var combo_var --max-fevals 2 --num-rollouts 2

# 2. (Optional) Visual smoke test with the field-guided policy
python3 scripts/test_field_guided_policy_sim.py
```

During the first run, inspect the console/logs for:
- `info['reward_field']` present and non-empty.
- `info['reward_field_scalar']` roughly aligned with the older `legacy_reward` values.
- If snapshots are enabled, `reward_field_traj` should contain per-substep arrays.

Once these checks pass, you can safely integrate the new reward field outputs with other “heavy coverage” algorithms inside this branch.
