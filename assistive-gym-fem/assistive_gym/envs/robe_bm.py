import os
import pickle
import time

import cv2
import numpy as np
import pybullet as p

# since there is no gnn_testing_envs file
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from .bu_gnn_util import *
from .env import AssistiveEnv
from .reward_functions import (
    aggregate_reward_field,
    apply_no_grasp_penalty,
    build_cloth_state,
    compute_body_contact_masks,
    compute_reward_field,
)

human_controllable_joint_indices = []

#! max_episode_steps=200 CHECK THAT THIS IS STILL OKAY MOVING FORWARD

class RobustBodyExposureEnv(AssistiveEnv):
    def __init__(self):
        obs_robot_len = 28
        self.single_ppo_model = False # will randomize target limb

        self.ppo = False
        if self.single_ppo_model:
            obs_robot_len = 28 + len(all_possible_target_limbs)
        
        super(RobustBodyExposureEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='bedding_manipulation', obs_robot_len=obs_robot_len, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)
        self.use_mesh = False
        
        self.take_pictures = False
        self.rendering = False
        self.target_limb_code = 5 if not self.single_ppo_model else None # randomize in reset function, None acts as a placeholder
        self.fixed_pose = False

        if self.ppo:
            self.collect_data = False
            self.blanket_pose_var = False
            self.high_pose_var = False
            self.body_shape_var = False
            self.body_shape = None if self.body_shape_var == True else np.zeros((1, 10))
            self.gender = 'random' if self.body_shape_var == True else 'female'
        else:
            self.collect_data = None
            self.blanket_pose_var = None
            self.high_pose_var = None
            self.body_shape_var = None

        self.naive = False
        self.clip = True #! Turn off for cma-data collect

        self.human_no_occlusion_RGB = None
        self.human_no_occlusion_depth = None
        self.point_cloud_initial = None
        self.point_cloud_final = None
        self.point_cloud_depth_img = None


        self.seed_val = 100
        self.cma_sim_dyn = False

        # Reward-field configuration
        self.reward_coverage_mode = "expose"
        self.reward_diffusion_iters = 1
        self.reward_diffusion_alpha = 0.5
        self.reward_normalize_wrinkle = True
        self.reward_no_grasp_penalty = 0.25
        self._last_action_on_cloth = True
        # Snapshot settings for temporal reward visualisations
        self.reward_snapshot_stride = 5  # capture every N internal sim steps per phase
        self.reward_snapshot_max = 180   # safety cap to avoid unbounded logging
    
    def set_seed_val(self, seed = 1001):
        # if seed != self.seed_val:
        #     self.seed_val = seed
        self.seed_val = seed

    def set_target_limb_code(self, code):
        self.target_limb_code = code
    
    def set_env_variations(self, collect_data, blanket_pose_var, high_pose_var, body_shape_var):
        self.collect_data = collect_data
        self.blanket_pose_var = blanket_pose_var
        self.high_pose_var = high_pose_var
        self.body_shape_var = body_shape_var

        self.body_shape = None if self.body_shape_var == True else np.zeros((1, 10))
        self.gender = 'random' if self.body_shape_var == True else 'female'

    def get_human_body_info(self):
        return self.human_creation.body_info if self.body_shape_var else None
    
        
    def step(self, action):
        self.execute_action = True
        obs = self._get_obs()

        if self.ppo:
            action = remap_action_ppo(action, remap_ranges=[[0, 1], [-0.5, 1], [0, 1], [-1, 1]])

        if self.rendering:
            print(obs)
            print(action)
            print(self.target_limb_code)

        action = scale_action(action) if not self.naive else scale_action(action, scale=[1, 1])

        command_grasp = np.array(action[0:2], dtype=np.float32)
        command_release = np.array(action[2:4], dtype=np.float32)

        grasp_loc = command_grasp.copy()
        release_loc = command_release.copy()

        reward_mode = self.reward_coverage_mode.lower()
        task_bit = 1.0 if reward_mode == "recover" else 0.0
        collecting_data = bool(self.collect_data)

        data_i = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        cloth_initial_full = np.asarray(data_i[1], dtype=np.float32)
        cloth_edge_index_full = None
        all_body_points = None
        trajectory_snapshots = []
        trajectory_params = None
        snapshot_enabled = not collecting_data

        if snapshot_enabled:
            snapshot_stride = max(1, int(getattr(self, "reward_snapshot_stride", 5) or 1))
            snapshot_max_setting = getattr(self, "reward_snapshot_max", 0)
            snapshot_max = int(snapshot_max_setting) if snapshot_max_setting else 0
            # Ensure camera is configured so we can capture per-substep frames
            try:
                self.setup_camera_rpy(
                    camera_target=[0, 0, 0.305 + 2.101],
                    distance=0.01,
                    rpy=[0, -90, 180],
                    fov=60,
                    camera_width=468 // 2,
                    camera_height=398,
                )
            except Exception:
                pass
            try:
                cloth_edge_index_full = get_edge_connectivity(
                    cloth_initial_full, edge_threshold=0.06, cloth_dim=3
                ).cpu().numpy().astype(np.int64)
            except Exception:
                cloth_edge_index_full = None
            human_pose = np.reshape(self.human_pose, (-1, 2))
            all_body_points = get_body_points_from_obs(
                human_pose,
                target_limb_code=self.target_limb_code,
                body_info=self.get_human_body_info(),
            ).astype(np.float32)

            def _record_snapshot(phase, substep, positions=None, note=None):
                if snapshot_max and len(trajectory_snapshots) >= snapshot_max:
                    return
                if positions is None:
                    positions = np.asarray(
                        p.getMeshData(
                            self.blanket,
                            -1,
                            flags=p.MESH_DATA_SIMULATION_MESH,
                            physicsClientId=self.id,
                        )[1],
                        dtype=np.float32,
                    )
                else:
                    positions = np.asarray(positions, dtype=np.float32)

                cloth_state_full = build_cloth_state(cloth_initial_full, positions)
                cloth_xy_full = cloth_state_full.positions[:, :2]
                (
                    body_mask_full,
                    target_mask_full,
                    nontarget_mask_full,
                    head_mask_full,
                ) = compute_body_contact_masks(cloth_xy_full, all_body_points)
                goal_contact_mask_full = -np.ones_like(target_mask_full, dtype=np.float32)
                if reward_mode == "recover":
                    goal_contact_mask_full[target_mask_full > 0.0] = 1.0
                elif reward_mode == "expose":
                    goal_contact_mask_full[target_mask_full > 0.0] = 0.0

                diffusion_edges = cloth_edge_index_full if cloth_edge_index_full is not None else None
                diffusion_iters = self.reward_diffusion_iters if diffusion_edges is not None else 0

                reward_field_full, term_components = compute_reward_field(
                    cloth_state_full,
                    body_mask_full,
                    target_mask_full,
                    nontarget_mask=nontarget_mask_full,
                    head_mask=head_mask_full,
                    goal_contact_mask=goal_contact_mask_full,
                    coverage_mode=self.reward_coverage_mode,
                    normalize_wrinkle=self.reward_normalize_wrinkle,
                    diffusion_edge_index=diffusion_edges,
                    diffusion_iters=diffusion_iters,
                    diffusion_alpha=self.reward_diffusion_alpha,
                    return_terms=True,
                    nontarget_mode="robe_points",
                    body_points=all_body_points,
                )

                feature_columns_full = [
                    cloth_state_full.positions,
                    cloth_state_full.displacement,
                    cloth_state_full.tension[:, None],
                    body_mask_full[:, None],
                    target_mask_full[:, None],
                    nontarget_mask_full[:, None],
                    head_mask_full[:, None],
                    goal_contact_mask_full[:, None],
                    np.full((cloth_state_full.positions.shape[0], 1), task_bit, dtype=np.float32),
                ]
                cloth_features_full = np.concatenate(feature_columns_full, axis=1)

                weighted_terms = term_components.get("weighted", {})
                term_weighted = {name: np.asarray(arr, dtype=np.float32) for name, arr in weighted_terms.items()}
                if "combined_pre_diff" in term_components:
                    term_weighted["total_pre_diff"] = np.asarray(term_components["combined_pre_diff"], dtype=np.float32)
                if "combined_post_diff" in term_components:
                    term_weighted["total_post_diff"] = np.asarray(term_components["combined_post_diff"], dtype=np.float32)

                raw_terms = term_components.get("raw", {})
                term_raw = {name: np.asarray(arr, dtype=np.float32) for name, arr in raw_terms.items()}

                scalar_contribs = {}
                for name, arr in term_weighted.items():
                    if arr.size:
                        scalar_contribs[name] = float(np.nanmean(arr))
                    else:
                        scalar_contribs[name] = 0.0

                try:
                    sphere_pos = self.sphere_ee.get_base_pos_orient()[0]
                    sphere_pos_list = np.asarray(sphere_pos, dtype=np.float32).tolist()
                except Exception:
                    sphere_pos_list = None

                snap = {
                        "phase": phase,
                        "substep": int(substep),
                        "note": note,
                        "sphere_pos": sphere_pos_list,
                        "reward_field": np.asarray(reward_field_full, dtype=np.float32),
                        "cloth_features": cloth_features_full.astype(np.float32),
                        "term_weighted": term_weighted,
                        "term_raw": term_raw,
                        "scalar_contribs": scalar_contribs,
                    }
                # Optionally capture a camera frame for this substep
                try:
                    img, depth_img = self.get_camera_image_depth()
                    snap["rgb"] = img
                    snap["depth"] = depth_img
                except Exception:
                    pass

                trajectory_snapshots.append(snap)

            def _maybe_record(phase, substep, force=False, note=None):
                if not force and (substep % snapshot_stride != 0):
                    return
                _record_snapshot(phase, substep, note=note)

            _record_snapshot("initial", 0, positions=cloth_initial_full, note="pre-action")

            trajectory_params = {
                "snapshot_stride": snapshot_stride,
                "snapshot_max": snapshot_max,
                "action_command_grasp": command_grasp.tolist(),
                "action_command_release": command_release.tolist(),
                "action_executed": bool(self.execute_action),
            }
        else:
            snapshot_stride = None
            snapshot_max = None

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist, is_on_cloth = check_grasp_on_cloth(action, np.array(data_i[1]))
        self._last_action_on_cloth = bool(is_on_cloth)
        # * if no points on the blanket are within 2.8 cm of the grasp location, exit (if collecting data) or proceed without executing the action (in all other conditions)
        if not self._last_action_on_cloth:
            if self.collect_data:
                return obs, 0, False, {} # for data collect
            elif self.clip:
                self.execute_action = False

        if self.execute_action:
            anchor_idx = np.argpartition(np.array(dist), 4)[:4]

            # * update grasp_loc var with the location of the central anchor point on the cloth
            grasp_loc = np.array(data_i[1][anchor_idx[0]][0:2])

            # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
            self.sphere_ee.set_base_pos_orient(data_i[1][anchor_idx[0]], np.array([0,0,0]))
            constraint_ids = []
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
            for i in anchor_idx[1:]:
                pos_diff = np.array(data_i[1][i]) - np.array(data_i[1][anchor_idx[0]])
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))

            if snapshot_enabled:
                _record_snapshot("anchor", 0, note="post-anchor")

            # * move sphere up by some delta z
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            delta_z = 0.4                           # distance to move up (with respect to the top of the bed)
            bed_height = 0.58                       # height of the bed
            final_z = delta_z + bed_height          # global goal z position
            lift_iter = 0
            while current_pos[2] <= final_z:
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                lift_iter += 1
                if snapshot_enabled:
                    _maybe_record("lift", lift_iter)
            if snapshot_enabled:
                _maybe_record("lift", lift_iter, force=True, note="post-lift")

            # * move sphere to the release location, release the blanket
            travel_dist = release_loc - grasp_loc

            # * determine delta x and y, make sure it is, at max, close to 0.005
            num_steps = np.abs(travel_dist//0.005).max()
            delta_x, delta_y = travel_dist/num_steps

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            translate_iter = 0
            for _ in range(int(num_steps)):
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, 0]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                translate_iter += 1
                if snapshot_enabled:
                    _maybe_record("translate", translate_iter)
            if snapshot_enabled:
                _maybe_record("translate", translate_iter, force=True, note="end-translate")

            # * continue stepping simulation to allow the cloth to settle before release
            pre_release_iter = 0
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.id)
                pre_release_iter += 1
                if snapshot_enabled:
                    _maybe_record("pre_release_settle", pre_release_iter)
            if snapshot_enabled:
                _maybe_record("pre_release_settle", pre_release_iter, force=True, note="before-release")

            # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
            for i in constraint_ids:
                p.removeConstraint(i, physicsClientId=self.id)
            if snapshot_enabled:
                _record_snapshot("release", 0, note="release-constraints")

            post_release_iter = 0
            for _ in range(50):
                p.stepSimulation(physicsClientId=self.id)
                post_release_iter += 1
                if snapshot_enabled:
                    _maybe_record("post_release_settle", post_release_iter)
            if snapshot_enabled:
                _maybe_record("post_release_settle", post_release_iter, force=True, note="post-settle")

        # * get points on the blanket, final state of the cloth
        data_f = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        if snapshot_enabled:
            _record_snapshot("final", 0, positions=np.asarray(data_f[1], dtype=np.float32), note="post-action")

        if trajectory_params is not None:
            trajectory_params["action_grasp_world"] = grasp_loc.tolist()
            trajectory_params["action_release_world"] = release_loc.tolist()
            trajectory_params["action_delta_world"] = (release_loc - grasp_loc).tolist()

        reward_field = np.array([], dtype=np.float32)
        scalar_reward = 0.0
        legacy_reward = None
        cloth_initial_subsample = cloth_final_subsample = -1  # none for data collection
        info = {}
        if not collecting_data:
            # TODO - REPLACE REWARD CALCULATION HERE WTIH FUNCTION FROM GNN_UTIL
            cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(data_i[1], data_f[1])
            cloth_initial_2D = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
            cloth_final_2D = np.delete(np.array(cloth_final_subsample), 2, axis = 1)
            if all_body_points is None:
                human_pose = np.reshape(self.human_pose, (-1,2))
                all_body_points = get_body_points_from_obs(
                    human_pose,
                    target_limb_code=self.target_limb_code,
                    body_info=self.get_human_body_info(),
                ).astype(np.float32)
            legacy_reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)

            cloth_initial_arr = np.asarray(cloth_initial_subsample, dtype=np.float32)
            cloth_final_arr = np.asarray(cloth_final_subsample, dtype=np.float32)
            cloth_state = build_cloth_state(cloth_initial_arr, cloth_final_arr)
            cloth_xy = cloth_state.positions[:, :2]
            edge_index = get_edge_connectivity(
                cloth_state.positions, edge_threshold=0.06, cloth_dim=3
            )
            cloth_edge_index = edge_index.cpu().numpy().astype(np.int64)
            (
                body_mask,
                target_mask,
                nontarget_mask,
                head_mask,
            ) = compute_body_contact_masks(cloth_xy, all_body_points)
            # Build goal contact mask for unified objective (only defined on target vertices)
            # Use -1 sentinel for undefined (non-target) so compute_reward_field ignores them.
            goal_contact_mask = -np.ones_like(target_mask, dtype=np.float32)
            if reward_mode == "recover":
                goal_contact_mask[target_mask > 0.0] = 1.0
            elif reward_mode == "expose":
                goal_contact_mask[target_mask > 0.0] = 0.0
            else:
                # 'none' → leave all undefined
                pass

            reward_field = compute_reward_field(
                cloth_state,
                body_mask,
                target_mask,
                nontarget_mask=nontarget_mask,
                head_mask=head_mask,
                goal_contact_mask=goal_contact_mask,
                coverage_mode=self.reward_coverage_mode,
                normalize_wrinkle=self.reward_normalize_wrinkle,
                diffusion_edge_index=cloth_edge_index,
                diffusion_iters=self.reward_diffusion_iters,
                diffusion_alpha=self.reward_diffusion_alpha,
                nontarget_mode="robe_points",
                body_points=all_body_points,
            )
            scalar_reward = aggregate_reward_field(reward_field)

            displacement = cloth_state.displacement
            tension = cloth_state.tension[:, None]
            feature_columns = [
                cloth_state.positions,
                displacement,
                tension,
                body_mask[:, None],
                target_mask[:, None],
                nontarget_mask[:, None],
                head_mask[:, None],
            ]
            # Append goal mask (where defined) and a global task bit for conditioning
            feature_columns.append(goal_contact_mask[:, None])
            feature_columns.append(np.full((cloth_state.positions.shape[0], 1), task_bit, dtype=np.float32))
            cloth_features = np.concatenate(feature_columns, axis=1)

            # Additionally compute an initial-state reward field and features to visualize change
            try:
                # Build initial-only cloth state (positions at initial, zero displacement)
                cloth_state_init = build_cloth_state(cloth_initial_arr, cloth_initial_arr)
                cloth_xy_init = cloth_state_init.positions[:, :2]
                # Edge connectivity on initial cloth positions
                cloth_edge_index_init = get_edge_connectivity(
                    cloth_state_init.positions, edge_threshold=0.06, cloth_dim=3
                ).cpu().numpy().astype(np.int64)
                # Body contact masks at initial state
                (
                    body_mask_init,
                    target_mask_init,
                    nontarget_mask_init,
                    head_mask_init,
                ) = compute_body_contact_masks(cloth_xy_init, all_body_points)
                # Goal mask (same semantics as final; defined only on target vertices)
                goal_contact_mask_init = -np.ones_like(target_mask_init, dtype=np.float32)
                if self.reward_coverage_mode.lower() == "recover":
                    goal_contact_mask_init[target_mask_init > 0.0] = 1.0
                elif self.reward_coverage_mode.lower() == "expose":
                    goal_contact_mask_init[target_mask_init > 0.0] = 0.0
                # Compute initial reward field (no change of config)
                reward_field_init = compute_reward_field(
                    cloth_state_init,
                    body_mask_init,
                    target_mask_init,
                    nontarget_mask=nontarget_mask_init,
                    head_mask=head_mask_init,
                    goal_contact_mask=goal_contact_mask_init,
                    coverage_mode=self.reward_coverage_mode,
                    normalize_wrinkle=self.reward_normalize_wrinkle,
                    diffusion_edge_index=cloth_edge_index_init,
                    diffusion_iters=self.reward_diffusion_iters,
                    diffusion_alpha=self.reward_diffusion_alpha,
                    nontarget_mode="robe_points",
                    body_points=all_body_points,
                )
                # Build initial features analogous to final
                feature_columns_init = [
                    cloth_state_init.positions,
                    np.zeros_like(displacement),  # zero displacement at initial
                    cloth_state_init.tension[:, None] if hasattr(cloth_state_init, 'tension') else np.zeros((cloth_state_init.positions.shape[0], 1), dtype=np.float32),
                    body_mask_init[:, None],
                    target_mask_init[:, None],
                    nontarget_mask_init[:, None],
                    head_mask_init[:, None],
                    goal_contact_mask_init[:, None],
                    np.full((cloth_state_init.positions.shape[0], 1), task_bit, dtype=np.float32),
                ]
                cloth_features_init = np.concatenate(feature_columns_init, axis=1)
            except Exception:
                reward_field_init = None
                cloth_features_init = None
                cloth_edge_index_init = None

            info = {
                "cloth_initial": data_i,
                "cloth_final": data_f,
                "RBG_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "cloth_initial_subsample": cloth_initial_subsample,
                "cloth_final_subsample": cloth_final_subsample,
                "covered_status_sim": covered_status,
                "target_limb_code":self.target_limb_code,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender,
                "grasp_on_cloth":self.execute_action,
                "reward_field": reward_field.astype(np.float32),
                "reward_field_scalar": float(scalar_reward),
                "legacy_reward": float(legacy_reward),
                "cloth_features": cloth_features.astype(np.float32),
                "cloth_edge_index": cloth_edge_index,
                # Optional initial-state artifacts for visualization
                "reward_field_init": reward_field_init.astype(np.float32) if isinstance(reward_field_init, np.ndarray) else None,
                "cloth_features_init": cloth_features_init.astype(np.float32) if isinstance(cloth_features_init, np.ndarray) else None,
                "cloth_edge_index_init": cloth_edge_index_init,
                }

            if trajectory_snapshots:
                info["reward_field_traj"] = [snap["reward_field"] for snap in trajectory_snapshots]
                info["cloth_features_traj"] = [snap["cloth_features"] for snap in trajectory_snapshots]
                info["term_traj_weighted"] = [snap["term_weighted"] for snap in trajectory_snapshots]
                info["term_traj_raw"] = [snap["term_raw"] for snap in trajectory_snapshots]
                info["term_traj_scalars"] = [snap["scalar_contribs"] for snap in trajectory_snapshots]
                # Optional per-substep images
                try:
                    rgb_list = [snap.get("rgb") for snap in trajectory_snapshots]
                    if any(v is not None for v in rgb_list):
                        info["traj_rgb"] = rgb_list
                except Exception:
                    pass
                try:
                    depth_list = [snap.get("depth") for snap in trajectory_snapshots]
                    if any(v is not None for v in depth_list):
                        info["traj_depth"] = depth_list
                except Exception:
                    pass
                info["traj_metadata"] = [
                    {
                        "phase": snap["phase"],
                        "substep": snap["substep"],
                        "note": snap["note"],
                        "sphere_pos": snap["sphere_pos"],
                        "scalar_contribs": snap["scalar_contribs"],
                    }
                    for snap in trajectory_snapshots
                ]
            if trajectory_params is not None:
                info["traj_params"] = trajectory_params
        else:
            info = {
                "cloth_initial": data_i,
                "cloth_final": data_f,
                "RGB_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "point_cloud_depth_img": self.point_cloud_depth_img,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender
                }
        penalty_value = float(getattr(self, "reward_no_grasp_penalty", 0.0) or 0.0)
        if reward_field.size == 0 and legacy_reward is not None:
            scalar_reward = float(legacy_reward)
        scalar_reward = apply_no_grasp_penalty(scalar_reward, self._last_action_on_cloth, penalty_value)

        if "reward_field" not in info:
            info["reward_field"] = reward_field.astype(np.float32)
        if "reward_field_scalar" not in info:
            info["reward_field_scalar"] = float(scalar_reward)
        if legacy_reward is not None and "legacy_reward" not in info:
            info["legacy_reward"] = float(legacy_reward)
        info["is_on_cloth"] = bool(self._last_action_on_cloth)
        if penalty_value > 0.0:
            info["no_grasp_penalty"] = penalty_value if not self._last_action_on_cloth else 0.0
        reward = scalar_reward
        self.iteration += 1
        done = self.iteration >= 1

        # return 0, 0, 1, {}
        return obs, reward, done, info
    
    def get_cloth_state(self):
        return p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]

    def _get_obs(self, agent=None):

        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]

        pose = []
        for limb in self.human.obs_limbs:
            pos, orient = self.human.get_pos_orient(limb)
            # print("pose", limb, pos, orient)
            pos2D = pos[0:2]
            # yaw = p.getEulerFromQuaternion(orient)[-1]
            # pose.append(np.concatenate((pos2D, np.array([yaw])), axis=0))
            pose.append(pos2D)
        pose = np.concatenate(pose, axis=0)
        self.human_pose = pose

        if self.collect_data:

            output = [None]*28
            all_joint_angles = self.human.get_joint_angles(self.human.all_joint_indices)
            all_pos_orient = [self.human.get_pos_orient(limb) for limb in self.human.all_body_parts]
            output[0], output[1], output[2] = pose, all_joint_angles, all_pos_orient
            return output

            
        if self.single_ppo_model:
            one_hot_target_limb = [0]*len(self.human.all_possible_target_limbs)
            one_hot_target_limb[self.target_limb_code] = 1
            pose = np.concatenate([one_hot_target_limb, pose], axis=0)
        return np.float32(pose)

    def reset(self):
        if self.cma_sim_dyn:
            self.seed(self.seed_val)

        super(RobustBodyExposureEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False, gender=self.gender, human_impairment='none', furniture_type='hospital_bed', body_shape=self.body_shape)
        # self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed', body_shape=self.body_shape)

        self.target_limb_code = self.target_limb_code if not self.single_ppo_model else randomize_target_limbs()

        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        
        # * Setup human in the air, with legs and arms slightly seperated
        joints_positions = [(self.human.j_left_hip_y, -10), (self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])

        if not self.fixed_pose:
            # * Add small variation to the body pose
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            # print(motor_positions)
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.2, 0.2, size=len(motor_indices)))
            if self.high_pose_var:
                self.increase_pose_variation()
            # * Increase friction of joints so human doesn't fail around exessively as they settle
            # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
            self.human.set_whole_body_frictions(spinning_friction=2)

        # * Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        # * step the simulation a few times so that the human has some initial velocity greater than the at rest threshold
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)
        # * continue stepping the simulation until the human joint velocities are under the threshold
        threshold = 1e-2
        settling = True
        numsteps = 0
        while settling:
            settling = False
            for i in self.human.all_joint_indices:
                if np.any(np.abs(self.human.get_velocity(i)) >= threshold):
                    p.stepSimulation(physicsClientId=self.id)
                    numsteps += 1
                    settling = True
                    break
            if numsteps > 400:
                break
        # print("steps to rest:", numsteps)

        # * Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        
        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
    
        # * Setup camera for taking images
        # *     Currently saves color images only to specified directory
        if self.take_pictures or self.collect_data:
            self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=398)
            img, depth = self.get_camera_image_depth()
            self.human_no_occlusion_RGB = img
            self.human_no_occlusion_depth = depth
            depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
            depth = (depth * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
            # filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
            # filename = "test_depth.png"
            # cv2.imwrite(os.path.join('/home/mycroft/git/vBMdev/pose_variation_images/lower_var2', filename), img)
            # cv2.imwrite(filename, depth_colormap)



        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)

        # * can apply some variation in the blanket's initial position, otherwise configure over the person so that they are covered up to the shoulders/neck
        if self.blanket_pose_var:
            delta_y = self.np_random.uniform(-0.25, 0.05)
            delta_x = self.np_random.uniform(-0.02, 0.02)
            deg = 45
            delta_rad = self.np_random.uniform(-np.radians(deg), np.radians(deg)) # * +/- degrees
            p.resetBasePositionAndOrientation(self.blanket, [0+delta_x, 0.2+delta_y, 1.5], self.get_quaternion([np.pi/2.0, 0, 0+delta_rad]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # time.sleep(2)

        # return 0


        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # self.non_target_initially_uncovered(data)
        # self.uncover_nontarget_reward(data)

    
        # * Initialize enviornment variables
        # *     if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
        # self.time = time.time()
        if self.robot is None:
            # * spawn sphere manipulator
            # position = np.array([-0.3, -0.86, 0.8])
            position = np.array([2 ,2, 0]) # move out of the way so it doesn't interfere with the initial depth image
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])
            
            # * initialize env variables
            from gym import spaces
            # * update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)
            # * Define action/obs lengths
            self.action_robot_len = 4
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()
        

        return self._get_obs()
    
    def increase_pose_variation(self):
        '''
        Allow more variation in the knee and elbow angles
          can be some random position within the lower and upper limits of the joint movement (range is made a little smaller than the limits of the joint to prevent angles that are too extreme)
        '''
        for joint in (self.human.j_left_knee, self.human.j_right_knee, self.human.j_left_elbow, self.human.j_right_elbow):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states([joint])
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(self.human.lower_limits[joint]+0.1, self.human.upper_limits[joint]-0.1, 1))

    
