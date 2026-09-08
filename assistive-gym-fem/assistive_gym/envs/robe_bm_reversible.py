import os
import pickle
import time

import trimesh

import cv2
import numpy as np
import pybullet as p

from .agents.human import Human
from .agents.human_mesh import HumanMesh
from .bu_gnn_util import *
from .env import AssistiveEnv
import pickle, pathlib
from pathlib import Path
import os.path as osp

human_controllable_joint_indices = []
class RobeReversibleEnv(AssistiveEnv):
    def __init__(self):
        obs_robot_len = 28

        super(RobeReversibleEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='bedding_manipulation', obs_robot_len=obs_robot_len, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)

        self.recover = None

        self.singulate_layers = None

        self.mesh = []
        self.mesh_dict = dict()

        # rewards
        self.uncover_reward = 0
        self.recover_reward = 0

        # actions
        self.uncover_action = []
        self.recover_action = []

        # thresholds for effector height
        self.min_threshold = .05
        self.max_threshold = .06
        self.line_threshold = .1
        self.release_threshold = .05
        self.pre_release_steps = 20
        self.post_release_steps = 50
        self.quiet_settle_enabled = False
        self.quiet_settle_speed_threshold = 0.15
        self.quiet_settle_max_steps = 20
        self.uncover_release_gravity_boost_enabled = False
        self.uncover_release_gravity_z = -39.24
        self.default_gravity_z = -9.81

        # cloth states
        self.cloth_initial = []
        self.cloth_intermediate = []
        self.cloth_final = []

        self.anchor_idx = []

        self.points_pos_limb_world = []
        self.render_body_points = True

        self.iteration = 0
        self.take_pictures = True
        self.rendering = False
        self.target_limb_code = None
        self.fixed_pose = False

        self.collect_data = None
        self.blanket_pose_var = None
        self.high_pose_var = None
        self.body_shape_var = None

        self.execute_uncover_action = None
        self.execute_recover_action = None

        self.naive = False
        self.clip = True
        self.show_cloth_vertex_ids = False

        self.human_no_occlusion_RGB = None
        self.human_no_occlusion_depth = None
        self.point_cloud_initial = None
        self.point_cloud_final = None
        self.point_cloud_depth_img = None

        # Optional diagnostics. Disabled by default so normal training/eval
        # payloads do not grow.
        self.record_cloth_trajectory = False
        self.cloth_trajectory_stride = 1
        self.cloth_trajectory_max_frames = 0
        self.cloth_trajectory = []
        self._cloth_trajectory_step = 0

        # seed values
        self.seed_val = None
        self.replay_seed = False

        # uncover release semantics (default matches current always-on lowering)
        self.lower_before_release_uncover = True
        # If set, lower by this absolute distance (m) before release instead of
        # adaptive body-clearance lowering via release_threshold.
        self.uncover_release_lower_delta = None
        self._last_uncover_dynamics = {}
        self._last_reverse_dynamics = {}
        self._last_uncover_grasp_xy = None
        self._last_uncover_release_xy = None
        self._last_uncover_grasp_z = None

    def set_seed_val(self, seed):
        self.seed_val = seed

    def set_target_limb_code(self, code):
        self.target_limb_code = code

    def set_recover(self, recover, replay_seed=True):
        self.recover = recover
        self.replay_seed = bool(replay_seed)

    def set_uncover_release_lowering(self, enabled, release_threshold=None, lower_delta=None):
        self.lower_before_release_uncover = bool(enabled)
        if release_threshold is not None:
            self.release_threshold = float(release_threshold)
        if lower_delta is None:
            self.uncover_release_lower_delta = None
        else:
            self.uncover_release_lower_delta = float(lower_delta)
            if self.uncover_release_lower_delta < 0:
                raise ValueError('uncover release lower_delta must be >= 0')

    def get_last_uncover_dynamics(self):
        return dict(self._last_uncover_dynamics)

    def get_last_reverse_dynamics(self):
        return dict(self._last_reverse_dynamics)

    def set_singulate(self, singulate_layers):
        self.singulate_layers = singulate_layers

    def set_env_variations(self, collect_data, blanket_pose_var, high_pose_var, body_shape_var):
        self.collect_data = collect_data
        self.blanket_pose_var = blanket_pose_var
        self.high_pose_var = high_pose_var
        self.body_shape_var = body_shape_var
        self.body_shape = None if self.body_shape_var == True else np.zeros((1, 10))
        self.gender = 'random' if self.body_shape_var == True else 'female'

    def set_release_sim_steps(self, pre_release_steps=None, post_release_steps=None):
        if pre_release_steps is not None:
            self.pre_release_steps = max(0, int(pre_release_steps))
        if post_release_steps is not None:
            self.post_release_steps = max(0, int(post_release_steps))

    def set_release_quiet_settle(self, enabled=None, speed_threshold=None, max_steps=None):
        if enabled is not None:
            self.quiet_settle_enabled = bool(enabled)
        if speed_threshold is not None:
            self.quiet_settle_speed_threshold = max(0.0, float(speed_threshold))
        if max_steps is not None:
            self.quiet_settle_max_steps = max(0, int(max_steps))

    def set_uncover_release_gravity_boost(self, enabled=None, gravity_z=None):
        if enabled is not None:
            self.uncover_release_gravity_boost_enabled = bool(enabled)
        if gravity_z is not None:
            self.uncover_release_gravity_z = float(gravity_z)

    def _get_cloth_mesh_positions(self):
        return np.asarray(
            p.getMeshData(
                self.blanket,
                -1,
                flags=p.MESH_DATA_SIMULATION_MESH,
                physicsClientId=self.id,
            )[1],
            dtype=np.float32,
        )

    def _cloth_vertex_speed_p95(self, prev_positions, curr_positions):
        dt = max(self._physics_timestep(), 1e-8)
        speed = np.linalg.norm(curr_positions - prev_positions, axis=1) / dt
        return float(np.percentile(speed, 95))

    def _run_post_release_settle(self, phase_name):
        prev_positions = self._get_cloth_mesh_positions()
        settle_iter = 0
        max_steps = self.post_release_steps if not self.quiet_settle_enabled else max(
            self.post_release_steps,
            self.quiet_settle_max_steps,
        )
        boost_gravity = (
            self.uncover_release_gravity_boost_enabled
            and phase_name == "uncover_post_release_settle"
        )
        if boost_gravity:
            p.setGravity(0, 0, self.uncover_release_gravity_z, physicsClientId=self.id)
        try:
            while settle_iter < max_steps:
                settle_iter += 1
                self._step_simulation_record_cloth(phase_name, settle_iter)
                if not self.quiet_settle_enabled:
                    if settle_iter >= self.post_release_steps:
                        break
                    continue

                curr_positions = self._get_cloth_mesh_positions()
                if settle_iter >= self.post_release_steps:
                    speed_p95 = self._cloth_vertex_speed_p95(prev_positions, curr_positions)
                    if speed_p95 <= self.quiet_settle_speed_threshold:
                        break
                prev_positions = curr_positions
        finally:
            if boost_gravity:
                p.setGravity(0, 0, self.default_gravity_z, physicsClientId=self.id)

    def set_cloth_trajectory_recording(self, enabled, stride=1, max_frames=0):
        self.record_cloth_trajectory = bool(enabled)
        self.cloth_trajectory_stride = max(1, int(stride))
        self.cloth_trajectory_max_frames = max(0, int(max_frames))
        self._reset_cloth_trajectory()

    def _reset_cloth_trajectory(self):
        self.cloth_trajectory = []
        self._cloth_trajectory_step = 0

    def _physics_timestep(self):
        try:
            params = p.getPhysicsEngineParameters(physicsClientId=self.id)
            return float(params.get("fixedTimeStep", self.time_step))
        except Exception:
            return float(getattr(self, "time_step", 0.01))

    def _record_cloth_trajectory_frame(self, phase, substep, note=None, force=False):
        if not self.record_cloth_trajectory:
            return
        if self.cloth_trajectory_max_frames and len(self.cloth_trajectory) >= self.cloth_trajectory_max_frames:
            return
        if not force and (self._cloth_trajectory_step % self.cloth_trajectory_stride != 0):
            return
        try:
            positions = np.asarray(
                p.getMeshData(
                    self.blanket,
                    -1,
                    flags=p.MESH_DATA_SIMULATION_MESH,
                    physicsClientId=self.id,
                )[1],
                dtype=np.float32,
            )
        except Exception:
            return
        try:
            sphere_pos = np.asarray(self.sphere_ee.get_base_pos_orient()[0], dtype=np.float32)
        except Exception:
            sphere_pos = None
        try:
            anchor_idx = [int(v) for v in list(self.anchor_idx)]
        except Exception:
            anchor_idx = []

        self.cloth_trajectory.append({
            "phase": str(phase),
            "substep": int(substep),
            "sim_step": int(self._cloth_trajectory_step),
            "note": note,
            "positions": positions,
            "sphere_pos": sphere_pos,
            "anchor_idx": anchor_idx,
        })

    def _step_simulation_record_cloth(self, phase, substep):
        p.stepSimulation(physicsClientId=self.id)
        self._cloth_trajectory_step += 1
        self._record_cloth_trajectory_frame(phase, substep)

    def _cloth_trajectory_payload(self):
        if not self.cloth_trajectory:
            return None
        return {
            "dt": self._physics_timestep(),
            "stride": int(self.cloth_trajectory_stride),
            "max_frames": int(self.cloth_trajectory_max_frames),
            "frames": self.cloth_trajectory,
        }

    def get_human_body_info(self):
        return self.human_creation.body_info if self.body_shape_var else None

    def initialize_skipped_uncover(self, reason="state_already_uncovered"):
        """Initialize cycle state without moving cloth when uncover is skipped."""
        self.uncover_action = np.zeros(4, dtype=np.float32)
        self.execute_uncover_action = False
        self.obs = self._get_obs()
        current = p.getMeshData(
            self.blanket,
            -1,
            flags=p.MESH_DATA_SIMULATION_MESH,
            physicsClientId=self.id,
        )
        self.cloth_initial = current
        self.cloth_intermediate = current
        self._reset_cloth_trajectory()
        self._record_cloth_trajectory_frame(
            "uncover_skipped", 0, note=str(reason), force=True
        )
        self.mesh = trimesh.load(
            os.path.join(self.directory, 'clothing', 'blanket_1061v.obj')
        )
        self.mesh_dict = {k: [] for k in range(1061)}
        for v in self.mesh.edges:
            v1, v2 = int(v[0]), int(v[1])
            self.mesh_dict[v1].append(v2)
            self.mesh_dict[v2].append(v1)
        self._last_uncover_grasp_xy = None
        self._last_uncover_release_xy = None
        self._last_uncover_grasp_z = None
        self._last_uncover_dynamics = {
            "executed": False,
            "skipped": True,
            "skip_reason": str(reason),
            "post_release_steps": 0,
        }
        return self.cloth_initial, self.cloth_intermediate, False

    def skip_recover_step(self, reason="state_already_covered"):
        """Finalize a cycle without moving cloth when recover is unnecessary."""
        self.recover_action = np.zeros(4, dtype=np.float32)
        self.execute_recover_action = False
        self.cloth_final = p.getMeshData(
            self.blanket,
            -1,
            flags=p.MESH_DATA_SIMULATION_MESH,
            physicsClientId=self.id,
        )
        self._record_cloth_trajectory_frame(
            "recover_skipped", 0, note=str(reason), force=True
        )
        return self.cloth_final, False

    def uncover_step(self, uncover_action):
        self.uncover_action = uncover_action
        assert len(self.uncover_action) == 4, "Error: uncover action != 4"

        self.execute_uncover_action = True
        self.obs = self._get_obs()

        uncover_action = scale_action(uncover_action) if not self.naive else scale_action(uncover_action, scale=[1, 1])
        grasp_loc = uncover_action[0:2]
        release_loc = uncover_action[2:4]

        # * get points on the blanket, initial state of the cloth
        self.cloth_initial = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        self._reset_cloth_trajectory()
        self._record_cloth_trajectory_frame("uncover_initial", 0, note="pre-action", force=True)
        self.mesh = trimesh.load(os.path.join(self.directory, 'clothing', 'blanket_1061v.obj'))

        # create dict with all the connected vertices for one vertex
        self.mesh_dict = {k:[] for k in range(1061)}
        for v in self.mesh.edges:
            v1 = v[0]
            v2 = v[1]
            self.mesh_dict[v1].append(v2)
            self.mesh_dict[v2].append(v1)

        if self.show_cloth_vertex_ids:
            for i, v in enumerate(self.cloth_initial[1]):
                color = [0, 0, 0]
                if i in [527, 14, 394]:
                    color = [1, 0, 0]
                p.addUserDebugText(text=str(i), textPosition=v, textColorRGB=color, textSize=1, lifeTime=0, physicsClientId=self.id)

        # p.setGravity(0, 0, 0, physicsClientId=self.id)

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist, is_on_cloth = check_grasp_on_cloth(uncover_action, np.array(self.cloth_initial[1]), clipping_thres=.028)
        # * if no points on the blanket are within 2.8 cm of the grasp location, exit (if collecting data) or proceed without executing the action (in all other conditions)
        if not is_on_cloth:
            self.execute_uncover_action = False
            self.cloth_intermediate = self.cloth_initial

        lower_step_count = 0
        release_z = None
        if self.execute_uncover_action:
            if self.singulate_layers:
                highest_vertex = singulate_layer_height(grasp_loc, np.array(self.cloth_initial[1]))
                neighbors = list(self.mesh_dict.get(highest_vertex, []))
                anchor_indices = [highest_vertex]
                anchor_pair_found = False
                for v1 in neighbors:
                    shared = [v2 for v2 in self.mesh_dict.get(v1, []) if v2 in neighbors and v2 != highest_vertex]
                    if shared:
                        anchor_indices.extend([v1, shared[0]])
                        anchor_pair_found = True
                        break
                if not anchor_pair_found or len(anchor_indices) < 3:
                    anchor_indices = list(np.argpartition(np.array(dist), 4)[:4])
                self.anchor_idx = anchor_indices[:4]
            else:
                self.anchor_idx = np.argpartition(np.array(dist), 4)[:4] # Finding set of points closest to grasp point
            # * update grasp_loc var with the location of the central anchor point on the cloth
            # Get the vertices on the blanket mesh and pull the x,y corresponding to the first anchor index
            grasp_loc = np.array(self.cloth_initial[1][self.anchor_idx[0]][0:2])
            # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
            self.sphere_ee.set_base_pos_orient(self.cloth_initial[1][self.anchor_idx[0]], np.array([0,0,0]))
            constraint_ids = []
            #Hold the first anchor
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, self.anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
            for i in self.anchor_idx[1:]:
                #Get the mesh vertex x,y corresponding to the anchor id index and subtract the 0th index
                #Gives the relative position to the first anchor index
                pos_diff = np.array(self.cloth_initial[1][i]) - np.array(self.cloth_initial[1][self.anchor_idx[0]])
                #Add the other anchors
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))
            # * move sphere up by some delta z
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            delta_z = self.min_threshold
            bed_height = 0.58                        # height of the bed
            final_z = delta_z + bed_height           # global goal z position

            #Moves sphere up to the height
            lift_iter = 0
            while current_pos[2] <= final_z:
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
                lift_iter += 1
                self._step_simulation_record_cloth("uncover_lift", lift_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            # * move sphere to the release location, release the blanket
            travel_dist = release_loc - grasp_loc

            # * determine delta x and y, make sure it is, at max, close to 0.005
            num_steps = np.abs(travel_dist//0.005).max()
            delta_x, delta_y = travel_dist/num_steps
            delta_z = 0
            current_pos = self.sphere_ee.get_base_pos_orient()[0]

            #Moves the sphere to the release location
            for translate_iter in range(1, int(num_steps) + 1):
                delta_z = check_height_of_effector(np.array(current_pos), np.array(self.points_pos_limb_world), self.min_threshold, self.max_threshold, self.line_threshold)
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, delta_z]), np.array([0,0,0]))
                self._step_simulation_record_cloth("uncover_translate", translate_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            if self.lower_before_release_uncover:
                if self.uncover_release_lower_delta is not None and self.uncover_release_lower_delta > 0:
                    target_z = float(current_pos[2]) - float(self.uncover_release_lower_delta)
                    while current_pos[2] > target_z + 1e-6:
                        step = min(0.005, float(current_pos[2]) - target_z)
                        self.sphere_ee.set_base_pos_orient(
                            current_pos + np.array([0, 0, -step]),
                            np.array([0, 0, 0]),
                        )
                        lower_step_count += 1
                        self._step_simulation_record_cloth("uncover_lower", lower_step_count)
                        current_pos = self.sphere_ee.get_base_pos_orient()[0]
                else:
                    while True:
                        delta_z = release_height_of_effector(
                            np.array(current_pos),
                            np.array(self.points_pos_limb_world),
                            self.release_threshold,
                        )
                        if delta_z >= -1e-6:
                            break
                        self.sphere_ee.set_base_pos_orient(
                            current_pos + np.array([0, 0, delta_z]),
                            np.array([0,0,0]),
                        )
                        lower_step_count += 1
                        self._step_simulation_record_cloth("uncover_lower", lower_step_count)
                        current_pos = self.sphere_ee.get_base_pos_orient()[0]

            for pre_release_iter in range(1, self.pre_release_steps + 1):
                self._step_simulation_record_cloth("uncover_pre_release_settle", pre_release_iter)
            # * release the cloth after lowering back to the original grasp height
            for i in constraint_ids:
                p.removeConstraint(i, physicsClientId=self.id)
            self._record_cloth_trajectory_frame("uncover_release", 0, note="constraints_removed", force=True)
            self._run_post_release_settle("uncover_post_release_settle")

            # * get points on the blanket, intermediate state of the cloth
            self.cloth_intermediate = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            self._record_cloth_trajectory_frame("uncover_intermediate", 0, note="post-uncover", force=True)
            try:
                release_z = float(self.sphere_ee.get_base_pos_orient()[0][2])
            except Exception:
                release_z = None
            # Persist actual world pick/place used for optional constant-Z reverse.
            self._last_uncover_grasp_xy = np.asarray(grasp_loc, dtype=np.float64)[:2].copy()
            self._last_uncover_release_xy = np.asarray(release_loc, dtype=np.float64)[:2].copy()
            try:
                self._last_uncover_grasp_z = float(self.cloth_initial[1][self.anchor_idx[0]][2])
            except Exception:
                self._last_uncover_grasp_z = None
        else:
            self._last_uncover_grasp_xy = np.asarray(grasp_loc, dtype=np.float64)[:2].copy()
            self._last_uncover_release_xy = np.asarray(release_loc, dtype=np.float64)[:2].copy()
            self._last_uncover_grasp_z = None

        self._last_uncover_dynamics = {
            "lower_before_release_uncover": bool(self.lower_before_release_uncover),
            "uncover_release_lower_delta": (
                None if self.uncover_release_lower_delta is None
                else float(self.uncover_release_lower_delta)
            ),
            "release_threshold": float(self.release_threshold),
            "lower_step_count": int(lower_step_count),
            "release_z": release_z,
            "pre_release_steps": int(self.pre_release_steps),
            "post_release_steps": int(self.post_release_steps),
            "anchor_count": int(len(self.anchor_idx)) if len(self.anchor_idx) else 0,
            "quiet_settle": bool(self.quiet_settle_enabled),
            "uncover_release_gravity_boost": bool(self.uncover_release_gravity_boost_enabled),
            "default_gravity_z": float(self.default_gravity_z),
            "grasp_xy": None if self._last_uncover_grasp_xy is None else self._last_uncover_grasp_xy.astype(float).tolist(),
            "release_xy": None if self._last_uncover_release_xy is None else self._last_uncover_release_xy.astype(float).tolist(),
            "grasp_z": self._last_uncover_grasp_z,
        }

        return self.cloth_initial, self.cloth_intermediate, self.execute_uncover_action

    def recover_step(self, recover_action):
        if not self.recover:
            self.cloth_final = self.cloth_intermediate
            return self.cloth_final, False

        self.recover_action = recover_action
        assert len(self.recover_action) == 4, "Error: recover action != 4"
        self.execute_recover_action = True

        recover_action = scale_action(recover_action) if not self.naive else scale_action(recover_action, scale=[1, 1])
        grasp_loc = recover_action[0:2]
        release_loc = recover_action[2:4]

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist, is_on_cloth = check_grasp_on_cloth(recover_action, np.array(self.cloth_intermediate[1]))

        # * if no points on the blanket are within 2.8 cm of the grasp location, exit (if collecting data) or proceed without executing the action (in all other conditions)
        if not is_on_cloth:
            self.execute_recover_action = False

        if self.execute_recover_action:
            if self.singulate_layers:
                # get the highest vertex near the grasp point
                highest_vertex = singulate_layer_height(grasp_loc, np.array(self.cloth_intermediate[1]))
                # change the anchor index to be the triangle of points connected to the highest vertex
                break_outer = False
                for i, v1 in enumerate(self.mesh_dict[highest_vertex]):
                    rest = self.mesh_dict[highest_vertex][:i] + self.mesh_dict[highest_vertex][i+1:]
                    for j in rest:
                        if v1 in self.mesh_dict[j]:
                            v2 = j
                            break_outer = True
                            break
                    if break_outer:
                        break
                self.anchor_idx = [highest_vertex, v1, v2]
            else:
                self.anchor_idx = np.argpartition(np.array(dist), 4)[:4] # Finding set of points closest to grasp point
            # * update grasp_loc var with the location of the central anchor point on the cloth
            grasp_loc = np.array(self.cloth_intermediate[1][self.anchor_idx[0]][0:2])

            # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
            self.sphere_ee.set_base_pos_orient(self.cloth_intermediate[1][self.anchor_idx[0]], np.array([0,0,0]))
            constraint_ids = []

            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, self.anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
            for i in self.anchor_idx[1:]:
                pos_diff = np.array(self.cloth_intermediate[1][i]) - np.array(self.cloth_intermediate[1][self.anchor_idx[0]])
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))

            # * move sphere up by some delta z
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            # Restore classic recover lift: 0.4 m above bed → final Z ≈ 0.98 m.
            delta_z = 0.4
            bed_height = 0.58                        # height of the bed
            final_z = delta_z + bed_height           # global goal z position
            lift_iter = 0
            while current_pos[2] <= final_z:
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
                lift_iter += 1
                self._step_simulation_record_cloth("recover_lift", lift_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            # * move sphere to the release location, release the blanket
            travel_dist = release_loc - grasp_loc

            # * determine delta x and y, make sure it is, at max, close to 0.005
            num_steps = np.abs(travel_dist//0.005).max()
            delta_x, delta_y = travel_dist/num_steps

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            for translate_iter in range(1, int(num_steps) + 1):
                self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, 0]), np.array([0,0,0]))
                self._step_simulation_record_cloth("recover_translate", translate_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            while True:
                delta_z = release_height_of_effector(
                    np.array(current_pos),
                    np.array(self.points_pos_limb_world),
                    self.release_threshold,
                )
                if delta_z >= -1e-6:
                    break
                self.sphere_ee.set_base_pos_orient(
                    current_pos + np.array([0, 0, delta_z]),
                    np.array([0,0,0]),
                )
                self._step_simulation_record_cloth("recover_lower", 0)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            for pre_release_iter in range(1, self.pre_release_steps + 1):
                self._step_simulation_record_cloth("recover_pre_release_settle", pre_release_iter)

            for i in constraint_ids:
                p.removeConstraint(i, physicsClientId=self.id)
            self._record_cloth_trajectory_frame("recover_release", 0, note="constraints_removed", force=True)
            self._run_post_release_settle("recover_post_release_settle")

        # * get points on the blanket, final state of the cloth
        self.cloth_final = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        self._record_cloth_trajectory_frame("recover_final", 0, note="post-recover", force=True)

        return self.cloth_final, self.execute_recover_action

    def recover_step_with_exposure_cutoff(
        self,
        recover_action,
        upper_body_points,
        tau_online=2,
        monitor_stride=2,
        record_exposure_timeline=True,
    ):
        """Recover with online upper-body exposure monitoring during translate.

        When newly-exposed upper-body points (vs recover-start coverage) exceed
        ``tau_online``, stop translate immediately and lower/release at the
        **current** EE XY (no snap-back to last_safe — cloth is not rewound by
        pulling the EE back).

        Cutoff only decides when to stop pulling; final accept/reject is decided
        by the collector from post-release metrics.

        Returns
        -------
        cloth_final, execute_recover, cutoff_meta
        """
        if not self.recover:
            self.cloth_final = self.cloth_intermediate
            meta = {
                "cutoff_triggered": False,
                "cutoff_fraction": 1.0,
                "cutoff_reason": "recover_disabled",
                "intended_release_world": None,
                "actual_release_world": None,
                "intended_release_policy": None,
                "actual_release_policy": None,
                "cutoff_step": 0,
                "last_safe_step": 0,
                "last_safe_fraction": 0.0,
                "num_translate_steps": 0,
                "exposure_timeline": [],
                "online_exceeded": False,
            }
            return self.cloth_final, False, meta

        self.recover_action = np.asarray(recover_action, dtype=np.float32).reshape(-1)
        assert len(self.recover_action) == 4, "Error: recover action != 4"
        self.execute_recover_action = True

        recover_world = scale_action(self.recover_action) if not self.naive else scale_action(
            self.recover_action, scale=[1, 1]
        )
        grasp_loc = np.asarray(recover_world[0:2], dtype=np.float64)
        release_loc_intended = np.asarray(recover_world[2:4], dtype=np.float64)

        dist, is_on_cloth = check_grasp_on_cloth(
            recover_world, np.array(self.cloth_intermediate[1])
        )
        if not is_on_cloth:
            self.execute_recover_action = False

        upper_pts = np.asarray(upper_body_points, dtype=np.float64)
        cloth_start = np.asarray(self.cloth_intermediate[1], dtype=np.float64)
        status_start = get_covered_status(upper_pts, cloth_start[:, :2]) if upper_pts.size else []

        def _newly_exposed(cloth_xyz):
            if upper_pts.size == 0 or len(status_start) == 0:
                return 0
            status_curr = get_covered_status(upper_pts, np.asarray(cloth_xyz, dtype=np.float64)[:, :2])
            n = 0
            for a, b in zip(status_start, status_curr):
                if a[0] == 1 and a[1] and (not b[1]):
                    n += 1
            return int(n)

        exposure_timeline = []
        cutoff_triggered = False
        online_exceeded = False
        cutoff_reason = ""
        last_safe_step = 0
        last_safe_fraction = 0.0
        cutoff_step = 0
        alpha_stop = 1.0
        num_steps = 0
        actual_release_world = release_loc_intended.copy()

        if self.execute_recover_action:
            if self.singulate_layers:
                highest_vertex = singulate_layer_height(grasp_loc, np.array(self.cloth_intermediate[1]))
                break_outer = False
                v1 = v2 = highest_vertex
                for i, v1 in enumerate(self.mesh_dict[highest_vertex]):
                    rest = self.mesh_dict[highest_vertex][:i] + self.mesh_dict[highest_vertex][i + 1 :]
                    for j in rest:
                        if v1 in self.mesh_dict[j]:
                            v2 = j
                            break_outer = True
                            break
                    if break_outer:
                        break
                self.anchor_idx = [highest_vertex, v1, v2]
            else:
                self.anchor_idx = np.argpartition(np.array(dist), 4)[:4]
            grasp_loc = np.array(self.cloth_intermediate[1][self.anchor_idx[0]][0:2], dtype=np.float64)

            self.sphere_ee.set_base_pos_orient(
                self.cloth_intermediate[1][self.anchor_idx[0]], np.array([0, 0, 0])
            )
            constraint_ids = []
            constraint_ids.append(
                p.createSoftBodyAnchor(self.blanket, self.anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0])
            )
            for i in self.anchor_idx[1:]:
                pos_diff = np.array(self.cloth_intermediate[1][i]) - np.array(
                    self.cloth_intermediate[1][self.anchor_idx[0]]
                )
                constraint_ids.append(
                    p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff)
                )

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            delta_z = 0.4
            bed_height = 0.58
            final_z = delta_z + bed_height
            lift_iter = 0
            while current_pos[2] <= final_z:
                self.sphere_ee.set_base_pos_orient(
                    current_pos + np.array([0, 0, 0.005]), np.array([0, 0, 0])
                )
                lift_iter += 1
                self._step_simulation_record_cloth("recover_lift", lift_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            travel_dist = release_loc_intended - grasp_loc
            step_norm = np.abs(travel_dist // 0.005).max()
            num_steps = int(step_norm) if np.isfinite(step_norm) else 0
            if num_steps < 1:
                num_steps = 1
            delta_x, delta_y = travel_dist / float(num_steps)

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            last_safe_step = 0
            last_safe_fraction = 0.0
            alpha_stop = 0.0

            for translate_iter in range(1, num_steps + 1):
                self.sphere_ee.set_base_pos_orient(
                    current_pos + np.array([delta_x, delta_y, 0]), np.array([0, 0, 0])
                )
                self._step_simulation_record_cloth("recover_translate", translate_iter)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]
                alpha = float(translate_iter) / float(num_steps)

                do_monitor = (translate_iter % max(1, int(monitor_stride)) == 0) or (
                    translate_iter == num_steps
                )
                newly = 0
                if do_monitor:
                    mesh = p.getMeshData(
                        self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id
                    )
                    newly = _newly_exposed(mesh[1])
                    if record_exposure_timeline:
                        exposure_timeline.append(
                            {
                                "phase": "recover_translate",
                                "step": int(translate_iter),
                                "alpha": alpha,
                                "newly_exposed_upper": int(newly),
                                "sphere_xy": [
                                    float(current_pos[0]),
                                    float(current_pos[1]),
                                ],
                            }
                        )
                    if newly <= float(tau_online):
                        # Audit only: last still-under-threshold pose.
                        last_safe_step = int(translate_iter)
                        last_safe_fraction = alpha
                    else:
                        # Stop immediately at current EE XY — do NOT pull EE back.
                        online_exceeded = True
                        cutoff_triggered = True
                        cutoff_reason = "upper_exposure_exceeded_during_motion"
                        cutoff_step = int(translate_iter)
                        alpha_stop = alpha
                        break

            actual_release_world = np.asarray(current_pos[:2], dtype=np.float64)
            if not cutoff_triggered:
                alpha_stop = 1.0
                cutoff_step = int(num_steps)
                last_safe_step = int(num_steps)
                last_safe_fraction = 1.0
                actual_release_world = release_loc_intended.copy()

            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            while True:
                delta_z = release_height_of_effector(
                    np.array(current_pos),
                    np.array(self.points_pos_limb_world),
                    self.release_threshold,
                )
                if delta_z >= -1e-6:
                    break
                self.sphere_ee.set_base_pos_orient(
                    current_pos + np.array([0, 0, delta_z]),
                    np.array([0, 0, 0]),
                )
                self._step_simulation_record_cloth("recover_lower", 0)
                current_pos = self.sphere_ee.get_base_pos_orient()[0]

            for pre_release_iter in range(1, self.pre_release_steps + 1):
                self._step_simulation_record_cloth("recover_pre_release_settle", pre_release_iter)

            for i in constraint_ids:
                p.removeConstraint(i, physicsClientId=self.id)
            self._record_cloth_trajectory_frame(
                "recover_release", 0, note="constraints_removed", force=True
            )
            self._run_post_release_settle("recover_post_release_settle")

        self.cloth_final = p.getMeshData(
            self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id
        )
        self._record_cloth_trajectory_frame("recover_final", 0, note="post-recover", force=True)

        # Persist actual policy action for training / logging.
        scale = np.array([1.0, 1.0], dtype=np.float64) if self.naive else np.array([0.44, 1.05], dtype=np.float64)
        g_pol = grasp_loc / scale
        r_pol = np.asarray(actual_release_world, dtype=np.float64) / scale
        actual_policy = np.array(
            [g_pol[0], g_pol[1], r_pol[0], r_pol[1]], dtype=np.float32
        )
        if self.execute_recover_action:
            self.recover_action = actual_policy

        intended_policy = np.asarray(recover_action, dtype=np.float32).reshape(-1)
        meta = {
            "cutoff_triggered": bool(cutoff_triggered),
            "cutoff_fraction": float(alpha_stop),
            "cutoff_reason": cutoff_reason or ("none" if not cutoff_triggered else cutoff_reason),
            "intended_release_world": [float(x) for x in release_loc_intended],
            "actual_release_world": [float(x) for x in actual_release_world],
            "intended_release_xy": [float(x) for x in release_loc_intended],
            "actual_release_xy": [float(x) for x in actual_release_world],
            "intended_release_policy": [float(x) for x in intended_policy[2:4]],
            "actual_release_policy": [float(x) for x in actual_policy[2:4]],
            "grasp_world": [float(x) for x in grasp_loc],
            "cutoff_step": int(cutoff_step),
            "last_safe_step": int(last_safe_step),
            "last_safe_fraction": float(last_safe_fraction),
            "num_translate_steps": int(num_steps),
            "tau_online": float(tau_online),
            "monitor_stride": int(monitor_stride),
            "online_exceeded": bool(online_exceeded),
            "exposure_timeline": exposure_timeline,
            "ee_pullback": False,
        }
        return self.cloth_final, self.execute_recover_action, meta

    def constant_z_reverse_step(self, xy_step=0.005, post_release_steps=None):
        """Inverse uncover via constant-Z reverse XY transport.

        Grasp near the original uncover place (highest local layer), keep the grasp
        Z fixed, move in XY toward the original uncover pick with steps <= xy_step,
        then release and settle. Does not use recover_step lift or adaptive lowering.
        """
        if self.cloth_intermediate is None or len(self.cloth_intermediate) == 0:
            raise RuntimeError("constant_z_reverse_step requires a completed uncover_step")
        if self._last_uncover_grasp_xy is None or self._last_uncover_release_xy is None:
            # Fallback to scaled uncover action if provenance was not stored.
            uncover_world = scale_action(self.uncover_action) if not self.naive else scale_action(self.uncover_action, scale=[1, 1])
            self._last_uncover_grasp_xy = np.asarray(uncover_world[0:2], dtype=np.float64)
            self._last_uncover_release_xy = np.asarray(uncover_world[2:4], dtype=np.float64)

        pick_xy = np.asarray(self._last_uncover_release_xy, dtype=np.float64)[:2]
        place_xy = np.asarray(self._last_uncover_grasp_xy, dtype=np.float64)[:2]
        # Store recover_action in policy units so get_info reward paths remain valid.
        recover_world = np.asarray([pick_xy[0], pick_xy[1], place_xy[0], place_xy[1]], dtype=np.float64)
        if self.naive:
            self.recover_action = recover_world.astype(np.float32)
        else:
            self.recover_action = (recover_world / np.array([0.44, 1.05, 0.44, 1.05], dtype=np.float64)).astype(np.float32)

        cloth = np.asarray(self.cloth_intermediate[1], dtype=np.float64)
        dist, is_on_cloth = check_grasp_on_cloth(recover_world, cloth, clipping_thres=.028)
        nearest_cloth_distance = float(np.min(dist)) if len(dist) else float("inf")
        self.execute_recover_action = bool(is_on_cloth)

        fixed_z = None
        z_samples = []
        num_steps = 0
        collision_or_error = False
        settle_steps = self.post_release_steps if post_release_steps is None else max(0, int(post_release_steps))
        previous_post = self.post_release_steps
        if post_release_steps is not None:
            self.post_release_steps = settle_steps

        if self.execute_recover_action:
            try:
                if self.singulate_layers:
                    highest_vertex = singulate_layer_height(pick_xy, cloth)
                    neighbors = list(self.mesh_dict.get(highest_vertex, []))
                    anchor_indices = [highest_vertex]
                    anchor_pair_found = False
                    for v1 in neighbors:
                        shared = [v2 for v2 in self.mesh_dict.get(v1, []) if v2 in neighbors and v2 != highest_vertex]
                        if shared:
                            anchor_indices.extend([v1, shared[0]])
                            anchor_pair_found = True
                            break
                    if not anchor_pair_found or len(anchor_indices) < 3:
                        anchor_indices = list(np.argpartition(np.array(dist), 4)[:4])
                    self.anchor_idx = anchor_indices[:4]
                else:
                    self.anchor_idx = np.argpartition(np.array(dist), 4)[:4]

                grasp_vertex = cloth[self.anchor_idx[0]]
                grasp_loc = np.asarray(grasp_vertex[0:2], dtype=np.float64)
                fixed_z = float(grasp_vertex[2])
                self.sphere_ee.set_base_pos_orient(grasp_vertex, np.array([0, 0, 0]))
                constraint_ids = []
                constraint_ids.append(p.createSoftBodyAnchor(self.blanket, self.anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
                for i in self.anchor_idx[1:]:
                    pos_diff = np.array(cloth[i]) - np.array(cloth[self.anchor_idx[0]])
                    constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))

                travel_dist = place_xy - grasp_loc
                step = max(1e-6, float(xy_step))
                num_steps = int(np.abs(travel_dist / step).max())
                num_steps = max(1, num_steps)
                delta_x, delta_y = travel_dist / float(num_steps)
                current_pos = np.asarray(self.sphere_ee.get_base_pos_orient()[0], dtype=np.float64)
                z_samples.append(float(current_pos[2]))
                for translate_iter in range(1, int(num_steps) + 1):
                    next_pos = np.array(
                        [current_pos[0] + delta_x, current_pos[1] + delta_y, fixed_z],
                        dtype=np.float64,
                    )
                    self.sphere_ee.set_base_pos_orient(next_pos, np.array([0, 0, 0]))
                    self._step_simulation_record_cloth("reverse_translate", translate_iter)
                    current_pos = np.asarray(self.sphere_ee.get_base_pos_orient()[0], dtype=np.float64)
                    # Re-assert constant Z each step (controller contract).
                    if abs(float(current_pos[2]) - float(fixed_z)) > 1e-6:
                        current_pos[2] = fixed_z
                        self.sphere_ee.set_base_pos_orient(current_pos, np.array([0, 0, 0]))
                    z_samples.append(float(current_pos[2]))

                for pre_release_iter in range(1, self.pre_release_steps + 1):
                    self._step_simulation_record_cloth("reverse_pre_release_settle", pre_release_iter)
                for constraint_id in constraint_ids:
                    p.removeConstraint(constraint_id, physicsClientId=self.id)
                self._record_cloth_trajectory_frame("reverse_release", 0, note="constraints_removed", force=True)
                self._run_post_release_settle("reverse_post_release_settle")
            except Exception as exc:
                collision_or_error = True
                self.execute_recover_action = False
                self._last_reverse_dynamics = {
                    "error": str(exc),
                }

        self.cloth_final = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        self._record_cloth_trajectory_frame("reverse_final", 0, note="post-reverse", force=True)
        if post_release_steps is not None:
            self.post_release_steps = previous_post

        z_arr = np.asarray(z_samples, dtype=np.float64) if z_samples else np.asarray([], dtype=np.float64)
        z_drift_max = float(np.max(np.abs(z_arr - fixed_z))) if z_arr.size and fixed_z is not None else None
        z_variance = float(np.var(z_arr)) if z_arr.size else None
        self._last_reverse_dynamics = {
            "controller": "constant_z_reverse_xy",
            "executed": bool(self.execute_recover_action),
            "pick_is_on_cloth": bool(is_on_cloth),
            "nearest_cloth_distance": float(nearest_cloth_distance),
            "fixed_z": None if fixed_z is None else float(fixed_z),
            "z_samples": z_arr.astype(float).tolist(),
            "z_drift_max": z_drift_max,
            "z_variance": z_variance,
            "num_steps": int(num_steps),
            "xy_step": float(xy_step),
            "pick_xy": pick_xy.astype(float).tolist(),
            "place_xy": place_xy.astype(float).tolist(),
            "post_release_steps": int(settle_steps),
            "pre_release_steps": int(self.pre_release_steps),
            "anchor_count": int(len(self.anchor_idx)) if len(self.anchor_idx) else 0,
            "collision_or_error": bool(collision_or_error),
            "default_gravity_z": float(self.default_gravity_z),
            "no_adaptive_lowering": True,
            "no_recover_lift": True,
        }
        return self.cloth_final, self.execute_recover_action

    def get_info(self):
        human_pose = np.reshape(self.human_pose, (-1,2))
        all_body_points = get_body_points_from_obs(human_pose, target_limb_code=self.target_limb_code, body_info=self.get_human_body_info())

        cloth_initial_subsample, cloth_intermediate_subsample, cloth_final_subsample = sub_sample_point_clouds_recover(self.cloth_initial[1], self.cloth_intermediate[1], self.cloth_final[1])

        cloth_initial_2D = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
        cloth_intermediate_2D = np.delete(np.array(cloth_intermediate_subsample), 2, axis = 1)
        cloth_final_2D = np.delete(np.array(cloth_final_subsample), 2, axis = 1)

        self.uncover_reward, uncovered_status = get_uncovering_reward(self.uncover_action, all_body_points, cloth_initial_2D, cloth_final_2D)
        self.recover_reward, recovered_status = get_recovering_reward(self.recover_action, all_body_points, cloth_initial_2D, cloth_intermediate_2D, cloth_final_2D)

        if not self.recover:
            self.cloth_intermediate = []
            self.recover_reward = []
            self.recovered_status = []

        try:
            anchor_idx = [int(v) for v in list(self.anchor_idx)]
        except Exception:
            anchor_idx = []

        if not self.collect_data:
            info = {
                "recovering" : self.recover,
                "cloth_initial" : self.cloth_initial,
                "cloth_intermediate" : self.cloth_intermediate,
                "cloth_final": self.cloth_final,
                "RBG_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "uncovered_status_sim": uncovered_status,
                "recovered_status_sim" : recovered_status,
                "target_limb_code":self.target_limb_code,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender,
                "grasp_on_cloth_uncover":self.execute_uncover_action,
                "grasp_on_cloth_recover":self.execute_recover_action,
                "anchor_idx": anchor_idx,
                "anchor_count": int(len(anchor_idx)),
                }
        else:
            info = {
                "recovering" : self.recover,
                "cloth_initial": self.cloth_initial,
                "cloth_intermediate" : self.cloth_intermediate,
                "cloth_final": self.cloth_final,
                "RGB_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth,
                "point_cloud_depth_img": self.point_cloud_depth_img,
                "human_body_info": self.human_creation.body_info if self.body_shape_var else None,
                "gender":self.human.gender,
                "all_body_points": all_body_points,
                "anchor_idx": anchor_idx,
                "anchor_count": int(len(anchor_idx)),
                "grasp_on_cloth_uncover": self.execute_uncover_action,
                "grasp_on_cloth_recover": self.execute_recover_action,
                }

        cloth_trajectory = self._cloth_trajectory_payload()
        if cloth_trajectory is not None:
            info["cloth_trajectory"] = cloth_trajectory

        self.iteration += 1
        done = self.iteration >= 1

        return self.obs, self.uncover_reward, self.recover_reward, done, info

    def set_pstate_file(self, filename):
        if self.pstate_file != filename:
            self.pstate_file = filename
            self.save_pstate = True


    def get_cloth_state(self):
        return p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]

    def _get_obs(self, agent=None):
        pose = []
        for limb in self.human.obs_limbs:
            pos, orient = self.human.get_pos_orient(limb)
            pos2D = pos[0:2]
            pose.append(pos2D)
        pose = np.concatenate(pose, axis=0)
        self.human_pose = pose


        if self.collect_data:
            output = [None]*28
            all_joint_angles = self.human.get_joint_angles(self.human.all_joint_indices)
            all_pos_orient = [self.human.get_pos_orient(limb) for limb in self.human.all_body_parts]
            output[0], output[1], output[2] = pose, all_joint_angles, all_pos_orient
            return output

        return np.float32(pose)

    def reset(self):
        if self.replay_seed:
            self.seed(self.seed_val)

        super(RobeReversibleEnv, self).reset()

        self.build_assistive_env(fixed_human_base=False, gender=self.gender, human_impairment='none', furniture_type='hospital_bed', body_shape=self.body_shape)

        self.target_limb_code = self.target_limb_code
        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # * Setup human in the air, with legs and arms slightly seperated
        joints_positions = [(self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]

        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)

        # move the human to new location above the bed (position, orientation)
        self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])
        random_variation = []
        if not self.fixed_pose:
            # * Add small variation to the body pose
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            random_variation = self.np_random.uniform(-.2, .2, size=len(motor_indices))
            self.human.set_joint_angles(motor_indices, motor_positions + random_variation)

            if self.high_pose_var:
                self.increase_pose_variation()
            # * Increase friction of joints so human doesn't fail around exessively as they settle
            # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
            self.human.set_whole_body_frictions(spinning_friction=2, lateral_friction=.6)

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

        # * Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        self.generate_points_along_body()

        # * Setup camera for taking images
        # *      Currently saves color images only to specified directory
        if self.take_pictures or self.collect_data:
            self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=398)
            img, depth = self.get_camera_image_depth()
            self.human_no_occlusion_RGB = img
            self.human_no_occlusion_depth = depth
            img = np.asarray(img)
            depth = np.asarray(depth)
            depth_range = np.amax(depth) - np.amin(depth)
            if depth_range > 0:
                depth = (depth - np.amin(depth)) / depth_range
            else:
                depth = np.zeros_like(depth, dtype=np.float32)
            depth = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)

        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_1061v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)

        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, .6], flags=0, physicsClientId=self.id)
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
            p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, .6], flags=0, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)

        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # * Initialize enviornment variables
        # *      if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
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
            self.action_robot_len = 8
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))     # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()

        return self._get_obs()

    def generate_points_along_body(self):
        '''
        generate all the target/nontarget posistions necessary to uniformly cover the body parts with points
        if rendering, generates sphere bodies as well
        '''

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0

        self.points_pos_on_nontarget_limb = {}
        self.points_nontarget_limb = {}
        self.total_nontarget_point_count = 0

        # * create points on all the body parts
        for limb in self.human.all_body_parts:

            # * get the length and radius of the given body part
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]


            # * create points seperately depending on whether or not the body part is/is a part of the target limb
            # *      generates list of point positions around the body part capsule (sphere if the hands)
            # *      creates all the spheres necessary to uniformly cover the body part (spheres created at some arbitrary position (transformed to correct location in update_points_along_body())
            # *      add to running total of target/nontarget points
            # *      only generate sphere bodies if self.rendering == True
            if hasattr(self.human, "target_limb"):
                if limb in self.target_limb:
                    if limb in [self.human.left_hand, self.human.right_hand]:
                        self.points_pos_on_target_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                    else:
                        self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                    if self.rendering:
                        self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                    self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_nontarget_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.rendering:
                    self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 1, 0, 0.2])
                self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])

        # * transforms the generated spheres to the correct coordinate space (aligns points to the limbs)
        self.update_points_along_body()

    def update_points_along_body(self):
        '''
        transforms the target/nontarget points created in generate_points_along_body() to the correct coordinate space so that they are aligned with their respective body part
        if rendering, transforms the sphere bodies as well
        '''

        # * positions of the points on the target/nontarget limbs in world coordinates
        self.points_pos_target_limb_world = {}
        self.points_pos_nontarget_limb_world = {}

        # * transform all spheres for all the body parts
        for limb in self.human.all_body_parts:

            # * get current position and orientation of the limbs, apply a correction to the pos, orient if necessary
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])

            # * transform target/nontarget point positions to the world coordinate system so they align with the body parts

            if hasattr(self.human, "target_limb"):
                if limb in self.target_limb:
                    for i in range(len(self.points_pos_on_target_limb[limb])):
                        point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_target_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                        self.points_pos_limb_world.append(point_pos)
                        if self.rendering:
                            self.points_target_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                    self.points_pos_target_limb_world[limb] = self.points_pos_limb_world
            else:
                for i in range(len(self.points_pos_on_nontarget_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_nontarget_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    self.points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_nontarget_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = self.points_pos_limb_world

    def increase_pose_variation(self):
        '''
        Allow more variation in the knee and elbow angles
          can be some random position within the lower and upper limits of the joint movement (range is made a little smaller than the limits of the joint to prevent angles that are too extreme)
        '''
        for joint in (self.human.j_left_knee, self.human.j_right_knee, self.human.j_left_elbow, self.human.j_right_elbow):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states([joint])
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(self.human.lower_limits[joint]+0.1, self.human.upper_limits[joint]-0.1, 1))

    def capture_images(self, state, iteration, height):
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=410)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        var_type = f"Testing_Recovering_Actions"
        filename = f'top_view_{state}_{iteration}.png'
        img_dir = osp.join(os.getcwd(), var_type)
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, filename), img)

    def set_iteration(self, seed):
        self.iteration = seed




