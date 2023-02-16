from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils import RandomizationError
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import quat2yaw
from robosuite.utils.primitive_skills import PrimitiveSkillDelta, PrimitiveSkillGlobal
from robosuite.controllers.controller_factory import load_controller_config

import pdb

class Cleanup(SingleArmEnv):
    """
    This class corresponds to the cleanup task used in MAPLE.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.5, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=2000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        task_config=None,
        use_skills=False,
        use_delta=None, # if set, ignore controller_configs and use osc controller (if True, use delta control, if false use global controller)
        normalized_params=True,
        use_aff_rewards=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # Get config
        # self.task_config = DEFAULT_CLEANUP_CONFIG.copy()
        if task_config is not None:
            assert all([k in self.task_config for k in task_config])
            self.task_config.update(task_config)

        self.gripper_state = -1 # -1 is open, 1 is closed

        # workspace limits
        self.workspace_x = (self.table_offset[0] - self.table_full_size[0]/2, self.table_offset[0] + self.table_full_size[0]/2)
        self.workspace_y = (self.table_offset[1] - self.table_full_size[1]/2, self.table_offset[1] + self.table_full_size[1]/2)
        self.workspace_z = (self.table_offset[2] + 0.03, self.table_offset[0] + 0.53)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        # setup controller
        if use_delta is not None:
            controller_configs = load_controller_config(default_controller="OSC_POSE")
            controller_configs["control_delta"] = use_delta
        
        # primitive skill mode 
        self.use_skills = use_skills  
        if use_delta == False:
            self.skill = PrimitiveSkillGlobal(
                skill_indices={
                    0 : "pick",
                    1 : "place",
                    2 : "push",
                }
            )
        else:
            self.skill = PrimitiveSkillDelta(
                skill_indices={
                    0 : "pick",
                    1 : "place",
                    2 : "push"
                }
            )

        self.keypoints = self.skill.get_keypoints_dict()
        self.use_aff_rewards = use_aff_rewards
        self.num_skills = self.skill.n_skills
        self.normalized_params = normalized_params

        # whether reward has been given - completion reward is only given once
        self.reward_given = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action):
        
        # full completion reward only
        reward = 0.0       

        if self._check_success() and not self.reward_given:
            reward = 10.0
            if not self.use_skills:
                self.reward_given = True
                print("~~~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~")            

        if self.reward_scale is not None:
            reward = self.reward_scale * reward / 10

        return reward

    def _reward(self, action=None):
        
        reward = 0.0
        
        # check if task is complete
        if self._check_success() and not self.reward_given:
            reward = 10.0
            self.reward_given = True
            print("~~~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~")

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 10

        return reward

    def reward_infos(self):
        rew_pnp = 0
        partial_rew_pnp = 0
        num_pnp_success = 0
        for i in range(self.task_config['num_pnp_objs']):
            r, g, l, h, b = self.pnp_staged_rewards(obj_id=i)

            if b == 1.0:
                rew_pnp += 1.0
                num_pnp_success += 1
            elif b == 0.0:
                partial_rew_pnp = max(partial_rew_pnp, max(r, g, l, h))
            else:
                raise ValueError

        if self.reward_shaping:
            rew_pnp += partial_rew_pnp

        rew_push = 0
        for i in range(self.task_config['num_push_objs']):
            r, p, d = self.push_staged_rewards(obj_id=i)
            rew_push += p
            if self.task_config['shaped_push_rew']:
                rew_push += r

        if self.task_config['use_pnp_rew'] and self.task_config['use_push_rew']:
            if self.task_config['rew_type'] == 'sum':
                reward = rew_pnp + rew_push
            elif self.task_config['rew_type'] == 'step':
                pnp_success = (num_pnp_success == self.task_config['num_pnp_objs'])
                reward = rew_pnp + float(pnp_success) * rew_push
            else:
                raise ValueError
        elif self.task_config['use_pnp_rew']:
            reward = rew_pnp
        elif self.task_config['use_push_rew']:
            reward = rew_push
        else:
            raise ValueError

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return rew_pnp, rew_push, reward

    def pnp_staged_rewards(self, obj_id=0):
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        obj_pos = self.sim.data.body_xpos[self.pnp_obj_body_ids[obj_id]]
        obj = self.pnp_objs[obj_id]

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=obj)
        ) * grasp_mult

        r_lift = 0.
        r_hover = 0.
        if r_grasp > 0.:
            table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
            z_target = table_pos[2] + 0.15
            obj_z = obj_pos[2]
            z_dist = np.maximum(z_target - obj_z, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (
                    lift_mult - grasp_mult
            )

            bin_xy = np.array(self.sim.data.body_xpos[self.bin_body_id])[:2]
            obj_xy = obj_pos[:2]
            dist = np.linalg.norm(bin_xy - obj_xy)
            r_hover = r_lift + (1 - np.tanh(10.0 * dist)) * (
                    hover_mult - lift_mult
            )

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_bin = self.in_bin(obj_pos)

        return r_reach, r_grasp, r_lift, r_hover, r_bin

    def push_staged_rewards(self, obj_id=0):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[obj_id]]
        target_pos_xy = self.table_offset[:2] + np.array([-0.15, 0.15])
        d_push = np.linalg.norm(obj_pos[:2] - target_pos_xy)

        th = [0.08, 0.08, 0.04]
        d_reach = np.sum(
            np.clip(
                np.abs(gripper_site_pos - obj_pos) - th,
                0, None
            )
        )
        r_reach = (1 - np.tanh(10.0 * d_reach)) * 0.25

        r_push = 1 - np.tanh(self.task_config['push_scale_fac'] * d_push)
        return r_reach, r_push, d_push

    def in_bin(self, obj_pos):
        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        res = False
        if (
                abs(obj_pos[0] - bin_pos[0]) < 0.10
                and abs(obj_pos[1] - bin_pos[1]) < 0.15
                and obj_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res

    def in_push_target(self, obj_pos):
        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        res = False
        if (
            obj_pos[0] < bin_pos[0] + 0.10
            and obj_pos[0] > self.table_offset[0] - self.table_full_size[0]
            and not self.in_bin(obj_pos)
        ):
            res = True
        return res
        
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            xml="arenas/table_arena_box.xml",
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        pnpmaterial = CustomMaterial(
            texture="Spam",
            tex_name="pnpobj_tex",
            mat_name="pnpobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        pushmaterial = CustomMaterial(
            texture="Jello",
            tex_name="pushobj_tex",
            mat_name="pushobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # initialize objects
        pnp_size = np.array([0.04, 0.022, 0.04]) * 0.75
        self.pnp_obj = BoxObject(
            name="obj_pnp",
            size=pnp_size,
            rgba=[1,0,0,1],
            material=pnpmaterial,
        )

        push_size = np.array([0.0350, 0.0425, 0.02]) * 1.20
        self.push_obj = BoxObject(
            name="obj_push",
            size=push_size,
            rgba=[0,1,0,1],
            material=pushmaterial,
            density=20,
        )

        objs = [self.pnp_obj, self.push_obj]
        
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objs)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=objs,
                x_range=[0.0, 0.16],
                y_range=[-0.16, 0.16],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objs,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.table_body_id = self.sim.model.body_name2id("table")

        self.pnp_obj_body_id = self.sim.model.body_name2id(self.pnp_obj.root_body)
        self.push_obj_body_id = self.sim.model.body_name2id(self.push_obj.root_body)

        self.bin_body_id = self.sim.model.body_name2id("bin")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            while True:
                try:
                    object_placements = self.placement_initializer.sample()
                    sample_success = True
                except RandomizationError:
                    sample_success = False

                if sample_success:
                    break

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
    def _setup_observables(self):
            """
            Sets up observables to be used for this environment. Creates object-based observables if enabled

            Returns:
                OrderedDict: Dictionary mapping observable names to its corresponding Observable object
            """
            observables = super()._setup_observables()

            # low-level object information
            if self.use_object_obs:
                # Get robot prefix and define observables modality
                pf = self.robots[0].robot_model.naming_prefix
                modality = "object"

                @sensor(modality=modality)
                def eef_xyz(obs_cache):
                    return (
                        np.array(obs_cache[f"{pf}eef_pos"])
                        if f"{pf}eef_pos" in obs_cache
                        else np.zeros(4)
                    )
                
                @sensor(modality=modality)
                def eef_yaw(obs_cache):
                    return (
                        quat2yaw(np.array(obs_cache[f"{pf}eef_quat"]))
                        if f"{pf}eef_quat" is obs_cache
                        else 0.0
                    )
                
                @sensor(modality=modality)
                def gripper_state(obs_cache):
                    return self.gripper_state
 
                @sensor(modality=modality)
                def pnp_obj_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.pnp_obj_body_id])

                @sensor(modality=modality)
                def pnp_obj_quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.pnp_obj_body_id]), to="xyzw")

                @sensor(modality=modality)
                def pnp_obj_pos_yaw(obs_cache):
                    pos = np.array(self.sim.data.body_xpos[self.pnp_obj_body_id])
                    quat = convert_quat(np.array(self.sim.data.body_xquat[self.pnp_obj_body_id]), to="xyzw")
                    return np.append(pos, quat2yaw(quat))

                @sensor(modality=modality)
                def push_obj_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.push_obj_body_id])

                @sensor(modality=modality)
                def push_obj_quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.push_obj_body_id]), to="xyzw")

                @sensor(modality=modality)
                def push_obj_pos_yaw(obs_cache):
                    pos = np.array(self.sim.data.body_xpos[self.push_obj_body_id])
                    quat = convert_quat(np.array(self.sim.data.body_xquat[self.push_obj_body_id]), to="xyzw")
                    return np.append(pos, quat2yaw(quat))

                sensors = [
                    eef_xyz, eef_yaw, gripper_state, 
                    pnp_obj_pos, pnp_obj_quat, pnp_obj_pos_yaw,
                    push_obj_pos, push_obj_quat, push_obj_pos_yaw,
                ]

                names = [s.__name__ for s in sensors]

                # Create observables
                for name, s in zip(names, sensors):
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )
    
            return observables
    
    def _update_keypoints(self):
        grasping_pnp = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.pnp_obj)
        pnp_obj_pos = np.array(self.sim.data.body_xpos[self.pnp_obj_body_id])
        push_obj_pos = np.array(self.sim.data.body_xpos[self.push_obj_body_id])

        if grasping_pnp: # holding pick and place object
            self.keypoints["pick"] = []
            self.keypoints["place"] = [np.array([-0.1, -0.1, 0.91])]
            self.keypoints["push"] = []

        elif self.in_bin(pnp_obj_pos):
            self.keypoints["pick"] = []
            self.keypoints["place"] = []
            self.keypoints["push"] = [push_obj_pos]
        
        else:
            self.keypoints["pick"] = [pnp_obj_pos]
            self.keypoints["place"] = []
            self.keypoints["push"] = [push_obj_pos]
        

    def step(self, action):

        self._update_keypoints()

        # if using primitive skills
        if self.use_skills:
            done, skill_done, skill_success = False, False, False
            obs = self.cur_obs

            if self.normalized_params:
                action = self._scale_params(action)
            
            num_timesteps = 0

            while not done and not skill_done:
                action_ll, skill_done, skill_success = self.skill.get_action(action, obs)
                obs, reward, done, info = super().step(action_ll)
                num_timesteps += 1
                if self.has_renderer:
                    self.render()
                self.gripper_state = action_ll[-1]
            self.cur_obs = obs

            info = {"num_timesteps": num_timesteps}

            # process rewards
            reward = self._reward()
            if self.use_aff_rewards:
                aff_penalty_factor = 1.0
                aff_reward = self.skill.compute_affordance_reward(action, self.keypoints)
                assert 0.0 <= aff_reward <= 1.0
                aff_penalty = 1.0 - aff_reward
                reward = reward - aff_penalty_factor * aff_penalty

            if reward > 0 and not skill_success:
                print("Reward earned on accident... Setting reward = 0")
                reward = 0.0
            
            info = {"num_timesteps": num_timesteps}
            
            if self._check_success(): # check if termination condition (success) is met
                done = True

            return self.cur_obs, reward, done, info 

        ###### when using low level actions ######
        # if input action dimension is 5, input is assumed to be [x, y, z, yaw, gripper]
        if action.shape[0] == 5:
            action = np.concatenate([action[:3], np.zeros(2), action[3:]])

        # update gripper state
        self.gripper_state = action[-1]

        # if end effector position is off the table, ignore the action
        action_in_bounds = self._check_action_in_bounds(action)
        if not action_in_bounds:
            action[:-1] = 0
            print("Action out of bounds")
        
        return super().step(action)

    def _check_action_in_bounds(self, action):

        sf = 3 # safety factor to prevent robot from moving out of bounds
        x_in_bounds = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_bounds = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        z_in_bounds = self.workspace_z[0] < self._eef_xpos[2] + sf * action[2] / self.control_freq < self.workspace_z[1]
        return x_in_bounds and y_in_bounds and z_in_bounds

    def _check_success(self):
        """
        returns True if pnp_obj is in bin and push_obj is in target region
        """
        if self.in_bin(self.sim.data.body_xpos[self.pnp_obj_body_id]) and self.in_push_target(self.sim.data.body_xpos[self.push_obj_body_id]):
            return True
        return False

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task completion

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if task is success
        if self._check_success() and not self.use_skills:
            terminated = True

        return terminated

    def _get_info_pnp(self, obj_id=0):
        pnp_obj_pos = self.sim.data.body_xpos[self.pnp_obj_body_ids[obj_id]]
        pnp_obj = self.pnp_objs[obj_id]

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        reach_dist = np.linalg.norm(gripper_site_pos - pnp_obj_pos)
        reached = reach_dist < 0.06

        grasping_cube = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=pnp_obj)
        if grasping_cube:
            grasped = True
        else:
            grasped = False

        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        hovering = (abs(pnp_obj_pos[0] - bin_pos[0]) < 0.10 and abs(pnp_obj_pos[1] - bin_pos[1]) < 0.15)

        in_bin = self.in_bin(pnp_obj_pos)

        return reached, grasped, hovering, in_bin

    def _get_info_push(self, obj_id=0):
        
        push_obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[obj_id]]

        target_pos_xy = self.table_offset[:2] + np.array([-0.15, 0.15])
        d_push = np.linalg.norm(push_obj_pos[:2] - target_pos_xy)

        pushed = (d_push <= 0.10)

        return pushed

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, _, info = super()._post_action(action)

        # allow episode to finish early if allowed
        self.done = self.done or self._check_terminated()

        return reward, self.done, info

    def reset(self):
        print("Resetting....")
        observation = super().reset()
        self.reward_given = False
        self.cur_obs = observation

        return observation

    def _scale_params(self, action):
        """
        Scales normalized parameters ([-1, 1]) to appropriate raw values
        """
        action = np.copy(action)
        params = action[self.num_skills:]

        params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_x[1] - self.workspace_x[0]) ) + self.workspace_x[0]
        params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_y[1] - self.workspace_y[0]) ) + self.workspace_y[0]
        params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_z[1] - self.workspace_z[0]) ) + self.workspace_z[0]
        
        if action[2] > 0: # action is push
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_x[1] - self.workspace_x[0]) ) + self.workspace_x[0]
            params[4] = ( ((params[4] + 1) / 2 ) * (self.workspace_y[1] - self.workspace_y[0]) ) + self.workspace_y[0]
            params[5] = ( ((params[5] + 1) / 2 ) * (self.workspace_z[1] - self.workspace_z[0]) ) + self.workspace_z[0]
            params[6] = ( ((params[6] + 1) / 2 ) * (self.yaw_bounds[1] - self.yaw_bounds[0]) ) + self.yaw_bounds[0]

        else: # action is pick or place
            params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_bounds[1] - self.yaw_bounds[0]) ) + self.yaw_bounds[0]

        return np.concatenate([action[:self.num_skills], params])

