from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, quat2yaw
from robosuite.utils.primitive_skills import PrimitiveSkillDelta, PrimitiveSkillGlobal

from robosuite.controllers.controller_factory import load_controller_config

import pdb

class StackCustom(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

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

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

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
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        horizon=200,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        use_skills=False,
        use_delta=None, # if set, ignore controller_configs and use osc controller (if True, use delta control, if false use global controller)
        normalized_params=True,
        use_aff_rewards=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # workspace boundaries
        self.workspace_x = (-0.2, 0.2)
        self.workspace_y = (-0.4, 0.4)
        self.workspace_z = (0.83, 1.3)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        # gripper state
        self.gripper_state = -1 

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
                }
            )
        else:
            self.skill = PrimitiveSkillDelta(
                skill_indices={
                    0 : "pick",
                    1 : "place",
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
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 10.0 is provided if the red block is stacked on the green block

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 10.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        if self._check_success() and not self.reward_given:
            reward = 10.0
            if not self.use_skills:
                self.reward_given = True
                print("~~~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~")            

        if self.reward_scale is not None:
            reward *= self.reward_scale / 10.0

        return reward

    def _reward(self, action=None): 
        """
        Reward function for the task. Used when using primitive skills

        Sparse un-normalized reward:

            - a discrete reward of 10.0 is provided if entire task is complete

        Note that the final reward is normalized and scaled by
        reward_scale / 10 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
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
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
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
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

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

            # eef observations
            @sensor(modality=modality)
            def eef_xyz(obs_cache):
                return (
                    np.array(obs_cache[f"{pf}eef_pos"])
                    if f"{pf}eef_pos" is obs_cache
                    else np.zeros(3)
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

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeA_pos_yaw(obs_cache):
                pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
                quat = convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")
                return np.append(pos, quat2yaw(quat))

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos_yaw(obs_cache):
                pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id])
                quat = convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")
                return np.append(pos, quat2yaw(quat))
                
            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return (
                    obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [
                eef_xyz, eef_yaw, gripper_state,
                cubeA_pos, cubeA_quat, cubeA_pos_yaw,
                cubeB_pos, cubeB_quat, cubeB_pos_yaw,
                gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB,
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

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        A_touching_B = self.check_contact(self.cubeA, self.cubeB)
        grasping_A = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
        cubeB_pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id])
        A_above_B = cubeA_pos[2] > cubeB_pos[2]

        if A_above_B and A_touching_B and not grasping_A:
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

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)
    
    def _check_action_in_bounds(self, action):

        sf = 3 # safety factor to prevent robot from moving out of bounds
        x_in_bounds = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_bounds = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        z_in_bounds = self.workspace_z[0] < self._eef_xpos[2] + sf * action[2] / self.control_freq < self.workspace_z[1]
        return x_in_bounds and y_in_bounds and z_in_bounds
    
    def _update_keypoints(self):
        grasping_A = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
        cubeB_pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id])
        place_pos = np.array([cubeB_pos[0], cubeB_pos[1], cubeB_pos[2] + 0.05])
        # if holding cube A, pick A is not relevant, and place on B becomes relevant
        if grasping_A:
            self.keypoints["pick"] = []
            self.keypoints["place"] = [place_pos]
        
        # if not holding cube A, pick A is relevant, place is irrelevant
        else:
            self.keypoints["pick"] = [cubeA_pos]
            self.keypoints["place"] = []


    def step(self, action):

        self._update_keypoints()
        
        # if using primitive skills
        if self.use_skills:
            done, skill_done, skill_success = False, False, False
            obs = self.cur_obs

            if self.normalized_params: # scale parameters if input params are normalized values
                action[self.num_skills:] = self._scale_params(action[self.num_skills:])
            
            num_timesteps = 0

            while not done and not skill_done:
                action_ll, skill_done, skill_success = self.skill.get_action(action, obs)
                obs, reward, done, info = super().step(action_ll)
                num_timesteps += 1
                if self.has_renderer:
                    self.render()
                self.gripper_state = action_ll[-1]
            self.cur_obs = obs
            
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

        ### when using low level actions
        # if input action dimension is 5, input is assumed to be [x, y, z, yaw, gripper]
        if action.shape[0] == 5:
            action = np.concatenate([action[:3], np.zeros(2), action[3:]])    
        
        # if end effector position is off the table, ignore the action
        action_in_bounds = self._check_action_in_bounds(action)
        if not action_in_bounds:
            action[:-1] = 0
            print("Action out of bounds")   
        
        self.gripper_state = action[-1] # update gripper state

        return super().step(action)

    def reset(self):
        print("Resetting....")
        observation = super().reset()
        self.reward_given = False
        self.cur_obs = observation

        return observation

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

    def _scale_params(self, params): # TODO - update after deciding which primitive skills to use
        """
        Scales normalized parameter ([-1, 1]) to appropriate raw values
        """
        params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_x[1] - self.workspace_x[0]) ) + self.workspace_x[0]
        params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_y[1] - self.workspace_y[0]) ) + self.workspace_y[0]
        params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_z[1] - self.workspace_z[0]) ) + self.workspace_z[0]
        params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_bounds[1] - self.yaw_bounds[0]) ) + self.yaw_bounds[0]

        return params