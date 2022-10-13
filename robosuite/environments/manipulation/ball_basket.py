from collections import OrderedDict
from pkgutil import ModuleInfo
from random import random

import numpy as np

import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BasketHoopObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class BallBasket(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

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

        prehensile (bool): If true, ball object starts on the table. Else, the object starts in robot's gripper

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

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

        control_freq (float): many control signals to receive in every second. This sets the amount of
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

        basket_half_size (3-tuple): x, y, z half size of basket object

        stage_height (float): height of stage the basket is placed on
        
        fix_basket_pos (2-tuple): If given, trash can object is intialized at this position (x, y) with no random placement sampling

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
        table_full_size=(1.2, 0.60, 0.05),
        table_friction=(100,100,100),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        prehensile=True, # start with ball in gripper if False
        random_init=True, # start in rondom position (somewhere inside sphere around home position)
    ):

        # task setting: start ball in hand or not
        self.prehensile = prehensile

        # random initialization or not
        self.random_init = random_init

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.75))

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # hoop size and position
        self.hoop_half_size = None
        self.fix_hoop_pos = np.array([0, 0])

        # gripper state
        self.gripper_state = 1 # 1 = closed, -1 = open

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


    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the ball ends up in the basket

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 10

        # penalty if ball is dropped outside of hoop
        # elif self._check_ball_dropped():
        #     reward = -1

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
        xpos += np.array([0, -0.159, 0]) # eef starts at [-0.315, 0.0, 1,0]
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
        self.ball = BallObject(
            name="ball",
            size_min=[0.025],
            size_max=[0.0275],  
            rgba=[1, 0, 0, 1],
            friction=150,
        )

        self.hoop = BasketHoopObject(
            name="hoop",
            # body_half_size=self.hoop_half_size,
            rgba=(0, 0.5, 0.5, 1),
            density=100,
            thickness=0.01,
        )
        self.hoop_half_size = self.hoop.body_half_size

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([self.ball, self.hoop])

            
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.ball, self.hoop],
                x_range=[-0.3, 0.3],
                y_range=[-0.3, 0.3],
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
            mujoco_objects=[self.ball, self.hoop],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)
        self.hoop_body_id = self.sim.model.body_name2id(self.hoop.root_body)

    def _setup_observables(self): ### TODO ###
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
            def eef_xyz_gripper(obs_cache):
                return (
                    np.append(obs_cache[f"{pf}eef_pos"], self.gripper_state)
                    if f"{pf}eef_pos" in obs_cache
                    else np.zeros(4)
                )

            # @sensor(modality=modality)
            # def basket_pos(obs_cache):
            #     return np.array(self.sim.data.body_xpos[self.basket_body_id])

            # @sensor(modality=modality)
            # def gripper_to_basket_pos(obs_cache):
            #     return (
            #         obs_cache[f"{pf}eef_pos"] - obs_cache["basket_pos"]
            #         if f"{pf}eef_pos" in obs_cache and "basket_pos" in obs_cache
            #         else np.zeros(3)
            #     )

            sensors = [eef_xyz_gripper]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
 

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # basket hoop position (fixed)
            hoop_xpos = np.append(np.array([self.fix_hoop_pos]), self.table_offset[2] + self.hoop_half_size[2])
            self.sim.data.set_joint_qpos( self.hoop.joints[0], np.concatenate([hoop_xpos, np.array([0, 0, 0, 1])]) )

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                
                # place ball object
                if obj.name == "ball":
                    if self.prehensile:
                        # randomly place object on table
                        self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                    else:
                        # place ball in gripper
                        eef_rot_quat = T.mat2quat(T.euler2mat([np.pi - T.mat2euler(self._eef_xmat)[2], 0, 0]))
                        obj_quat = T.quat_multiply(obj_quat, eef_rot_quat)

                        # take a few steps to ensure that the ball is gripped
                        for j in range(100):
                            # Set object in hand
                            self.sim.data.set_joint_qpos(
                                obj.joints[0], np.concatenate([self._eef_xpos, np.array(obj_quat)])
                            )

                            # Execute no-op action with gravity compensation
                            self.sim.data.ctrl[self.robots[0]._ref_joint_actuator_indexes] = self.robots[0].controller.torque_compensation

                            # Execute gripper action
                            self.robots[0].grip_action(gripper=self.robots[0].gripper, gripper_action=[1])
                            
                            # Take forward step
                            self.sim.step()

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

        # Visualize gripper position (projection onto table)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.hoop)
    
    def _check_ball_above_hoop(self): 

        ball_pos = self.sim.data.body_xpos[self.ball_body_id]
        hoop_pos = self.sim.data.body_xpos[self.hoop_body_id] + np.array([0, 0, self.hoop_half_size[2]])
        hoop_size = self.hoop.hoop_half_size

        return (hoop_pos[0] - hoop_size[0] < ball_pos[0] < hoop_pos[0] + hoop_size[0]
                and hoop_pos[1] - hoop_size[1] < ball_pos[1] < hoop_pos[1] + hoop_size[1]
                and ball_pos[2] > hoop_pos[2])

    def _check_grasping_ball(self):
        
        grasping_ball = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self.ball
        )

        return grasping_ball

    def _check_success(self):
        """
        Check if ball has been placed into basket.

        Returns:
            bool: True if ball is in basket
        """
        return self._check_ball_above_hoop() and not self._check_grasping_ball()
    
    def _check_ball_dropped(self):

        return not (self._check_grasping_ball()) and not (self._check_ball_above_hoop())

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if arm contacts any other object including table
        if self.check_contact(self.robots[0].robot_model):
            print("arm collision")
            return True

        # Prematurely terminate if contacting the table with the gripper - No need (boundary prevents gripper-table contact)

        # Prematurely terminate if gripper collides with hoop
        if self.check_contact(self.robots[0].gripper, self.hoop):
            print("gripper collision: hoop")
            return True

        # Prematurely terminate if task is success
        if self._check_success():
            print("success")
            return True

        # Prematurely terminate if ball is dropped outside of basket
        if self._check_ball_dropped():
            print("ball dropped")
            return True

        return False

    def step(self, action):

        """
        Setup
        - Table dimensions: 1.2 x 0.6 x 0.75 [m]
        - World xy origin is at center of the table
        - Hoop is placed at [0, 0, 0.4 + table_height], where table height = 0.75 [m]
        
        Workspace Limits:
        - x: (-0.6, 0.1) from edge of table to further edge of hoop
        - y: (-0.3, 0.3) table edges
        - z: (table_height + 0.12, table_height + 0.9) to avoid gripper-table collision

        No Entry Zone Bounds (eef cannot enter this zone to avoid robot-hoop collision):
        - x: (-0.185, 0.1)
        - y: (-0.157, 0.157)
        - z: (table_height + 0.12, 1.19)

        Step function ignores any action that send the eef out of the workspace or into the No Entry Zone with a safety factor of 10
        """
        
        # print("action", action)
        # safety factor - increase this to make robot "more cautious" of boundaries
        sf = 0.07
        
        # workspace limits
        ws_x = (-0.6, 0.20)
        ws_y = (-0.3, 0.3)
        ws_z = (self.table_offset[2] + 0.05, self.table_offset[2] + 0.9)

        # no entry zone boundaries
        ne_x = (-0.170, 0.20)
        ne_y = (-0.16, 0.16)
        ne_z = (self.table_offset[2] + 0.05, self.table_offset[2] + 2*self.hoop_half_size[2] + 0.03)

        x, y, z = self._eef_xpos + (sf * action[:-1])

        # check workspace limits (True = action DOES NOT send eef out of bounds)
        in_ws = (ws_x[0] < x < ws_x[1]) and (ws_y[0] < y < ws_y[1]) and (ws_z[0] < z < ws_z[1])

        # check no entry zone violation (True = action DOES send eef into no entry zone)
        in_ne = (ne_x[0] < x < ne_x[1]) and (ne_y[0] < y < ne_y[1]) and (ne_z[0] < z < ne_z[1])

        # ignore action?
        if not in_ws or in_ne:
            # set dx, dy, dz = 0
            # print(f"ignoring action: in_ws = {in_ws}, in_ne = {in_ne}")
            action[:-1] = 0

        # update gripper state 
        if action[-1] < 0:
            self.gripper_state = -1
        else:
            self.gripper_state = 1

        return super().step(action)

    def reset(self):

        observations = super().reset()

        if not self.random_init:
            # if random initialization is not requested, return observation from parent function
            return observations

        # otherwise, apply random initialization
        # random initialization in 12.5 [cm] radius sphere centered around home
        noise_strength = 0.05
        for i in range(50):
            noise = noise_strength * np.random.uniform(-1, 1, 3)
            action = np.array(np.append(noise, 1))
            observations, reward, done, info = self.step(action)

        return observations

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
            reward, done, info = super()._post_action(action)

            # allow episode to finish early if allowed
            done = done or self._check_terminated()

            return reward, done, info


