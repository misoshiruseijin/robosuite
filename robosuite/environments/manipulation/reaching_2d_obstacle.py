import numpy as np

import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BlockObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.controllers import load_controller_config
import pdb
import time

class Reaching2DObstacle(SingleArmEnv):
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

        target_half_size (2-tuple): half size of target area

        target_position (2-tuple): position of target area

        
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        # table_full_size=(0.65, 0.65, 0.15),
        table_full_size=(0.65, 0.8, 0.15),
        table_friction=(100, 100, 100),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        target_placement_initializer=None,
        obstacle_placement_initializer=None,
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
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
        target_position=(0.0, 0.0), # target position
        obstacle_half_size=(0.025, 0.025, 0.15), # obstacle size
        random_init=True,
        random_target=True,
    ):

        print("INIT")

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.85))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # target
        self.target_half_size = target_half_size

        # obstacle
        self.obstacle_half_size = obstacle_half_size

        # initial eef position
        self.initial_eef_pos = None

        # whether to use random eef position and target position
        self.random_init = random_init
        self.random_target = random_target

        # workspace boundaries
        self.workspace_x = (-0.2, 0.2)
        self.workspace_y = (-0.35, 0.35)
        self.workspace_z = (0.83, 1.3)

        # placement initializer
        self.target_placement_initializer = target_placement_initializer
        self.obstacle_placement_initializer = obstacle_placement_initializer

        # initial z position
        self.initial_z = None

        self.reset_ready = False # hack to fix target initialized in wrong position issue
        # self.loadmodel_ready = False

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


    def reward(self, action=None): ### TODO ###
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the gripper ends up in the target region

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

        # collision negative reward
        if self._check_collision():
            reward = -10
        
        # # Scale reward if requested
        # if self.reward_scale is not None:
        #     reward *= self.reward_scale / 10

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """

        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos += np.array([-0.15, 0, 0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        # Initialize target object
        self.target = BlockObject(
            name="target",
            body_half_size=self.target_half_size,
            rgba=(1,0,0,1),
        )

        # Initialize obstacle object
        self.obstacle = BlockObject(
            name="obstacle",
            body_half_size=self.obstacle_half_size,
            density=500,
            rgba=(0,0,1,1)
        )

        # create placement initializer
        if self.target_placement_initializer is not None:
            self.target_placement_initializer.reset()
            self.target_placement_initializer.add_objects([self.target])
        else:
            self.target_placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.target],
                x_range=[-0.15, 0.15],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001
            )
        if self.obstacle_placement_initializer is not None:
            self.obstacle_placement_initializer.reset()
            self.obstacle_placement_initializer.add_objects([self.obstacle])
        else:
            self.obstacle_placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.obstacle],
                x_range=[-0.1, 0.1],
                y_range=[-0.25, 0.25],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.target, self.obstacle],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.target_body_id = self.sim.model.body_name2id(self.target.root_body)
        self.obstacle_body_id = self.sim.model.body_name2id(self.obstacle.root_body)

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
                def target_pos(obs_cache):
                    return self.sim.data.get_joint_qpos(self.target.joints[0])

                @sensor(modality=modality)
                def obstacle_pos(obs_cache):
                    return self.sim.data.get_joint_qpos(self.obstacle.joints[0])

                sensors = [target_pos, obstacle_pos]
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
        # print("START reset internal")
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.initial_z = self._eef_xpos[2]

        # reset all object position using placement initializer
        if not self.deterministic_reset:
            # sample from placement initializer for all objects
            target_placement = self.target_placement_initializer.sample()
            obstacle_placement = self.obstacle_placement_initializer.sample()

            # reset position of all objects
            obj_pos, obj_quat, obj = target_placement["target"]
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array([0, 0, 0, 1])])) # no rotations

            obj_pos, obj_quat, obj = obstacle_placement["obstacle"]
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array([0, 0, 0, 1])])) # no rotations

        # target_xpos = np.concatenate((self.target_position, np.array([self.table_offset[2]+self.target_half_size[2], 0, 0, 0, 1])))
        # # target_xpos[2] = self.table_offset[2] + self.target.base_half_size[1]
        # self.sim.data.set_joint_qpos(self.target.joints[0], target_xpos)

        # self.reset_ready = True

        # print("TARGET PLACED AT: ", self.target_position)
        # print("END reset internal")

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.target)

    def _check_success(self):
        """
        Check if ball has been placed into basket.;

        Returns:
            bool: True if ball is in basket
        """
       
        ##### Success = center point of end effector is within target cylinder #####
        success = self._check_in_region(
            region_center = self.sim.data.get_joint_qpos(self.target.joints[0]),
            region_bounds = self.target_half_size,
            coord = self._eef_xpos
        )

        return success

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task completion

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if task is success
        if self._check_success():
            print("~~~~~~~~in target~~~~~~~~~~~~~~")
            terminated = True

        # Prematurely terminate if collision occurs
        if self._check_collision():
            print("!!! collision detected !!!")
            terminated = True

        return terminated
    
    def _check_in_region(self, region_center, region_bounds, coord):
        """
        Check if input coordinate is inside the target

        Args
            region_center (array): 3d coordiante of region center
            region_bounds (array): defines bounds of region
                box (half size)
            coord (array): 2d coordinate to check
        """
        in_x = region_center[0] - region_bounds[0] < coord[0] < region_center[0] + region_bounds[0]
        in_y = region_center[1] - region_bounds[1] < coord[1] < region_center[1] + region_bounds[1]
        return in_x and in_y

    def _check_action_in_bounds(self, action):

        sf = 2 # safety factor to prevent robot from moving out of bounds
        x_in_bounds = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_bounds = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        return x_in_bounds and y_in_bounds

    def _check_collision(self):
        # Arm contacts any other object including table
        if self.check_contact(self.robots[0].robot_model):
            # print("arm collision")
            return True

        # Prematurely terminate if gripper collides with obstacle
        if self.check_contact(self.robots[0].gripper, self.obstacle):
            # print("gripper collision: hoop")
            return True

    def step(self, action):

        action_in_bounds = self._check_action_in_bounds(action)

        # ignore orientation inputs
        action[3:6] = 0

        # override z action to maintain constant height above table
        action[2] = self._z_correction()

        # ignore gripper inputs
        action[-1] = -1

        # if end effector position is off the table, ignore the action
        if not action_in_bounds:
            action[:-1] = 0
            print("Action out of bounds")
        
        # print("eefpos ", self._eef_xpos)
        # print("target ", self.target_position)

        return super().step(action)

    def step_no_count(self, action):
        """
        Modified version of step in base environment. Used for random initialization. 
        Returns observations, but actions taken using this step function does not affect the number of steps, time, etc.
        """

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False

        reward = 0
        done = False
        info = {}

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info
    
    def reset(self):
        print("Resetting....")
        observations = super().reset()

        # # wait for reset_internal to run
        # while not self.reset_ready:
        #     pass

        # if self.random_init:
        #     # sample random position inside workspace
        #     self.initial_eef_pos = np.concatenate((
        #         np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
        #         np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
        #     ))

        #     # sample again if start position is already inside or too close to the target
        #     thresh = (0.1, 0.1) # how far the starting position must be from target bounds (radius, height)
        #     while (self._check_in_region(self.target_position[:2], self.target_half_size[:2] + thresh, self.initial_eef_pos)):
        #         self.initial_eef_pos = np.concatenate((
        #             np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
        #             np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
        #         ))

        #     # move the eef to the sampled initial position
        #     thresh = 0.005
        #     while np.any(np.abs(self._eef_xpos[:2] - self.initial_eef_pos) > thresh):
        #         action = 4 * (self.initial_eef_pos - self._eef_xpos[:2]) / np.linalg.norm(self.initial_eef_pos - self._eef_xpos[:2])
        #         action = np.concatenate((action, np.array([0, -1])))
        #         observations, reward, done, info = self.step_no_count(action)
        #         # print("error to initial pos ", initial_pos - self._eef_xpos)
        
        # self.reset_ready = False
        # print(f"EEF position {self.initial_eef_pos} sampled based on target {self.target_position}")
        # print("End reset")
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

    def _z_correction(self):
        return 3 * (self.initial_z - self._eef_xpos[2])

    @property
    def target_pos(self):
        return self.sim.data.get_joint_qpos(self.target.joints[0])