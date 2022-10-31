from collections import OrderedDict
from pkgutil import ModuleInfo
from random import random

import numpy as np
from robosuite.models.objects.composite.stage import StageObject

import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BlockObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import pdb

class Drop(SingleArmEnv):
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

        prehensile (bool): If true, object starts on the table. Else, the object starts in robot's gripper

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
        table_full_size=(0.65, 0.65, 0.15),
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
        stage_half_size=(0.05, 0.05, 0.075),
        stage_pos=(0.0, 0.0, 0.15), # target position (height = table surface to stage top surface)
        prehensile=False, # start with obj in gripper if False * NOT USED
        random_init=True, # start in rondom position (somewhere inside sphere around home position)
        random_stage=True, # place stage at random location
    ):

        # task setting: start obj in hand or not
        self.prehensile = prehensile

        # random initialization or not
        self.random_init = random_init
        self.random_stage = random_stage

        # no entry zone (prevent collision with stage)
        self.no_entry_x = None
        self.no_entry_y = None
        self.no_entry_z = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.75))

        # workspace boundaries
        self.workspace_x = (-0.2 + self.table_offset[0], 0.2 + self.table_offset[0])
        self.workspace_y = (-0.3 + self.table_offset[1], 0.3 + self.table_offset[1])
        self.workspace_z = (self.table_offset[2] + 0.03, self.table_offset[2] + 0.5)

        # initial eef position
        self.initial_eef_pos = None

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # stage size and position
        self.stage_half_size = np.array(stage_half_size)
        self.stage_pos = stage_pos + self.table_offset - np.array([0, 0, self.stage_half_size[2]])

        # no entry zone around stage (avoid collision between eef and stage)
        buf = (0.055, 0.11, 0.03)
        self.ne_x = (self.stage_pos[0] - self.stage_half_size[0] - buf[0], self.stage_pos[0] + self.stage_half_size[0] + buf[1])
        self.ne_y = (self.stage_pos[1] - self.stage_half_size[1] - buf[1], self.stage_pos[1] + self.stage_half_size[1] + buf[1])
        self.ne_z = (self.table_offset[2], self.stage_pos[2] + self.stage_half_size[2] + buf[2])

        # gripper state
        self.gripper_state = 1 # 1 = closed, -1 = open

        self.reset_ready = False

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

            - a discrete reward of 2.25 is provided if the object ends up on the stage

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

        # initialize block object
        self.block = BlockObject(
            name="block",
            body_half_size=(0.025,0.025,0.025),
            rgba=(1,0,0,1),
            density=500
        )

        # initialize stage object
        if self.random_stage: # sample target position if using random target initialization
            self.stage_half_size[2] = np.random.uniform(0.05/2, 0.25/2, 1)[0]
            self.stage_pos = np.concatenate((
                np.random.uniform(self.workspace_x[0] + self.stage_half_size[0], self.workspace_x[1] - self.stage_half_size[0], 1),
                np.random.uniform(self.workspace_y[0] + self.stage_half_size[1], self.workspace_y[1] - self.stage_half_size[1], 1),
                np.array([self.table_offset[2] + self.stage_half_size[2]])
            ))
        self.stage = StageObject(
            name="stage",
            body_half_size=(self.stage_half_size[0], self.stage_half_size[1], self.stage_half_size[2]),
            rgba=(0,0,1,1)
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.block, self.stage],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.block_body_id = self.sim.model.body_name2id(self.block.root_body)
        self.stage_body_id = self.sim.model.body_name2id(self.stage.root_body)

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

            sensors = [eef_xyz_gripper]#, basket_pos]
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

        # stage position
        stage_xpos = self.stage_pos
        self.sim.data.set_joint_qpos( self.stage.joints[0], np.concatenate([stage_xpos, np.array([0, 0, 0, 1])]) )
        
        self.sim.data.set_joint_qpos(
            self.block.joints[0], np.array([-0.37, 0, self.table_offset[2]+0.15, 0, 0, 0, 1])
        )
        # take a few steps to grip the block
        for j in range(100):
            # place object in gripper
            self.sim.data.set_joint_qpos(
                self.block.joints[0], np.concatenate([self._eef_xpos, np.array([0, 0, 0, 1])])
            )
            # Execute no-op action with gravity compensation
            self.sim.data.ctrl[self.robots[0]._ref_joint_actuator_indexes] = self.robots[0].controller.torque_compensation

            # Execute gripper action
            self.robots[0].grip_action(gripper=self.robots[0].gripper, gripper_action=[1])
            
            # Take forward step
            self.sim.step()

        self.reset_ready = True

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.stage)
    
    def _check_obj_above_stage(self): 

        obj_pos = self.sim.data.body_xpos[self.block_body_id]
        stage_pos = self.sim.data.body_xpos[self.stage_body_id] + np.array([0, 0, self.stage_half_size[2]])
        stage_size = self.stage_half_size
        above_stage = (stage_pos[0] - stage_size[0] < obj_pos[0] < stage_pos[0] + stage_size[0]
                and stage_pos[1] - stage_size[1] < obj_pos[1] < stage_pos[1] + stage_size[1]
                and obj_pos[2] > stage_pos[2])
        # if above_stage:
        #     print("Above Stage")
        return above_stage

    def _check_grasping_obj(self):
        
        grasping_obj = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self.block
        )

        return grasping_obj

    def _check_success(self):
        """
        Check if obj has been placed on stage.

        Returns:
            bool: True if obj is placed on stage
        """
        return self._check_obj_above_stage() and not self._check_grasping_obj()
    
    def _check_obj_dropped(self):

        return not (self._check_grasping_obj()) and not (self._check_obj_above_stage())

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

        # Prematurely terminate if gripper collides with stage
        if self.check_contact(self.robots[0].gripper, self.stage):
            print("gripper collision: stage")
            return True

        # Prematurely terminate if task is success
        if self._check_success():
            # print("success")
            return True

        # Prematurely terminate if block is dropped off the stage
        if self._check_obj_dropped():
            # print("obj dropped")
            return True

        return False
  
    def _check_action_in_bounds(self, action):

        sf = 3 # safety factor to prevent robot from moving out of bounds

        # within workspace bounds?
        x_in_ws = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_ws = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        z_in_ws = self.workspace_z[0] < self._eef_xpos[2] + sf * action[2] / self.control_freq < self.workspace_z[1]
        in_ws = x_in_ws and y_in_ws and z_in_ws

        # in no-entry zone?
        x_in_ne = self.ne_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.ne_x[1]
        y_in_ne = self.ne_y[0] < self._eef_xpos[1] + sf * action[0] / self.control_freq < self.ne_y[1]
        z_in_ne = self.ne_z[0] < self._eef_xpos[2] + sf * action[0] / self.control_freq < self.ne_z[1]
        in_ne = x_in_ne and y_in_ne and z_in_ne

        # print(f"in_ws {in_ws}, in_ne {in_ne}")
        return in_ws and not in_ne

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
  
    def step(self, action):

        """ 
        No Entry Zone Bounds (eef cannot enter this zone to avoid robot-stage collision):

        Step function ignores any action that send the eef out of the workspace or into the No Entry Zone with a safety factor of 10
        """
        
        # print("action", action)
        # safety factor - increase this to make robot "more cautious" of boundaries
        sf = 0.035

        # # no entry zone boundaries
        buf = (0.055, 0.11, 0.03)
        self.ne_x = (self.stage_pos[0] - self.stage_half_size[0] - buf[0], self.stage_pos[0] + self.stage_half_size[0] + buf[1])
        self.ne_y = (self.stage_pos[1] - self.stage_half_size[1] - buf[1], self.stage_pos[1] + self.stage_half_size[1] + buf[1])
        self.ne_z = (self.table_offset[2], self.stage_pos[2] + self.stage_half_size[2] + buf[2])

        # ignore action?
        if not self._check_action_in_bounds(action):
            # set dx, dy, dz = 0
            print(f"Action ignored")
            action[:-1] = 0

        # update gripper state 
        if action[-1] < 0:
            self.gripper_state = -1
        else:
            self.gripper_state = 1

        return super().step(action)

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
        in_z = region_center[2] - region_bounds[2] < coord[2] < region_center[2] + region_bounds[2]
        return in_x and in_y and in_z

    def reset(self):

        observations = super().reset()

        while not self.reset_ready:
            pass

        if self.random_init:
            # sample random position inside workspace
            self.initial_eef_pos = np.concatenate((
                np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
                np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
                np.random.uniform(self.workspace_z[0], self.workspace_z[1], 1)
            ))

            # sample again if start position collides with stage or is too close 
            thresh = (0.05, 0.05, 0.0) # how far the starting position must be from target bounds
            buf = (0.055, 0.11, 0.03)
            # while (self._check_in_region(self.stage_pos, self.stage_half_size + buf, self.initial_eef_pos)):
            while (self._check_in_region(self.stage_pos, self.stage_half_size + buf, self.initial_eef_pos)):
                self.initial_eef_pos = np.concatenate((
                    np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
                    np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
                    np.random.uniform(self.workspace_z[0], self.workspace_z[1], 1),
                ))
                print("sampled pos: ", self.initial_eef_pos)

            # move the eef to the sampled initial position
            thresh = 0.01
            while np.any(np.abs(self._eef_xpos - self.initial_eef_pos) > thresh):
                action = 4 * (self.initial_eef_pos - self._eef_xpos) / np.linalg.norm(self.initial_eef_pos - self._eef_xpos)
                action = np.concatenate((action, np.array([1])))
                observations, reward, done, info = self.step_no_count(action)
                # print("error to initial pos ", initial_pos - self._eef_xpos)
        
        self.reset_ready = False
        print(f"EEF position {self.initial_eef_pos} sampled based on target {self.stage_pos}")
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

    @property
    def stage_top(self):
        # return coordinate of stage top surface center
        return self.stage_pos + np.array([0, 0, self.stage_half_size[2]])