import robosuite as suite
import numpy as np
import redis
from robosuite import load_controller_config
from robosuite.devices import Keyboard, SpaceMouse
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.primitive_skills import PrimitiveSkill
import pdb
############### For BallBasket Environment #################
class BallBasketExperiment():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSITION"),
        action_dof=4,
        basket_half_size=(0.06, 0.06, 0.05), # size of rectangular basket
        stage_half_size=(0.07,0.07,0.1), # size of floating stage
        fix_basket_pos=(0.1, 0), # fixed position of basket on table (x, y)
    ):

        self.action_dof = action_dof # eef [dx, dy]

        # initialize environment
        self.env = suite.make(
            env_name="BallBasket",
            controller_configs=controller_config,
            robots="Sawyer",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="frontview",
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            prehensile=False, # whether to start the experiment with the ball in the gripper (False = start with ball in gripper)
            random_init=False,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()

        
    # Run Simulation - keyboard input mode
    def keyboard_input(self):
        # raise(NotImplementedError)

        # Initialize Device (keyboard)
        device = Keyboard(pos_sensitivity=0.3, rot_sensitivity=1.0)
        self.env.viewer.add_keypress_callback("any", device.on_press)
        self.env.viewer.add_keyup_callback("any", device.on_release)
        self.env.viewer.add_keyrepeat_callback("any", device.on_press)

        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            cam_id = 0
            num_cam = len(self.env.sim.model.camera_names)
            self.env.render()

            action = np.zeros(4)
            # action[-1] = 1

            # Initialize device control
            device.start_control()

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grasp = input2action(
                    device=device,
                    robot=active_robot,
                )

                # action = [x y z eef]

                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                # print(obs.keys())
                print("EE_POS_XYZ", obs['robot0_eef_pos'])
                # print("EE_ORIENTATION", self.env.robots[0]._hand_orn)
                # print("JOINT_CONFIG", obs["robot0_joint_pos"])
                # print("BASKET_POS", obs["basket_pos"])
                # print("REWARD", reward)

                # if done:
                #     print("------terminating--------")
                #     break

                self.env.render()


    # Run Simulation - Redis Mode
    def redis_control(self):
        
        # redis setup
        r = redis.Redis()

        # read keys
        self.ACTION_KEY = "action"
        
        while True:
            # Reset environment
            r.set("action", "0, 0, 0, 0") # reset action to neutral

            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            self.env.render()

            # change below while loop condition to terminate session after certain number of steps
            while True: 
                action = r.get(self.ACTION_KEY)
                action = str2ndarray(action, (self.action_dof, ))
                
                obs, reward, done, info = self.env.step(action)

                if done:
                    print("--- termination condition met ---")
                    break

                # print(obs)
                print(obs['eef_xyz_gripper'])
                # print("EE_POS_XYZ", obs['robot0_eef_pos']) # end effector cartesian position wrt world origin
                # print("REWARD", reward) # reward (1 when ball is inside basket and end effector is not gripping anything)

                self.env.render()

class SawyerReachingExperiment():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSITION"),
        action_dof=2,
        target_half_size=(0.08, 0.15),
        camera_view="agentview",
        random_init=True,
    ):

        self.action_dof = action_dof # eef [dx, dy]
        self.target_half_size = target_half_size

        # initialize environment
        self.env = suite.make(
            env_name="ReachingSawyerEF",
            controller_configs=controller_config,
            robots="Sawyer",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=camera_view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            target_half_size=self.target_half_size,
            random_init=random_init,
            lock_z=False,
        )


        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
        print(obs['robot0_eef_pos_xy'])

    ################# For Reaching Environment ###################
    # Run Simulation - Redis Mode
    def redis_control(self):
        """
        Runs simulation in redis mode.
        Reads action of form "dx, dy" from redis, and executes the action.
        Session terminates and restarts when one of the below termination conditions are met:
            - Arm collides with the table (should not happen since z is constant)
            - Joint limits are reached
            - Task is successfully completed
        """
        
        # redis setup
        r = redis.Redis()

        # read keys
        self.ACTION_KEY = "action"
        
        while True:
            # Reset environment
            r.set("action", "0, 0") # reset action to neutral

            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            self.env.render()

            # change below while loop condition to terminate session after certain number of steps
            while True: 
                action = r.get(self.ACTION_KEY)
                action = str2ndarray(action, (self.action_dof, ))
                action = np.append(action, np.array([0, -1]))
                
                obs, reward, done, info = self.env.step(action)
                if done:
                    print("--- termination condition met ---")
                    break

                print("EE_POS_XY", obs['robot0_eef_pos'][:-1]) # end effector cartesian position wrt robot base
                print("REWARD", reward) # reward (1 when ball is inside basket and end effector is not gripping anything)

                self.env.render()

    def keyboard_input(self):
        """
        Run simulation in keyboard input mode
        NOTE: This function is mostly for testing purposes.
        """

        # Initialize Device (keyboard)
        device = Keyboard(pos_sensitivity=0.3, rot_sensitivity=1.0)
        self.env.viewer.add_keypress_callback("any", device.on_press)
        self.env.viewer.add_keyup_callback("any", device.on_release)
        self.env.viewer.add_keyrepeat_callback("any", device.on_press)

        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )

                print("action", action)

                # action = [x y z eef]

                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                # print("EE_POS_XY", obs['robot0_eef_pos'][:-1])
                # print("REWARD", reward)
                # print(obs)
                # print("EEF_XY", obs['robot0_eef_pos_xy'])

                if done:
                    print("--------Termination Condition Met-------------")
                    break

                self.env.render()

    def test_step(self):

        action = np.array([0.05, 0, 0, 1])
        self.env.render()
        obs = self.env.reset()
        print("initial position", obs["robot0_eef_pos"])
        print("action", action)

        for i in range(20):
            obs, reward, done, info = self.env.step(action)
            # print("action", action)
            print(f"step {i}")
            print("eef pos", obs["robot0_eef_pos"])

        print("final position", obs['robot0_eef_pos'])

    
    def record_action_test(self, axis=0, n_steps=50, magnitude=0.05, back_scale=2):

        import imageio

        # Reset environment
        obs = self.env.reset()
        self.env.render()
        self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

        action_hist = []
        state_hist = []

        action = np.zeros(4)
        action[axis] = magnitude

        for i in range(n_steps):
            # generate random action in action space [0.1, 0.1]
            # action = np.random.uniform(-1, 1, 4)
            # action[:-1] = 0.1 * action[:-1]
            action_hist.append(action)

            obs, reward, _, _ = self.env.step(action)
            state_hist.append(obs['robot0_eef_pos'][:2])

            self.env.render()

        action = -action
        for i in range(int(back_scale*n_steps)):

            action_hist.append(action)

            obs, reward, _, _ = self.env.step(action)
            state_hist.append(obs['robot0_eef_pos'][:2])

            self.env.render()

        for i in range(2*n_steps):
            pass

        np.savetxt(f"actions{axis}.txt", np.array(action_hist), delimiter=',', newline='\n', fmt='%1.5f')
        np.savetxt(f"robosuite_states{axis}.txt", np.array(state_hist), delimiter=',', newline='\n', fmt="%1.5f")

class FrankaReachingExperiment():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSE"),
        target_half_size=(0.08, 0.15),
        camera_view="agentview",
        random_init=False,
        random_target=False,
    ):

        self.target_half_size = target_half_size

        # initialize environment
        self.env = suite.make(
            env_name="ReachingFrankaBC",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=camera_view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            # target_half_size=self.target_half_size,
            random_init=random_init,
            random_target=random_target,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)
        self.env.set_visualization_setting("grippers", False)
        obs = self.env.reset()

    ################# For Reaching Environment ###################
    # run simulation with spacemouse
    def spacemouse_control(self):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )

                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                print("eef_pos ", obs["robot0_eef_pos"])
                # print("delta ", obs["robot0_delta_to_target"])
                print("target ", self.env.target_position)
                self.env.render()

class Franka2DReachingExperiment():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSITION"),
        target_half_size=(0.05, 0.05, 0.001),
        camera_view="agentview",
        random_init=False,
        random_target=False,
    ):

        self.target_half_size = target_half_size

        # initialize environment
        self.env = suite.make(
            env_name="Reaching2D",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=camera_view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            target_half_size=self.target_half_size,
            random_init=random_init,
            random_target=random_target,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)
        # self.env.set_visualization_setting("grippers", False)
        obs = self.env.reset()

    ################# For Reaching Environment ###################
    # run simulation with spacemouse
    def spacemouse_control(self):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )
                action = action[:4]

                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                print("eef_pos ", obs["robot0_eef_pos"])
                # print("delta ", obs["robot0_delta_to_target"])
                print("target ", self.env.target_position)
                self.env.render()

class FrankaDropExperiment():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSITION"),
        action_dof=4,
        # basket_half_size=(0.06, 0.06, 0.05), # size of rectangular basket
        stage_half_size=(0.07,0.07,0.05), # size of floating stage
        fix_basket_pos=(0.1, 0), # fixed position of basket on table (x, y)
    ):

        self.action_dof = action_dof # eef [dx, dy]

        # initialize environment
        self.env = suite.make(
            env_name="Drop",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="frontview",
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            random_init=True,
            random_stage=True,
            stage_type="basket"
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
    
    
    # run simulation with spacemouse
    def spacemouse_control(self):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            done = False

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )
                action = action[:4]
                # action[-1] = 1
                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                print("action ", action)
                # print("eef_pos ", obs["robot0_eef_pos"])
                print("eef_pos ", obs["eef_xyz_gripper"])

                # print("delta ", obs["robot0_delta_to_target"])
                # print("target ", self.env.target_position)
                self.env.render()

class FrankaDataCollection():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSITION"),
        action_dof=4,
        initial_eef_pos=(0.0, 0.0, 0.0), # where on the table eef should start from
        obj_half_size=(0.025, 0.025, 0.025), # half size of object (cube)
        obj_rgba=(1.0,0.0,0.0,1.0), # object rgba 
        view="frontview",
    ):

        self.action_dof = action_dof # eef [dx, dy]

        # initialize environment
        self.env = suite.make(
            env_name="Object_and_Table",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            initial_eef_pos=initial_eef_pos, # where on the table eef should start from
            obj_half_size=(0.025, 0.025, 0.025), # half size of object (cube)
            obj_rgba=(1.0,0.0,0.0,1.0), # object rgba 
        )

        # self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
    
    
    # run simulation with spacemouse
    def spacemouse_control(self, gripper_closed=False):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            done = False

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )
                action = action[:4]
                if gripper_closed:
                    action[-1] = 1
                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                print("action ", action)
                # print("eef_pos ", obs["robot0_eef_pos"])
                print("eef_pos ", obs["eef_xyz_gripper"])

                # print("delta ", obs["robot0_delta_to_target"])
                # print("target ", self.env.target_position)
                self.env.render()
        
class FrankaLift():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSE"),
        action_dof=4,
        initial_eef_pos=(0.0, 0.0, 0.0), # where on the table eef should start from
        view="frontview",
    ):

        self.action_dof = action_dof # eef [dx, dy]

        # initialize environment
        self.env = suite.make(
            env_name="Lift2",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
    
    
    # run simulation with spacemouse
    def spacemouse_control(self):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            done = False

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )
                # action = action[:4]
                action[3:5] = 0
 
                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                print("action ", action)
                print("eef_pos ", obs["robot0_eef_pos"])
                # print("eef_pos ", obs["eef_xyz_gripper"])

                # print("delta ", obs["robot0_delta_to_target"])
                # print("target ", self.env.target_position)
                self.env.render()
   
    def primitive_control(self):
        
        primitive = PrimitiveSkill(self.env)

        # Reset environment
        obs = self.env.reset()
        # pdb.set_trace()
        # self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

        # rendering setup
        # self.env.render()

        primitive.move_to_pos(
            obs=obs,
            goal_pos=(0.16,0.16,0.988),
            gripper_closed=False,
            robot_id=0
        )                

class FrankaGridWall():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSE"),
        action_dof=4,
        initial_state=1,
        final_state=2,
        view="frontview",
    ):

        self.action_dof = action_dof # eef [dx, dy]
        self.initial_state = initial_state
        self.final_state = final_state

        # initialize environment
        self.env = suite.make(
            env_name="GridWall",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            obj_intial_abs_state=initial_state,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
    
    
    # run simulation with spacemouse
    def spacemouse_control(self):
        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

        device.start_control()
        while True:
            # Reset environment
            obs = self.env.reset()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            # Initialize device control
            device.start_control()

            done = False

            while True:
                # set active robot
                active_robot = self.env.robots[0]
                # get action
                action, grip = input2action(
                    device=device,
                    robot=active_robot,
                )
                # action = action[:4]
                action[3:5] = 0
 
                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                # print("action ", action)
                print("eef_pos ", obs["robot0_eef_pos"])
                # print("eef state ", obs["eef_abstract_state"])
                # print("obj state ", obs["obj_abstract_state"])

                self.env.render()
    
    def hardcode_control(self):
        obs = self.env.reset()
        eef_pos = obs["robot0_eef_pos"]
        obj_pos = obs["obj_pos"]
        lift_height = eef_pos[2]
        pick_place_height = self.env.table_offset[2] + self.env.obj_half_size[2]
        goal_pos = self.env.abstract_states[self.final_state]
        thresh = 0.005
        n_frames = 0
        lift_steps = 0
        stop_steps = 0
        gripper_cnt = 0 # buffer to make sure object is gripped

        phase = 1

        while True:
            # Reset environment
            obs = self.env.reset()
            # pdb.set_trace()
            self.env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

            # rendering setup
            self.env.render()

            done = False
            speed = 0.15

            while True:
                print("phase ", phase)
                # set active robot
                active_robot = self.env.robots[0]
                
                # get action
                if phase == 1: # reach to pick pos
                    print("obj pos ", obj_pos)
                    print("error ", np.abs(eef_pos[:2] - obj_pos[:2]))
                    u = (obj_pos[:2] - eef_pos[:2]) / np.linalg.norm(obj_pos[:2] - eef_pos[:2])
                    action = speed * u
                    action = np.array([action[0], action[1], 0, 0, 0, 0, -1])
                    if np.all(np.abs(eef_pos[:2] - obj_pos[:2]) < thresh):
                        phase = 2
                if phase == 2: # move down (open)
                    action = np.array([0, 0, -speed, 0, 0, 0, -1])
                    if np.abs(eef_pos[2] - pick_place_height) < thresh:
                        phase = 3
                if phase == 3: # grip
                    action = np.array([0, 0, 0, 0, 0, 0, 1])
                    gripper_cnt += 1
                    if gripper_cnt > 15:
                        gripper_cnt = 0
                        phase = 4
                if phase == 4: # move up (close)
                    action = np.array([0, 0, speed, 0, 0, 0, 1])
                    if np.abs(eef_pos[2] - lift_height) < thresh:
                        phase = 5
                if phase == 5: # reach to drop pos
                    u = (goal_pos[:2] - eef_pos[:2]) / np.linalg.norm(goal_pos[:2] - eef_pos[:2])
                    action = speed * u
                    action = np.array([action[0], action[1], 0, 0, 0, 0, 1])
                    if np.all(np.abs(eef_pos[:2] - goal_pos[:2]) < thresh):
                        phase = 6
                if phase == 6: # move down (close)
                    action = np.array([0, 0, -speed, 0, 0, 0, 1])
                    if abs(eef_pos[2] - pick_place_height) < thresh:
                        phase = 7
                if phase == 7: # drop
                    action = np.array([0, 0, 0, 0, 0, 0, -1])
                    gripper_cnt += 1
                    if gripper_cnt >= 15:
                        gripper_cnt = 0
                        phase = 8
                if phase == 8: # move up (open)
                    action = np.array([0, 0, speed, 0, 0, 0, -1])
                    if np.abs(eef_pos[2] - lift_height) < thresh:
                        break

                if action is None:
                    break

                # take step in simulation
                obs, reward, done, info = self.env.step(action)
                # print("action ", action)
                eef_pos = obs["robot0_eef_pos"]
                print("eef_pos ", eef_pos)
                # print("contact ", obs["robot0_contact"])

                self.env.render()

class FrankaDrawer():
    def __init__(
        self,
        controller_config=load_controller_config(default_controller="OSC_POSE"),
        action_dof=4,
        initial_eef_pos=(0.0, 0.0, 0.0), # where on the table eef should start from
        view="agentview",
    ):

        self.action_dof = action_dof # eef [dx, dy]

        # initialize environment
        self.env = suite.make(
            env_name="DrawerEnv",
            controller_configs=controller_config,
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=view,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
        )

        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        obs = self.env.reset()
        # pdb.set_trace()
    
def spacemouse_control(env, obs_to_print=["robot0_eef_pos"], indicator_on=True, gripper_closed=False):
    
    if indicator_on:
        env = VisualizationWrapper(env, indicator_configs=None)

    controller_type = env.robot_configs[0]["controller_config"]["type"]
    if controller_type == "OSC_POSE":
        action_dim = 6
    elif controller_type == "OSC_POSITION":
        action_dim = 3
    
    device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )

    device.start_control()
    while True:
        # Reset environment
        obs = env.reset()
        env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)

        # rendering setup
        env.render()

        # Initialize device control
        device.start_control()

        while True:
            # set active robot
            active_robot = env.robots[0]
            # get action
            action, grip = input2action(
                device=device,
                robot=active_robot,
            )

            if action is None:
                break

            if action_dim < 6:
                action = np.append(action[:action_dim], action[-1])

            if gripper_closed:
                action[-1] = 1

            # take step in simulation
            obs, reward, done, info = env.step(action)

            for ob in obs_to_print:
                print(f"{ob}: {obs[ob]}")

            env.render()


def str2ndarray(array_str, shape):
    """
    Helper function to convert action read from redis to np array
    takes input array_str of form "[[a, b, c],...]"
    returns nparray of specified shape
    """
    
    # array_str = array_str.translate(str.maketrans('','','[]')) # remove brackets
    output = np.fromstring(array_str, dtype=float, sep=',')
    output = output.reshape(shape)
    return output

def main():
    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # spacemouse_control(env, gripper_closed=True, indicator_on=False)
    # env.render()
    # for _ in range(25):
    #     action = 0.05 * np.random.uniform(-1, 1, 6)
    #     action = np.append(action, 1)
    #     env.step(action)
    #     env.render()

    while True:
        obs = env.reset()
        env.render()
        while True: 
            action = np.array([0, 0, 0, 0, 0, 0, -1])
            env.step(action)
            env.render()
    

if __name__ == "__main__":

    env = suite.make(
        env_name="Reaching2D",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        gripper_types="default",
        initialization_noise=None,
        table_full_size=(0.65, 0.8, 0.15),
        table_friction=(100, 100, 100),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,
        hard_reset=True,
        camera_names="frontview",
        target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
        target_position=(0.0, 0.0), # target position (height above the table)
        random_init=True,
        random_target=False,
    )



    # env = suite.make(
    #     env_name="Reaching2D",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     gripper_types="default",
    #     initialization_noise=None,
    #     table_full_size=(0.65, 0.8, 0.15),
    #     table_friction=(100, 100, 100),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     ignore_done=True,
    #     camera_names="frontview",
    #     target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
    #     target_position=(0.0, 0.0), # target position (height above the table)
    #     random_init=True,
    #     random_target=True,
    # )
    # spacemouse_control(env,)

    # env = suite.make(
    #     env_name="DrawerEnv",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     robots="Panda",
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     use_camera_obs=False,
    #     control_freq=20,
    #     ignore_done=True,
    # )
    # prim = PrimitiveSkill(env)
    # obs = env.reset()
    # env.render()
    # prim.open_drawer(obs, env.obj_body_id["drawer"], pull_dist=0.1)

    # env = suite.make(
    #     env_name="LeftRight",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     table_full_size=(0.8, 2.0, 0.05),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview2",
    #     ignore_done=True,
    #     camera_names="agentview",
    #     camera_heights=256,
    #     camera_widths=256,
    #     line_thickness=0.05,
    #     line_rgba=(0,0,0,1),
    #     ball_radius=0.04,
    #     ball_rgba=(1,0,0,1)
    # )

    # env = suite.make(
    #     env_name="PickPlacePrimitive",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     table_full_size=(0.65, 0.65, 0.15),
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     use_camera_obs=False,
    #     render_camera="frontview",
    #     ignore_done=True,
    #     camera_names="agentview",
    #     block_half_size=(0.025, 0.025, 0.025),
    #     plate_half_size=(0.05, 0.05, 0.02),
    #     block_rgba=(1,0,0,1),
    #     plate_rgba=(0,0,1,1),
    # )

    # env = suite.make(
    #     env_name="DrawerEnv",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     robots="Panda",
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     use_camera_obs=False,
    #     control_freq=20,
    #     ignore_done=True,
    # )

    # env = suite.make(
    #     env_name="Lift2",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     robots="Panda",
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     use_camera_obs=False,
    #     control_freq=20,
    #     ignore_done=True,
    # )

    # env = suite.make(
    #     env_name="Reaching2DObstacle",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     robots="Panda",
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="agentview2",
    #     use_camera_obs=False,
    #     control_freq=20,
    #     ignore_done=True,
    #     target_half_size=(0.05,0.05,0.001),
    #     obstacle_half_size=(0.025, 0.025, 0.15),
    #     random_init=True,
    #     random_target=True,
    # )

    main()

