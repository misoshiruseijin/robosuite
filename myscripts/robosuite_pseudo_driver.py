import robosuite as suite
import numpy as np
import redis
from robosuite import load_controller_config
from robosuite.devices import Keyboard, SpaceMouse
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper, DomainRandomizationWrapper
from robosuite.utils.primitive_skills import PrimitiveSkill
import pdb
import time

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
        env.modify_observable(observable_name="robot0_joint_pos", attribute="active", modifier=True)
        obs = env.reset()
        # rendering setup
        env.render()

        # Initialize device control
        device.start_control()
        steps = 0

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
            steps += 1

            for ob in obs_to_print:
                print(f"{ob}: {obs[ob]}")
            print("steps", steps)
            # print("Reward ", reward)
            # print("action ", action[3:-1])
            # print("yaw", _quat_to_yaw(obs[f"robot0_eef_quat"]))
            # print("Done ", done)

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

    # env.render()
    # for _ in range(25):
    #     action = 0.05 * np.random.uniform(-1, 1, 6)
    #     action = np.append(action, 1)
    #     env.step(action)
    #     env.render()

    # obs = env.reset()
    # start_time = time.time()
    # while time.time() - start_time < 10:
    #     action = np.array([0, 0, 0, 0, 0, 0, -1])
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    # print(env.steps)

    # while True:
    #     obs = env.reset()
    #     env.render()
    #     while True: 
    #         action = np.array([0, 0, 0, 0, 0, 0, -1])
    #         obs, reward, done, info = env.step(action)
    #         # pdb.set_trace()
    #         print("eef pos ", obs["robot0_eef_pos"])
    #         # print("CubeA pos ", obs["cubeA_pos"])
    #         # print("CubeB pos ", obs["cubeB_pos"])
    #         env.render()
    

if __name__ == "__main__":

    from robosuite.utils.primitive_skills import _wrap_to_pi, _quat_to_yaw

    # env = suite.make(
    #     env_name="LiftFlash",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     use_camera_obs=False,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     ignore_done=True,
    #     render_camera="frontview2",
    #     camera_names="frontview2",
    # )
    # obs = env.reset()
    # env.render()

    # spacemouse_control(env, obs_to_print=[])

    # p = PrimitiveSkill()
    # skill_done = False
    # one_hot = np.array([1, 0, 0, 0, 0, 0])
    # while not skill_done:
    #     action, skill_done = p.get_action(obs=obs, action=np.concatenate([one_hot, np.array([0, -0.1, 0.9, 0.1, -1])]))
    #     obs, reward, done, info = env.step(action)
    #     env.render()

    # p.pick(
    #     obs=obs,
    #     goal_pos=obs["cube_pos"],
    #     speed=0.8
    # )
    # p.place(
    #     obs=obs,
    #     goal_pos=(0, 0, env.table_offset[2]+0.035),
    #     speed=0.8
    # )

    # env = suite.make(
    #     env_name="POCReaching",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     initialization_noise=None,
    #     table_full_size=(0.65, 0.8, 0.15),
    #     table_friction=(100, 100, 100),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     ignore_done=True,
    #     camera_names="frontview",
    #     random_init=False,
    #     use_skills=True,
    # )

    # # spacemouse_control(env)
    # obs = env.reset()
    # env.render()
    # targetA_pos = obs["targetA_pos"]
    # targetB_pos = obs["targetB_pos"]

    # # move to target A
    # one_hot = np.array([1, 0])
    # param = np.array([targetA_pos[0], targetA_pos[1], 0, 1])
    # obs, reward, done, info = env.step(np.concatenate([one_hot, param]))
    # # gripper release
    # obs, reward, done, info = env.step(np.array([0, 1]))
    # # move to target B
    # one_hot = np.array([1, 0])
    # param = np.array([targetB_pos[0], targetB_pos[1], 0, -1])
    # obs, reward, done, info = env.step(np.concatenate([one_hot, param]))
    # action = np.concatenate([one_hot,  param])
    # obs, reward, done, info = env.step(action)
    # pdb.set_trace()
    # env.modify_observable(observable_name=f"robot0_joint_pos", attribute="active", modifier=True)
    # obs = env.reset()

    # p = PrimitiveSkill(env)
    # obs, reward, done, info = p.move_to_pos_xy(
    #     obs=obs,
    #     goal_pos=(0, 0.2),
    #     gripper_closed=True,
    #     wrist_ori=0.2,
    #     thresh=0.001
    # )

    # env = suite.make(
    #     env_name="Cleanup",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     initialization_noise="default",
    #     table_full_size=(0.8, 0.8, 0.05),
    #     table_friction=(1., 5e-3, 1e-4),
    #     table_offset=(0, 0, 0.8),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     reward_shaping=False,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     ignore_done=True,
    #     hard_reset=True,
    #     camera_names="agentview",
    #     task_config=None,
    # )
    # env.render()
    # pdb.set_trace()
    # spacemouse_control(env, obs_to_print=["robot0_eef_pos", "pnp_obj_pos", "pnp_obj_quat", "push_obj_pos", "push_obj_quat"], gripper_closed=False, indicator_on=True)


    # env = suite.make(
    #     env_name="StackCustom",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     initialization_noise="default",
    #     table_full_size=(0.8, 0.8, 0.05),
    #     table_friction=(1.0, 5e-3, 1e-4),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     placement_initializer=None,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     ignore_done=True,
    #     hard_reset=True,
    #     camera_names="agentview",
    #     camera_heights=512,
    #     camera_widths=512,
    # )
    # env.modify_observable(observable_name=f"robot0_joint_pos", attribute="active", modifier=True)
    # pdb.set_trace()
    # obs = env.reset()
    # env.render()
    # pdb.set_trace()
    # p = PrimitiveSkill(env)
    # obs, reward, done, info = p.move_to_pos(
    #     obs=obs,
    #     goal_pos=(0, 0, obs["robot0_eef_pos"][2]+0.1),
    #     gripper_closed=True,
    #     wrist_ori=0,
    #     # thresh=0.005
    # )
    # obs, reward, done, info = p.gripper_release()
    # obs, reward, done, info = p.push(
    #     obs=obs,
    #     start_pos=obs["cubeA_pos"],# + np.array([-0.05, -0.05, 0]),
    #     end_pos=obs["cubeA_pos"] + np.array([0.05, 0.05, 0]),
    #     wrist_ori=None,
    # )
    # obs, rewad, done, info = p.pick(
    #     obs=obs,
    #     goal_pos=obs["cubeA_pos"],
    #     wrist_ori=0.5*np.pi,
    # )
    # obs, rewad, done, info = p.place(
    #     obs=obs,
    #     goal_pos=(0,0,0.9),
    #     wrist_ori=-0.5*np.pi,
    # )


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
    #     hard_reset=True,
    #     camera_names="frontview",
    #     target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
    #     target_position=(0.0, 0.0), # target position (height above the table)
    #     random_init=True,
    #     random_target=False,
    # )
    # spacemouse_control(env)

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
    # pdb.set_trace()
    # prim.open_drawer(obs, env.obj_body_id["drawer"], pull_dist=0.1)
    # pdb.set_trace()
    # prim.close_drawer(obs, env.obj_body_id["drawer"], pull_dist=0.05)
    # pdb.set_trace()

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

    env = suite.make(
        env_name="Lift2",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True,
        use_skills=True,
    )
    obs = env.reset()
    one_hot = np.array([0, 1, 0])
    params = np.append(obs["cube_pos"], 0)
    action = np.concatenate([one_hot, params])
    obs, reward, done, info = env.step(action)

    one_hot = np.array([0, 0, 1])
    params = np.array([0, 0, 0.9, 0])
    action = np.concatenate([one_hot, params])
    obs, reward, done, info = env.step(action)
    
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

