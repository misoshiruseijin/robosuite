import robosuite as suite
import numpy as np
import redis
from robosuite import load_controller_config
from robosuite.devices import Keyboard, SpaceMouse
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import GymWrapper, VisualizationWrapper, DomainRandomizationWrapper
from robosuite.utils.primitive_skills import PrimitiveSkill
from robosuite.controllers.osc import OperationalSpaceController
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
        done = False
        # while True:
        while not done:
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
            # print("steps", steps)
            print("Reward ", reward)
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

    pose_controller_config = load_controller_config(default_controller="OSC_POSE")
    pose_controller_config["control_delta"] = False
    pos_controller_config = load_controller_config(default_controller="OSC_POSITION")
    pos_controller_config["control_delta"] = False

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
    # vis = False
    # while True:
    #     env.render()
    #     pdb.set_trace()

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
    #     use_skills=False,
    #     normalized_params=False,
    # )
    # obs = env.reset()

    # while True:
    #     obs = env.reset()
    #     env.render()

    #     targetA_pos = obs["targetA_pos"]
    #     targetB_pos = obs["targetB_pos"]
    #     height = obs["robot0_eef_pos"][2]

    #     eef_pos_0 = obs["robot0_eef_pos"][:2]
    #     u = (targetA_pos - eef_pos_0) / np.linalg.norm(targetA_pos - eef_pos_0)

    #     # move to target A
    #     one_hot = np.array([1., 0.])
    #     param = np.array([targetA_pos[0], targetA_pos[1], height, 0., 1.])
    #     # param = np.array([1.1*u[0], 1.1*u[1], height, 0., 1.])
    #     obs, reward, done, info = env.step(np.concatenate([one_hot, param]))
    #     pdb.set_trace()

    #     # # move out of target A
    #     # one_hot = np.array([1., 0.])
    #     # param = np.array([eef_pos_0[0], eef_pos_0[1], height, 0., 1.])
    #     # # param = np.array([1.1*u[0], 1.1*u[1], height, 0., 1.])
    #     # obs, reward, done, info = env.step(np.concatenate([one_hot, param]))
    #     # pdb.set_trace()

    #     # gripper release
    #     obs, reward, done, info = env.step(np.array([0, 1, 0, 0, 0, 0, 0]))
    #     pdb.set_trace()

    #     # move to target B
    #     param = np.array([targetB_pos[0], targetB_pos[1], height, 0., -1.])
    #     obs, reward, done, info = env.step(np.concatenate([one_hot, param]))
    #     action = np.concatenate([one_hot,  param])

    #     print("reward", reward)
    #     pdb.set_trace()


    # env = suite.make(
    #     env_name="Cleanup",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     reward_scale=1.0,
    #     initialization_noise=None,
    #     has_renderer=True,
    #     use_camera_obs=False,
    #     has_offscreen_renderer=False,
    #     ignore_done=False,
    #     use_skills=True,
    #     normalized_params=False,
    # )
    # obs = env.reset()
    # pnp_pos = obs["pnp_obj_pos"]
    # push_pos = obs["push_obj_pos"]
    # # pnp
    # action = np.array([1, 0, 0, pnp_pos[0], pnp_pos[1], pnp_pos[2], 0.])
    # obs, reward, done, info = env.step(action)
    # action = np.array([0, 1, 0, -0.1, -0.1, pnp_pos[2]+0.07, 0])
    # obs, reward, done, info = env.step(action)
    # # push
    # push_pos = obs["push_obj_pos"]
    # start_push_pos = np.array([push_pos[0], push_pos[1]-0.1, push_pos[2]+0.001])
    # end_push_pos = np.array([push_pos[0], 0.13, push_pos[2]+0.001])
    # if push_pos[1] < 0.1:
    #     action = np.concatenate([np.array([0, 0, 1]), start_push_pos, end_push_pos, np.array([0, 1])])
    #     # action = np.array([0, 0, 1, push_pos[0], push_pos[1]-0.1, push_pos[2]+0.005, 0.15, push_pos[1], push_pos[2]+0.005, 0, 1])
    #     obs, reward, done, info = env.step(action)
    #     push_pos = obs["push_obj_pos"]
    # action = np.array([0, 0, 1, push_pos[0]+0.1, push_pos[1], push_pos[2]+0.005, -0.12, 0.13, push_pos[2]+0.005, 0, 1])
    # obs, reward, done, info = env.step(action)
    # print("reward ", reward)
    # pdb.set_trace()


    # env = suite.make(
    #     env_name="StackCustom",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     # controller_configs=controller_config,
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     ignore_done=False,
    #     use_skills=False,
    #     normalized_params=False,
    #     control_freq=10,
    #     initialization_noise=None,
    # )

    # from robosuite.utils.transform_utils import *
    # obs = env.reset()
    # mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]) # what the rotation matrix should be
    # rot = quat2mat(obs["robot0_eef_quat"]) # rotation matrix corresponding to initial eef_quat
    # trans = np.linalg.solve(rot, mat) # transforms rot to mat : rot @ trans = mat
    # quat = obs["robot0_eef_quat"]
    # aa = quat2axisangle(mat2quat(mat))
    # print(aa)
    # print(quat)

    # eef_pos = obs["robot0_eef_pos"]
    # pdb.set_trace()
    # cubeA_pos = obs["cubeA_pos"]
    # cubeB_pos = obs["cubeB_pos"]

    # action = np.concatenate([np.array([1., 0.]), cubeA_pos, np.array([0.,0., 1])])
    # obs, reward, done, info = env.step(action)
    # # pdb.set_trace()
    # print(env.timestep)

    # action = np.concatenate([np.array([0., 1.]), cubeB_pos, np.array([0., 1])])
    # action[4] = action[4] + 0.03
    # obs, reward, done, info = env.step(action)
    # print(env.timestep)
    # pdb.set_trace()


    env = suite.make(
        env_name="Reaching2D",
        robots="Panda",
        # controller_configs=load_controller_config(default_controller="OSC_POSE"),
        controller_configs=pose_controller_config,
        reward_scale=1.0, # change this to any value
        use_skills=True,
        normalized_params=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=10,
        initialization_noise=None,
    )
    from robosuite.utils.transform_utils import *

    aas = np.array([
        [ 2.1551063 ,  2.1561687 ,  0.13730317 ],
        [ 2.138706 ,  2.1728508 ,  0.13781095 ],
        [ 2.116718 ,  2.1949935 ,  0.13874738 ],
        [ 2.0941348 ,  2.2175202 ,  0.14011 ],
        [ 2.070974 ,  2.2402024 ,  0.14169703 ],
        [ 2.047399 ,  2.2627926 ,  0.14342014 ],
        [ 2.023483 ,  2.2851899 ,  0.14522704 ],
        [ 1.9992777 ,  2.3073602 ,  0.14706703 ],
        [ 1.9748074 ,  2.3292835 ,  0.14891909 ],
        [ 1.9500854 ,  2.350954 ,  0.15077078 ],
        [ 1.9251182 ,  2.372365 ,  0.15261756 ],
        [ 1.8999119 ,  2.3935108 ,  0.15445565 ],
        [ 1.87447 ,  2.4143918 ,  0.15628238 ],
        [ 1.8487967 ,  2.435005 ,  0.15809536 ],
        [ 1.822895 ,  2.4553487 ,  0.15989266 ],
        [ 1.7967695 ,  2.4754221 ,  0.16167231 ],
        [ 1.7704229 ,  2.495224 ,  0.16343278 ],
        [ 1.7438593 ,  2.514752 ,  0.16517204 ],
        [ 1.7170827 ,  2.53401 ,  0.16688734 ],
        [ 1.6900983 ,  2.5529966 ,  0.16857544 ],
        [ 1.6629088 ,  2.5717165 ,  0.17023428 ],
        [ 1.6355186 ,  2.5901642 ,  0.17186284 ],
        [ 1.6079293 ,  2.608339 ,  0.17346075 ],
        [ 1.5801449 ,  2.6262398 ,  0.175028 ],
        [ 1.5521659 ,  2.6438642 ,  0.17656454 ],
        [ 1.5239967 ,  2.6612089 ,  0.17807044 ],
        [ 1.4956391 ,  2.6782708 ,  0.1795459 ],
        [ 1.4670955 ,  2.6950495 ,  0.18099095 ],
        [ 1.4383688 ,  2.711542 ,  0.18240574 ],
        [ 1.4094621 ,  2.7277448 ,  0.18379036 ],
        [ 1.3803768 ,  2.7436566 ,  0.18514492 ],
        [ 1.3511165 ,  2.7592752 ,  0.18646942 ],
        [ 1.3216844 ,  2.7745972 ,  0.18776399 ],
        [ 1.2920823 ,  2.7896216 ,  0.18902862 ],
        [ 1.2623136 ,  2.8043458 ,  0.19026345 ],
        [ 1.232381 ,  2.8187673 ,  0.19146842 ],
        [ 1.2022877 ,  2.8328843 ,  0.19264364 ],
        [ 1.172036 ,  2.8466938 ,  0.19378912 ],
        [ 1.1416293 ,  2.8601942 ,  0.19490494 ],
        [ 1.1110702 ,  2.8733847 ,  0.19599119 ],
        [ 1.0803622 ,  2.8862605 ,  0.19704781 ],
        [ 1.0495077 ,  2.8988223 ,  0.19807489 ],
        [ 1.0185096 ,  2.9110663 ,  0.199073 ],
    ])

    obs = env.reset()
    eef_pos = obs["robot0_eef_pos"]
    # for aa in aas:
    #     action = np.concatenate([eef_pos, aa, np.array([-1])])
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    # while True:
    #     action = np.array([0.1, 0.15, 1.0, 0, 0, 0.1, -1])
    #     obs, reward, done, info = env.step(action)
    #     env.render()

    obs = env.reset()
    goal_pos = np.array([0.1, 0.15, 1])
    action = np.array([1, goal_pos[0], goal_pos[1], goal_pos[2], -0.4*np.pi, -1])
    print("Action ", action)
    env.render()
    
    obs, reward, done, info = env.step(action)
    env.render()
    pdb.set_trace()

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
    # obs = env.reset()
    # env.render()
    # pdb.set_trace()
    # while True:
    #     env.render()

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
    #     use_skills=True,
    # )
    # obs = env.reset()
    # one_hot = np.array([0, 1, 0])
    # params = np.append(obs["cube_pos"], 0)
    # action = np.concatenate([one_hot, params])
    # obs, reward, done, info = env.step(action)

    # one_hot = np.array([0, 0, 1])
    # params = np.array([0, 0, 0.9, 0])
    # action = np.concatenate([one_hot, params])
    # obs, reward, done, info = env.step(action)
    
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

    # skill_config=dict(
    #     skills=['open', 'reach', 'grasp'],
    #     aff_penalty_fac=15.0,

    #     base_config=dict(
    #         global_xyz_bounds=[
    #             [-0.30, -0.30, 0.80],
    #             [0.15, 0.30, 0.95]
    #         ],
    #         lift_height=0.95,
    #         binary_gripper=True,

    #         aff_threshold=0.06,
    #         aff_type='dense',
    #         aff_tanh_scaling=10.0,
    #     ),
    #     atomic_config=dict(
    #         use_ori_params=True,
    #     ),
    #     reach_config=dict(
    #         use_gripper_params=False,
    #         local_xyz_scale=[0.0, 0.0, 0.06],
    #         use_ori_params=False,
    #         max_ac_calls=15,
    #     ),
    #     grasp_config=dict(
    #         global_xyz_bounds=[
    #             [-0.30, -0.30, 0.80],
    #             [0.15, 0.30, 0.85]
    #         ],
    #         aff_threshold=0.03,

    #         local_xyz_scale=[0.0, 0.0, 0.0],
    #         use_ori_params=True,
    #         max_ac_calls=20,
    #         num_reach_steps=2,
    #         num_grasp_steps=3,
    #     ),
    #     push_config=dict(
    #         global_xyz_bounds=[
    #             [-0.30, -0.30, 0.80],
    #             [0.15, 0.30, 0.85]
    #         ],
    #         delta_xyz_scale=[0.25, 0.25, 0.05],

    #         max_ac_calls=20,
    #         use_ori_params=True,

    #         aff_threshold=[0.12, 0.12, 0.04],
    #     ),
    # )

    # env = suite.make(
    #     env_name="StackMAPLE",
    #     robots="Panda",
    #     env_configuration="default",
    #     controller_configs=load_controller_config(default_controller="OSC_POSITION"),
    #     gripper_types="default",
    #     initialization_noise="default",
    #     table_full_size=(0.8, 0.8, 0.05),
    #     table_friction=(1., 5e-3, 1e-4),
    #     table_offset=(0, 0, 0.8),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     reward_shaping=False,
    #     placement_initializer=None,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     render_collision_mesh=False,
    #     render_visual_mesh=True,
    #     render_gpu_device_id=-1,
    #     control_freq=10,
    #     horizon=1000,
    #     ignore_done=False,
    #     hard_reset=True,
    #     camera_names="agentview",
    #     camera_heights=256,
    #     camera_widths=256,
    #     camera_depths=False,
    #     skill_config=skill_config,
    # )
    # obs = env.reset()
    # env.render()
    # skill_action = np.array([0, 1, 0]) # reach
    # params = np.array([-0.1, 0.1, 0.83])
    # action = np.concatenate([skill_action, params])
    # obs, reward, done, info = env.step()
    

    main()

