"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.

!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!
Fix issue with num_samples and total attributes before collecting next dataset!
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob
import copy

import h5py
import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import CustomDataCollectionWrapper, VisualizationWrapper
import pdb

def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        observations, reward, done, info = env.step(action)
        print("delta ", observations["robot0_delta_to_target"])
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 25  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")
    grp_mask = f.create_group("mask")

    num_eps = 0
    env_name = None  # will get populated at some point
    total_samples = 0
    demo_names = []

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        observations = []
        rewards = []
        dones = []
        num_samples = 0

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            
            observations.extend(dic["obs_infos"])
            rewards.extend(dic["rewards"])
            dones.extend(dic["dones"])

            num_samples += dic["num_samples"]
            total_samples += dic["num_samples"]

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1

        ep_data_grp = grp.create_group("demo_{}".format(num_eps))
        demo_names.append(f"demo_{num_eps}")

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store number of samples in this episode
        ep_data_grp.attrs["num_samples"] = num_samples

        # write datasets for states, actions, rewards, dones, obs, next_obs
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        ep_data_grp.create_dataset("dones", data=np.array(dones))

        obs = {key : observations[0][key] for key in observations[0].keys()}
        # pdb.set_trace()
        
        for key in obs:
            for observation in observations:
                obs[key] = np.vstack((obs[key], observation[key]))
            ep_data_grp[f"obs/{key}"] = obs[key][:-1]
            ep_data_grp[f"next_obs/{key}"] = obs[key][1:]
        # pdb.set_trace()

    # write dataset attributes (metadata)
    # now = datetime.datetime.now()
    # grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    # grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    # grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env_name"] = env_name
    grp.attrs["type"] = 1
    grp.attrs["env_args"] = env_info
    grp.attrs["total"] = total_samples
    # pdb.set_trace()
    
    # write masks
    np.random.shuffle(demo_names)
    n_valid = max(num_eps // 10, 1)
    grp_mask.create_dataset("train", data=demo_names[n_valid:], dtype="S8")
    grp_mask.create_dataset("valid", data=demo_names[:n_valid], dtype="S8")
    print("===============Saved hdf5=============")
    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="ReachingFrankaBC")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="frontview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    # env_info = json.dumps(config)

    # rearrange dictionary to match robomimic requirements
    env_args = copy.deepcopy(config)
    env_args["type"] = 1 # add type (1 = robosuite)
    env_kwargs = {}
    for key in config:
        if key == "type" or key == "env_name":
            continue
        env_kwargs[key] = config[key]
        del env_args[key]
    env_args["env_kwargs"] = env_kwargs

    env_info = json.dumps(env_args)
    # pdb.set_trace()

    # file = h5py.File("/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/franka_reaching_robosuite_50.hdf5", "r+")
    # file["data"].attrs["env_args"] = json.dumps(env_args)
    # file.close()

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = CustomDataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            vendor_id=9583,
            product_id=50734,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))

    os.makedirs(new_dir)

    # collect demonstrations
    n_eps_to_collect = 120
    for i in range(n_eps_to_collect):
        print(f"----------{i}------------")
        collect_human_trajectory(env, device, args.arm, args.config)    
    gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
