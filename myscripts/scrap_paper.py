from __future__ import print_function

import os
import sys
import pdb

sys.path.insert(1, '/home/robot/perls2')

import numpy as np
import yaml

sys.path.insert(1, '/home/ayanoh/stable-baselines3/')

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import robosuite as suite
from robosuite.wrappers.gym_wrapper_custom import GymWrapper
from robosuite import load_controller_config

def main():
    with open("sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    
    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    env = suite.make(
        env_name="Reaching2D",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        use_skills=True,
        normalized_params=True,
        random_init=False,
        target_position=(0.1, 0.3),
        reward_scale=10.0,
    )
    # pdb.set_trace()
    env = GymWrapper(env, keys=["eef_xyz", "eef_yaw"])

    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[64, 64],
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    # while os.path.exists(config_data['data_save_path']):
    #     config_data['data_save_path'] = "/".join(config_data['data_save_path'].split("/")[:-1]) + '/run_' + str(int(random.random() * 1000000000))

    # try pre-generating file and read 
    # print(f"---------{config_data['data_save_path']}----------")
    # os.makedirs(config_data['data_save_path'], exist_ok=True)
    # f = open(f"{config_data['data_save_path']}/data_file.txt", "a")
    # f.close()

    model = SAC(
        config_data["policy_name"],
        env,
        verbose=config_data["verbose"],
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        # save_every=config_data["save_every_steps"],
        learning_rate=config_data["learning_rate"],
        buffer_size=config_data["buffer_size"],
        learning_starts=config_data["learning_starts"],
        batch_size=config_data["batch_size"],
        tau=config_data["tau"],
        gamma=config_data["gamma"],
        train_freq=config_data["train_freq"],
        gradient_steps=config_data["gradient_steps"],
        seed=config_data["seed"],
        # experiment_save_dir=config_data["data_save_path"],
        # render=False,
    )

    print(f"Model Policy = " + str(model.policy))

    if not config_data["load_model"]:
        model.learn(
            config_data["steps"],
        )
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=20, render=False
        )
        print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    else:
        del model
        model_num = config_data["load_model"]
        model = SAC.load(f"models/SAC_{model_num}.pt", env=env)
        print("Loaded pretrained model")


if __name__ == "__main__":
    # msg = "Overwrite config params"
    # parser = argparse.ArgumentParser(description = msg)
    # parser.add_argument("--seed", type=int, default=None)

    # args = parser.parse_args()
    # main(args)
    main()



# import h5py
# import pdb
# import random

# import robosuite as suite
# from robosuite import load_controller_config
# import pandas as pd
# import numpy as np

# def func(angles):
#     """
#     normalize angle in rad to range [-pi, 2pi]
#     """
#     pi2 = 2 * np.pi
#     result = np.fmod( np.fmod(angles, pi2) + pi2, pi2)
#     print(result)
#     if result > np.pi:
#         print("[1]")
#         result = result - pi2
#         print("result", result)
#     if result < -np.pi:
#         print("[2]")
#         print("result", result)
#         result = result + pi2
#     return result

# def do_nothing(hdf5_path):
#     f = h5py.File(hdf5_path)
#     pdb.set_trace()

# def update_num_samples(hdf5_path):
#     f = h5py.File(hdf5_path, "r+")
#     total = 0
#     for key in f["data"].keys():
#         num_samples = f["data"][key]["actions"].shape[0]
#         total += num_samples
#         f["data"][key].attrs["num_samples"] = num_samples

#     f["data"].attrs["total"] = total
#     pdb.set_trace()
#     f.close()

# def copy_model_file(hdf5_path):
#     f = h5py.File(hdf5_path)
#     model_file = f["data/demo_1"].attrs["model_file"]
#     for key in f["data"].keys():
#         f["data"][key].attrs["model_file"] = model_file

# env = suite.make(
#     env_name="Drop2",
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
# pick_block_transition(env, obs, {"goal_pos":(0.16,0.16,0.988)})


# path1 = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/reaching_100.hdf5"
# path2 = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/reaching_50_2.hdf5"

# do_nothing(path1)
# update_num_samples(path1)

# csv_path = "/home/ayanoh/robosuite/myscripts/video0.csv"
# pdb.set_trace()



# if "mask" in f.keys():
#     del f["mask"]

# demo_names = []
# for i in range(1, 51):
#     demo_names.append(f"demo_{i}")


# pdb.set_trace()
# random.shuffle(demo_names)
# n_valid = max(len(demo_names)//10, 1)
# trainset = demo_names[n_valid:]
# validset = demo_names[:n_valid]
# pdb.set_trace()

# grp_mask = f.create_group("mask")
# grp_mask.create_dataset(name="train", dtype="S8", data=trainset)
# grp_mask.create_dataset(name="valid", dtype="S8", data=validset)
# pdb.set_trace()
