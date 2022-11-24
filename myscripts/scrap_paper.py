import h5py
import pdb
import random

import robosuite as suite
from robosuite import load_controller_config
import pandas as pd
import numpy as np

def do_nothing(hdf5_path):
    f = h5py.File(hdf5_path)
    pdb.set_trace()

def update_num_samples(hdf5_path):
    f = h5py.File(hdf5_path, "r+")
    total = 0
    for key in f["data"].keys():
        num_samples = f["data"][key]["actions"].shape[0]
        total += num_samples
        f["data"][key].attrs["num_samples"] = num_samples

    f["data"].attrs["total"] = total
    pdb.set_trace()
    f.close()

def copy_model_file(hdf5_path):
    f = h5py.File(hdf5_path)
    model_file = f["data/demo_1"].attrs["model_file"]
    for key in f["data"].keys():
        f["data"][key].attrs["model_file"] = model_file

env = suite.make(
    env_name="DrawerEnv",
    controller_configs=load_controller_config(default_controller="OSC_POSITION"),
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    use_camera_obs=False,
    control_freq=20,
    ignore_done=True,
)

pdb.set_trace()

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
