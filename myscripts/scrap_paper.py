import h5py
import pdb
import random

import robosuite

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

# def concat_hdf5_files(hdf5_path1, hdf5_path2):
#     f1 = h5py.File(hdf5_path1, "r+")
#     f2 = h5py.File(hdf5_path2, "r+")
#     # pdb.set_trace()
#     n = len(f1["data"].keys()) + 1
#     n_added = 0
#     for key in f2["data"].keys():
#         f1["data"].create_group(f"demo_{n}")
#         f2.copy(f2["data"][key], f1[f"data/demo_{n}"])
#         f1[f"data/demo_{n}"].attrs["model_file"] = f2["data"][key].attrs["model_file"]
#         f1[f"data/demo_{n}"].attrs["num_samples"] = f2["data"][key].attrs["num_samples"]
#         n_added += f2["data"][key].attrs["num_samples"]
#         n += 1
#     f1["data"].attrs["total"] = f1["data"].attrs["total"] + n_added

#     pdb.set_trace()
#     f1.close()
#     f2.close()


path1 = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/reaching_100.hdf5"
path2 = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/reaching_50_2.hdf5"

# do_nothing(path1)
update_num_samples(path1)

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
