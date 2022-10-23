import h5py
import pdb
import random

path = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/franka_reaching_robosuite/franka_reaching_robosuite_50.hdf5"
f = h5py.File(path, "r+")
f2 = h5py.File("/home/ayanoh/robomimic/datasets/lift/ph/low_dim.hdf5", "r")

pdb.set_trace()
# print("myfile")
# for i in range(1, 51):
#     print(f"demo_{i}")
#     print(f[f"data/demo_{i}/actions"][0])
#     print(f[f"data/demo_{i}/obs/robot0_eef_pos"][0])
#     print(f[f"data/demo_{i}/obs/robot0_delta_to_target"][0])
#     print(f[f"data/demo_{i}/next_obs/robot0_eef_pos"][0])
#     print(f[f"data/demo_{i}/next_obs/robot0_delta_to_target"][0])
#     print(f[f"data/demo_{i}/rewards"][0])
#     print(f[f"data/demo_{i}/dones"][0])
#     print(f[f"data/demo_{i}/states"][0])

# pdb.set_trace()
# print("example file")
# for i in range(1, 51):
#     print(f"demo_{i}")
#     print(f2[f"data/demo_{i}/actions"][0])
#     print(f2[f"data/demo_{i}/obs/robot0_eef_pos"][0])
#     print(f2[f"data/demo_{i}/next_obs/robot0_eef_pos"][0])
#     print(f2[f"data/demo_{i}/rewards"][0])
#     print(f2[f"data/demo_{i}/dones"][0])
#     print(f2[f"data/demo_{i}/states"][0])
# for i in range(1, 51):
#     print(f"demo_{i}")
#     print(f[f"data/demo_{i}/actions"].shape)
#     print(f[f"data/demo_{i}/obs/robot0_eef_pos"].shape)
#     print(f[f"data/demo_{i}/obs/robot0_delta_to_target"].shape)
#     print(f[f"data/demo_{i}/next_obs/robot0_eef_pos"].shape)
#     print(f[f"data/demo_{i}/next_obs/robot0_delta_to_target"].shape)
#     print(f[f"data/demo_{i}/rewards"].shape)
#     print(f[f"data/demo_{i}/dones"].shape)
#     print(f[f"data/demo_{i}/states"].shape)

# pdb.set_trace()

# print("example file")

# for i in range(1, 51):
#     print(f"demo_{i}")
#     print(f2[f"data/demo_{i}/actions"].shape)
#     print(f2[f"data/demo_{i}/obs/robot0_eef_pos"].shape)
#     print(f2[f"data/demo_{i}/next_obs/robot0_eef_pos"].shape)
#     print(f2[f"data/demo_{i}/rewards"].shape)
#     print(f2[f"data/demo_{i}/dones"].shape)
#     print(f2[f"data/demo_{i}/states"].shape)


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
