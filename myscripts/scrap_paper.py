import h5py
import pdb

path = "/home/ayanoh/robosuite/robosuite/models/assets/demonstrations/1666082464_8826947/demo.hdf5"
f = h5py.File(path, "r")
f2 = h5py.File("/home/ayanoh/robomimic/datasets/lift/ph/low_dim.hdf5")
pdb.set_trace()
