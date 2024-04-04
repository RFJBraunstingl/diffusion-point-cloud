import h5py

output_file_path = './data/modelnet40.hdf5'

# read shapenet for comparison
with h5py.File(output_file_path, mode='r') as f:
    print("debug")
    f['']
