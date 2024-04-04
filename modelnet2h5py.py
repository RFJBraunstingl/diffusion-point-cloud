import os.path
import numpy as np
import h5py

shapenet_path = './data/shapenet.hdf5'
modelnet_path = './data/modelnet40_normal_resampled'
modelnet_train_list_name = 'modelnet40_train.txt'
modelnet_test_list_name = 'modelnet40_test.txt'
category_prefix = 'modelnet40_'
output_file_path = './data/modelnet40.hdf5'
val_split_percentage = 25
dtype = 'f4'
num_points = 2048

# read shapenet for comparison
with h5py.File(shapenet_path, mode='r') as f:
    print("debug")
    # f['04460130'] -> group "/04460130" (3 members)
    # f['04460130']['train'] -> dataset "train": shape (104, 2048, 3), type f4

# read modelnet pointclouds
modelnet_train_list_path = os.path.join(modelnet_path, modelnet_train_list_name)
modelnet_test_list_path = os.path.join(modelnet_path, modelnet_test_list_name)
# read every line of train list file and strip the trailing newline
modelnet_train_list = [s[:-1] for s in open(modelnet_train_list_path, 'r')]
modelnet_test_list = [s[:-1] for s in open(modelnet_test_list_path, 'r')]
# every entry in modelnet_train_list is in this format: xbox_0103
# split at underline and filter duplicates to get category names
modelnet_categories = set([s[:s.rindex('_')] for s in modelnet_train_list])
print(f"{len(modelnet_categories)} categories found!")
print(modelnet_categories)


def convert_file_to_pc(category, file_name):
    path = os.path.join(modelnet_path, category, file_name)
    with open(path, 'r') as file:
        lines = [line for line in file]
        selected_lines = np.random.choice(lines, num_points)
        pc = [line.split(',')[0:3] for line in selected_lines]

    as_array = np.array(pc).astype(dtype)
    return as_array


with h5py.File(output_file_path, mode='w') as f:
    for c in modelnet_categories:
        train_files = [s + '.txt' for s in modelnet_train_list if s.startswith(c)]
        test_files = [s + '.txt' for s in modelnet_test_list if s.startswith(c)]
        print(f"train files: {train_files}")
        number_of_files_in_group = len(train_files)
        print(f"found {number_of_files_in_group} files for category {c}")
        number_of_val_files = number_of_files_in_group * val_split_percentage // 100
        print(f"using {number_of_val_files} files for validation")
        val_files = np.random.choice(train_files, number_of_val_files)
        print(f"val files: {val_files}")

        train_pcs = np.array([convert_file_to_pc(c, f) for f in train_files])  # i.e.: (104, 10000, 3)
        val_pcs = np.array([convert_file_to_pc(c, f) for f in val_files])
        test_pcs = np.array([convert_file_to_pc(c, f) for f in test_files])

        grp = f.create_group(category_prefix + c)
        grp.create_dataset('train', data=train_pcs)
        grp.create_dataset('val', data=val_pcs)
        grp.create_dataset('test', data=test_pcs)
