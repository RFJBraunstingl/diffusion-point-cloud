import os
import argparse

import numpy

from utils.misc import *
from utils.dataset import *
from utils.data import *

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, default='./results/viz/npy/out.npy')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
pc_name = args.file.split("/")[-1].replace(".npy", "")
save_dir = os.path.join(args.save_dir, 'viz', pc_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logger = get_logger('test', save_dir)
logger.info(f"Loading pointclouds from '${args.file}'")
gen_pcs = numpy.load(args.file)
for i, pc in enumerate(gen_pcs):
    # pc: (2048, 3)
    output_path = save_dir + ("/%03d.txt" % (i))
    with open(output_path, "w") as outfile:
        for point in pc:
            x = point[0].item()
            y = point[1].item()
            z = point[2].item()
            outfile.write("%f %f %f\n" % (x, y, z))
