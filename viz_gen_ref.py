import os
import argparse

from utils.misc import *
from utils.dataset import *
from utils.data import *

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

save_dir = os.path.join(args.save_dir, 'viz', 'ref')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logger = get_logger('test', save_dir)
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=args.normalize,
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

ref_pcs = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)
# ref_pcs: (607, 2048, 3)
logger.info('Saving point clouds...')
for i, pc in enumerate(ref_pcs):
    # pc: (2048, 3)
    output_path = save_dir + ("/ref_%03d.txt" % (i))
    with open(output_path, "w") as outfile:
        for point in pc:
            x = point[0].item()
            y = point[1].item()
            z = point[2].item()
            outfile.write("%f %f %f\n" % (x, y, z))
