import os
import sys
DATA_BASE_PATH = os.path.abspath('./Stanford3dDataset_v1.2_Aligned_Version')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
# anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]
anno_paths = [os.path.join(DATA_BASE_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    elements = anno_path.split('/')
    out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
    out_filepath = os.path.join(output_folder, out_filename)
    if os.path.exists(out_filepath):
        print('Exist {}'.format(out_filename))
        continue

    indoor3d_util.collect_point_label(anno_path, out_filepath, 'numpy')

    try:

        indoor3d_util.collect_point_label(anno_path, out_filepath, 'numpy')
    except:
        print(anno_path, 'ERROR!!')