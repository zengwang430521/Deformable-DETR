import pickle
from os.path import join, exists
import os
import shutil
from tqdm import tqdm





panoptic_root = '/home/wzeng/mydata/panoptic/'
annot_files = [panoptic_root + 'processed/annotations/160422_ultimatum1.pkl',
               panoptic_root + 'processed/annotations/160422_haggling1.pkl',
               panoptic_root + 'processed/annotations/160906_pizza1.pkl',
               panoptic_root + 'processed/annotations/160422_mafia2.pkl']

out_dir = join(panoptic_root, 'filtered')
if not exists(out_dir):
    os.makedirs(out_dir)

for ann_file in annot_files:
    with open(ann_file, 'rb') as f:
        raw_infos = pickle.load(f)
    for img_info in tqdm(raw_infos):
        imgname = img_info['filename']
        ori_path = join(panoptic_root, imgname)
        tar_path = join(out_dir, imgname)
        tar_dir = os.path.dirname(tar_path)
        if not exists(tar_dir):
            os.makedirs(tar_dir)
        shutil.copy(ori_path, tar_path)
print('finish')