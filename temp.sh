#!/usr/bin/env bash

rsync -avzu --progress /home/wzeng/mydata/H36Mnew/c2f_vol/rcnn wzeng:/home/wzeng/mydata/H36Mnew/c2f_vol/
rsync -avzu --progress /home/wzeng/mydata/mpii/rcnn wzeng:/home/wzeng/mydata/mpii/
rsync -avzu --progress /home/wzeng/mydata/coco/annotations wzeng:/home/wzeng/mydata/coco/
rsync -avzu --progress /home/wzeng/mydata/posetrack2018/rcnn wzeng:/home/wzeng/mydata/posetrack2018/
rsync -avzu --progress /home/wzeng/mydata/mpi_inf_3dhp_new/rcnn wzeng:/home/wzeng/mydata/mpi_inf_3dhp_new/

rsync -avzu --progress /home/wzeng/mycodes/Deformable-DETR/data wzeng:/home/wzeng/mycodes/transformer/Deformable-DETR/



conda create -n deformable_detr python=3.7 pip
source activate deformable_detr
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt

cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py

cd neural_renderer/
python3 setup.py install
cd ../sdf
python3 setup.py install

pip install smplx
pip install scikit-image
pip install opencv-python


ln -s  /home/wzeng/mydata/mupots-3d data/mupots-3d