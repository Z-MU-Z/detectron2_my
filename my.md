git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

conda install nvidia/label/cuda-11.7.0::cuda-toolkit
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./



python create_instance_by_panseg.py --mask-dir  /home/zmz/code2/coconut_cvpr2024/coconut_dataset/coconut_s \
--image-dir /home/zmz/code2/coconut_cvpr2024/datasets/coco/train2017 \
--panseg-info /home/zmz/code2/coconut_cvpr2024/coconut_dataset/annotations/annotations/coconut_s.json \
--output coco_instances_train2017_coconut_s.json
