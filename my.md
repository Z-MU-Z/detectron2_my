git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

conda install nvidia/label/cuda-11.7.0::cuda-toolkit
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./



python create_instance_by_panseg.py --mask-dir  /home/zmz/code2/coconut_cvpr2024/coconut_dataset/coconut_s \
--image-dir /home/zmz/code2/coconut_cvpr2024/datasets/coco/train2017 \
--panseg-info /home/zmz/code2/coconut_cvpr2024/coconut_dataset/annotations/annotations/coconut_s.json \
--output coco_instances_train2017_coconut_s.json


if self.binary_thres > 0:
            keep = scores_per_image > self.binary_thres
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]


#!/bin/bash

# 定义你的输入和输出目录
INPUT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/save/output"

# 定义你的模型权重文件路径
MODEL_WEIGHTS="/path/to/your/checkpoint"

# 遍历输入目录下的所有mp4文件
for VIDEO in $INPUT_DIR/*.mp4
do
  # 使用basename命令获取不带路径的文件名
  BASENAME=$(basename "$VIDEO")

  # 使用python命令处理每一个视频文件
  python demo/demo.py --config-file configs/ytvis_2019/CTVIS_R50.yaml --video-input "$VIDEO" --output "$OUTPUT_DIR/$BASENAME" --save-frames --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"
done
