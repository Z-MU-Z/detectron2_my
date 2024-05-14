# 定义你的输入和输出目录
INPUT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/save/output"
AVI_OUTPUT_DIR=$OUTPUT_DIR+"avi"

# 定义你的模型权重文件路径
MODEL_WEIGHTS="/path/to/your/checkpoint"

# 遍历输入目录下的所有mp4文件
for VIDEO in $INPUT_DIR/*.mp4
do
  # 使用basename命令获取不带路径的文件名
  BASENAME=$(basename "$VIDEO")

  # 使用python命令处理每一个视频文件
  python demo/demo.py --config-file configs/ytvis_2019/CTVIS_R50.yaml --video-input "$VIDEO" --output "$OUTPUT_DIR/$BASENAME" --save-frames --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"
  python demo/convert_images_to_video.py --video_dir "$OUTPUT_DIR/$BASENAME" --out_dir "$AVI_OUTPUT_DIR" 

done


python demo/convert_avi_to_mp4.py --avi_dir "$AVI_OUTPUT_DIR"