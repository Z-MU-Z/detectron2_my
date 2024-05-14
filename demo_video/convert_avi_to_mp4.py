import cv2
import os
import subprocess
import argparse
def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))



if __name__ == '__main__':

    source_dir = 'debug_avi' # avi文件的路径
    dest_dir = source_dir + '_mp4' # mp4文件的路径
    parser = argparse.ArgumentParser(description='Convert avi to mp4')
    parser.add_argument('--source_dir', type=str, help='source directory',default=source_dir)
    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = source_dir + '_mp4'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for video in os.listdir(source_dir):
        if video.endswith('.avi'):
            avi_file_path = os.path.join(source_dir, video)
            output_name = os.path.join(dest_dir, os.path.splitext(video)[0]) # 无后缀文件名
            convert_avi_to_mp4(avi_file_path, output_name)
