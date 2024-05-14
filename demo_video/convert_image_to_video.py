from glob import glob
import cv2
import os
import argparse
def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width
 
    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1
 
    return img_array, (_width, _height)
 
def images_to_video(images, video_file):
    # make all image size is same
    img_array = []
    for i in images:
        img = cv2.imread(i)
        if img is None:
            continue
        img_array.append(img)
    img_array, size = resize(img_array, 'largest')
    fps = 5
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# files = sorted(glob('/home/francis/Downloads/pole_frames/*.png'))
# images_to_video(files, 'video_pole.avi')

# files = sorted(glob('/home/francis/codes/vis/output/debug/230302/2366_wfLKz8hN-uw/*.jpg'))
# images_to_video(files, '/home/francis/codes/vis/output/debug/230301/2366_wfLKz8hN-uw.avi')
    
if __name__ == '__main__':
    # video_file = '/home/francis/codes/Mask2Former/input/skating/skating.mp4'
    # main(video_file)
    video_dir = 'debug'
    out_dir = 'debug_mp4'
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('--video_dir', type=str, help='video directory',default=video_dir)
    parser.add_argument('--out_dir', type=str, help='output directory',default=out_dir)
    args = parser.parse_args()
    video_dir = args.video_dir
    out_dir = args.out_dir
    file_list = os.listdir(video_dir)

    # debug/1.jpg
    # debug/0.jpg
    jpg_file_list = []
    for file in file_list:
        if file.endswith('.jpg'):
            jpg_file_list.append(file)
    jpg_file_list = sorted(jpg_file_list)
    files = [os.path.join(video_dir, file) for file in jpg_file_list]
    images_to_video(files, os.path.join(out_dir, video_dir + '.avi'))
    