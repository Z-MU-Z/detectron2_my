import os

source_dir = '/data1/models/ZJU_segment/Segment/ann'
 
target_ip = ['135','232']

target_prefix = '/data1/gpu/data1_{}/'

for ip in target_ip:
    target_dir_ = target_prefix.format(ip)
    target_dir = ''.join([target_dir_, '/'.join(source_dir.replace('/data1','').split('/')[:-1])])
    os.system('mkdir -p {}'.format(target_dir))
    os.system('rsync -avzP {} {}'.format(source_dir, target_dir))
    print('rsync -avzP {} {}'.format(source_dir, target_dir))
