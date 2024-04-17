coco_path = 'datasets/output_new_417.json'
standard_coco_path = 'datasets/coco/annotations/instances_val2017.json'
output_json = 'datasets/output_new_417_coco.json'
import json 
import os
import numpy as np
import cv2
import detectron2
from tqdm import tqdm
# load COCO annotations
with open(coco_path, 'r') as f:
    coco = json.load(f)
with open(standard_coco_path, 'r') as f:
    standard_coco = json.load(f)
import torchshow
subject_txt = 'subject_set_new.txt'
CLASS2ID = { coco['categories'][i]['name']: coco['categories'][i]['id'] for i in range(len(coco['categories'])) }
ID2CLASS = { coco['categories'][i]['id']: coco['categories'][i]['name'] for i in range(len(coco['categories'])) }
# load subject set
with open(subject_txt, 'r') as f:
    subject_set = f.readlines()
    subject_set = [x.strip().split(':')[-1] for x in subject_set]
#subject_set = set(subject_set)
# load super class set
super_class_path = 'FCCLIP-2030/demo/result.txt'
# format as follows:
'''
- People (including parts):
    - person

- Cars:
    - ambulance
    - bus (vehicle)
    - car (automobile)
    - cab (taxi)
    - cabin car
    - camper (vehicle)
    - convertible (automobile)
    - garbage truck
    - golfcart
    - jeep
    - minivan
    - motor scooter
    - motor vehicle
    - motorcycle
    - race car
    - school bus
    - snowmobile
    - dirt bike
'''
class2super = {}
super_class_list = []
with open(super_class_path, 'r') as f:
    super_class_set = f.readlines()
    for line in super_class_set:
        if ":" in line:
            super_class = line.strip().replace(':', '')
            super_class_list.append(super_class)
        else:
            class2super[line.strip().replace('- ','')] = super_class

# build a mapping from super class to coco class, according to the subject set
super_class2coco = {}
for super_class in super_class_list:
    super_class2coco[super_class] = subject_set[super_class_list.index(super_class)]

class2coco = {}
for class_name in class2super.keys():
    super_class = class2super[class_name]
    coco_class = super_class2coco[super_class]
    class2coco[class_name] = coco_class
candidate_class = list(class2coco.keys())
candidate_id_set = { CLASS2ID[x] for x in candidate_class }
# '- People (including parts)':
# 'person'
# '- Cars':
# 'car'
# '- Dogs':
# 'dog'
# '- Cats':
# 'cat'
# '- Birds':
# 'bird'
# '- Flowers':
# 'vase'
# '- Potted Plants':
# 'potted plant'
# '- Chairs':
# 'chair'

# 去掉flower 和 person
# subject_set.remove('flower')
# subject_set.remove('person')
# subject_set = { CLASS2ID[x] for x in subject_set }

# 29:
# {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}
# 30:
# {'supercategory': 'sports', 'id': 35, 'name': 'skis'}
# 31:
# {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}
# 32:
# {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}
# 33:
# {'supercategory': 'sports', 'id': 38, 'name': 'kite'}
# 34:
# {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}
# 35:
# {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}
# 36:
# {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}
# 37:
# {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}
# 38:
# {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}
# accessory_list = ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'tennis racket', 'cell phone','frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard']
# accessory_set = { CLASS2ID[x] for x in accessory_list }
# read accessory set from file
accessory_txt = 'app_set.txt'
with open(accessory_txt, 'r') as f:
    accessory_set = f.readlines()
    accessory_set = [x.strip().split(':')[-1] for x in accessory_set]
accessory_set = { CLASS2ID[x]  for x in accessory_set }
backpack_id = CLASS2ID['backpack'] # 便于后续处理需要，所有accessory都当做backpack处理
# 遍历所有图片的标注
# 保存包含subject_set中的类别的图片，去除不包含subject_set中的类别的图片
# 接着保留下的图片中，如果有accessory_set中的类别，计算与所有person的重叠程度，如果重叠程度大于0.5，mask merge起来
# 处理后生成新的json文件
new_coco = standard_coco
new_images = []
new_annotations = []
new_coco['annotations'] = new_annotations
new_coco['images'] = new_images
from pycocotools.coco import COCO
coco = COCO(coco_path)
for img_id in tqdm(coco.imgs.keys()):
    if img_id in [785]:
        print('debug')
    img_info = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    new_anns_sub = []
    new_anns_acc = []
    for ann in anns:
        if ann['category_id'] in candidate_id_set:
            print(class2coco[ID2CLASS[ann['category_id']]])
            ann['category_id'] = CLASS2ID[class2coco[ID2CLASS[ann['category_id']]]]
            new_anns_sub.append(ann)
        if ann['category_id'] in accessory_set:
            ann['category_id'] = backpack_id
            new_anns_acc.append(ann)
    if len(new_anns_sub) == 0:
        continue
    else:
        new_annotations.extend(new_anns_sub)
        new_annotations.extend(new_anns_acc)
        new_images.append(img_info)
        

ann_ids = [anns['id'] for anns in new_annotations]
print(len(ann_ids))
from collections import Counter
# Count the occurrences of each annotation id
counter = Counter(ann_ids)

duplicates = [(id,count) for id, count in counter.items() if count > 1]

with open(output_json, 'w') as f:
    json.dump(new_coco, f)




    

