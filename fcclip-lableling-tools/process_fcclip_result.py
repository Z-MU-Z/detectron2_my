# convert FCCLIP prediction result to COCO format.
coco_path = 'datasets/coco/annotations/instances_val2017.json'
fcclip_result_path = '/home/zmz/code/grounding-sam/FCCLIP-2030/output_new_417'
import json
import numpy as np
# load COCO annotations
with open(coco_path, 'r') as f:
    coco = json.load(f)
import os
import torchshow
new_coco = {'images': [], 'annotations': [], 'categories': coco['categories']}
CLASS2ID = { coco['categories'][i]['name']: coco['categories'][i]['id'] for i in range(len(coco['categories'])) }
ID2CLASS = { coco['categories'][i]['id']: coco['categories'][i]['name'] for i in range(len(coco['categories'])) }
new_images = []
new_annotations = []
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
# FCCLIP-2030/output_new_417/human_clear_20220617_1.json
# FCCLIP-2030/output_new_417/human_clear_20220617_2.json
# FCCLIP-2030/output_new_417/human_clear_20220617_3.json
# add new categories
class_txt = '/home/zmz/code/grounding-sam/FCCLIP-2030/stuff_classes.txt'
with open(class_txt, 'r') as f:
    stuff_classes = f.readlines()
    stuff_classes = [x.strip() for x in stuff_classes]
    stuff_classes = {i: stuff_classes[i] for i in range(len(stuff_classes))}
MAX_ID = max([x['id'] for x in new_coco['categories']]) + 1
categories_set = set([x['name'] for x in new_coco['categories']])
for i in range(len(stuff_classes)):
    if stuff_classes[i] not in categories_set:
        new_coco['categories'].append({'supercategory': 'addtional', 'id': MAX_ID, 'name': stuff_classes[i]})
        MAX_ID += 1


NEW_CLASS2ID = { new_coco['categories'][i]['name']: new_coco['categories'][i]['id'] for i in range(len(new_coco['categories'])) }
NEW_ID2CLASS = { new_coco['categories'][i]['id']: new_coco['categories'][i]['name'] for i in range(len(new_coco['categories'])) }

MAX_IMAGE_ID = max([x['id'] for x in coco['images']]) + 1
MAX_ANNOTATION_ID = max([x['id'] for x in coco['annotations']]) + 1
for json_path in os.listdir(fcclip_result_path):
    if not json_path.endswith('.json'):
        continue
    with open(os.path.join(fcclip_result_path, json_path), 'r') as f:
        fcclip_result = json.load(f) # [{'category_name': 'person', 'mask': {...}}, {'category_name': 'shoe', 'mask': {...}}] 
        # 'mask': {'counts': 'g^_6=ac0=E4K2O1O0000...1N3M3Mhhb5', 'size': [640, 640]}
    image_id = MAX_IMAGE_ID
    MAX_IMAGE_ID += 1
    file_name = json_path.replace('.json', '.jpg')
    height, width = fcclip_result[0]['mask']['size']
    new_images.append({'file_name': file_name, 'height': height, 'width': width, 'id': image_id})
    for i in range(len(fcclip_result)):
        category_name = fcclip_result[i]['category_name']
        if category_name not in NEW_CLASS2ID:
            continue
        category_id = NEW_CLASS2ID[category_name]
        mask = fcclip_result[i]['mask']
        box = maskUtils.toBbox(mask).tolist()
        new_annotations.append({'image_id': image_id, 'category_id': category_id, 'segmentation': mask, 'id': MAX_ANNOTATION_ID, 'bbox': box, 'area': maskUtils.area(mask).item()})
        MAX_ANNOTATION_ID += 1

new_coco['images'] = new_images
new_coco['annotations'] = new_annotations
output_json = 'datasets/output_new_417.json'
with open(output_json, 'w') as f:
    json.dump(new_coco, f)



    


