coco_path = 'datasets/coco/annotations/instances_val2017.json'
output_json = 'datasets/instances_val2017_sport_merge.json'
import json 
import os
import numpy as np
import cv2
import detectron2
from tqdm import tqdm
# load COCO annotations
with open(coco_path, 'r') as f:
    coco = json.load(f)
import torchshow
subject_txt = 'subject_set.txt'
CLASS2ID = { coco['categories'][i]['name']: coco['categories'][i]['id'] for i in range(len(coco['categories'])) }
ID2CLASS = { coco['categories'][i]['id']: coco['categories'][i]['name'] for i in range(len(coco['categories'])) }
# load subject set
with open(subject_txt, 'r') as f:
    subject_set = f.readlines()
    subject_set = [x.strip().split(':')[-1] for x in subject_set]
subject_set = set(subject_set)
# 去掉flower 和 person
subject_set.remove('flower')
subject_set.remove('person')
subject_set = { CLASS2ID[x] for x in subject_set }

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
accessory_list = ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'tennis racket', 'cell phone','frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard']
accessory_set = { CLASS2ID[x] for x in accessory_list }


# 遍历所有图片的标注
# 保存包含subject_set中的类别的图片，去除不包含subject_set中的类别的图片
# 接着保留下的图片中，如果有accessory_set中的类别，计算与所有person的重叠程度，如果重叠程度大于0.5，mask merge起来
# 处理后生成新的json文件
new_coco = {'images': [], 'annotations': [], 'categories': coco['categories']}
new_images = []
new_annotations = []
from pycocotools.coco import COCO
overlap_thres = 0.3
coco = COCO(coco_path)
for img_id in tqdm(coco.imgs.keys()):
    if img_id in [785]:
        print('debug')
    img_info = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    new_anns_sub = []
    new_anns_acc = []
    new_anns_person = []
    for ann in anns:
        if ann['category_id'] in subject_set:
            new_anns_sub.append(ann)
        if ann['category_id'] in accessory_set:
            new_anns_acc.append(ann)
        if ann['category_id'] == 1:
            new_anns_person.append(ann)
    print('human',len(new_anns_person), len(new_anns_acc), 'sub',len(new_anns_sub))
    if len(new_anns_person) and len( new_anns_acc):
        # 进行mask merge 操作
        from pycocotools import mask
        from pycocotools.mask import merge
        from pycocotools.mask import decode
        from pycocotools.mask import encode
        import numpy as np
        for person_ann in new_anns_person:
            if isinstance(person_ann['segmentation'], list):
                person_mask = mask.decode(mask.frPyObjects(person_ann['segmentation'], img_info['height'], img_info['width']))
            else:
                person_mask = mask.decode(mask.frPyObjects(person_ann['segmentation'], img_info['height'], img_info['width']))
            for acc_ann in new_anns_acc:
                if isinstance(acc_ann['segmentation'], list):
                    acc_mask = mask.decode(mask.frPyObjects(acc_ann['segmentation'], img_info['height'], img_info['width']))
                else:
                    acc_mask = mask.decode(mask.frPyObjects(acc_ann['segmentation'], img_info['height'], img_info['width']))
                person_box =  person_ann['bbox']
                acc_box = acc_ann['bbox']
                def compute_overlap(box1, box2):
                    x1, y1, w1, h1 = box1
                    x2, y2, w2, h2 = box2
                    x1_ = x1 + w1
                    y1_ = y1 + h1
                    x2_ = x2 + w2
                    y2_ = y2 + h2
                    x_overlap = max(0, min(x1_, x2_) - max(x1, x2))
                    y_overlap = max(0, min(y1_, y2_) - max(y1, y2))
                    overlap = x_overlap * y_overlap
                    area1 = w1 * h1
                    area2 = w2 * h2
                    return overlap / area2
                overlap = compute_overlap(person_box, acc_box)
                if overlap > overlap_thres:
                    # if len(person_mask.shape) == 3:
                    #     person_mask = person_mask[:, :, 0]
                    # if len(acc_mask.shape) == 3:
                    #     acc_mask = acc_mask[:, :, 0]
                    # merged_mask = np.logical_or(person_mask, acc_mask)
                    # person_mask = merged_mask
                    if len(person_mask.shape) == 2:
                        person_mask = np.expand_dims(person_mask, axis=-1)
                    if len(acc_mask.shape) == 2:
                        acc_mask = np.expand_dims(acc_mask, axis=-1)

                    merged_mask = np.concatenate((person_mask, acc_mask), axis=-1)
                    person_mask = merged_mask
            person_mask = mask.encode(np.array(person_mask, order='F'))
            if isinstance(person_mask, list):
                person_mask = mask.merge(person_mask)
            # acc_mask = mask.encode(np.array(acc_mask, order='F'))
            # acc_mask = mask.decode(mask.frPyObjects(acc_mask, img_info['height'], img_info['width']))
            #person_ann['segmentation'] = person_mask
            #person_ann['segmentation'] = [{'counts': m['counts'].decode('utf-8'), 'size': m['size']} for m in person_mask]
            if isinstance(person_mask, dict):
                person_ann['segmentation'] = {'counts': person_mask['counts'].decode('utf-8'), 'size': person_mask['size']}
            elif isinstance(person_mask, list):
                person_ann['segmentation'] = [{'counts': m['counts'].decode('utf-8'), 'size': m['size']} for m in person_mask]
            else:
                raise NotImplementedError
            person_ann['bbox'] = mask.toBbox(person_mask).tolist()
            #ori_person_mask = mask.decode(mask.frPyObjects(person_ann['segmentation'], img_info['height'], img_info['width']))
            try:
                person_ann['area'] = mask.area(person_mask).item()
            except: 
                person_ann['area'] = mask.area(person_mask).sum().item()
                print(img_info['file_name'])
            new_annotations.append(person_ann)
            # add other annotations
        for ann in new_anns_sub:
                new_annotations.append(ann)
    else:
        for ann in new_anns_sub:
            new_annotations.append(ann)
        for ann in new_anns_person:
            new_annotations.append(ann)

    if len(new_anns_person) or len(new_anns_sub):
        new_images.append(img_info)
        new_coco['annotations'] = new_annotations
        new_coco['images'] = new_images
        print(len(new_annotations), len(new_images))

ann_ids = [anns['id'] for anns in new_annotations]
print(len(ann_ids))
from collections import Counter
# Count the occurrences of each annotation id
counter = Counter(ann_ids)

duplicates = [(id,count) for id, count in counter.items() if count > 1]

with open(output_json, 'w') as f:
    json.dump(new_coco, f)




    

