coco_path = 'datasets/coco/annotations/instances_train2017.json'
output_json = 'datasets/instances_train2017_p3m_10k_pro.json'
image_path = 'datasets/P3M-10k/train'
rgb_path = image_path  + '/blurred_image'
label_path = image_path + '/mask'
import json 
import os
import numpy as np
import cv2
from tqdm import tqdm
# load COCO annotations
with open(coco_path, 'r') as f:
    coco = json.load(f)
# matting are all person
new_coco = {'images': [], 'annotations': [], 'categories': coco['categories']}# original coco['annotations'] , coco['images']
new_images = []
new_annotations = []
# img list
img_list = os.listdir(rgb_path) 
MAX_IMAGE_ID = max([x['id'] for x in coco['images']]) + 1
MAX_ANNOTATION_ID = max([x['id'] for x in coco['annotations']]) + 1
img_id = MAX_IMAGE_ID
ann_id = MAX_ANNOTATION_ID
import pycocotools.mask as mask_util
for img in tqdm(img_list):
    img_id += 1
    ann_id += 1
    img_name = img.split('.')[0]
    img_path = os.path.join(rgb_path, img)
    mask_path = os.path.join(label_path, img_name + '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    new_images.append({'file_name': img, 'height': h, 'width': w, 'id': img_id})
    # convert to binary mask
    original_mask = mask.copy()
    mask = original_mask > 128
    #腐蚀膨胀
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(np.uint8)
    # import torchshow
    # torchshow.save(mask,str(img_id)+'128.png')
    # mask = original_mask > 0
    # mask = mask.astype(np.uint8)
    # torchshow.save(mask,str(img_id)+'0.png')
    mask = mask_util.encode(np.array(mask, order='F'))
    if isinstance(mask, list):
        mask = mask_util.merge(mask)
    if isinstance(mask, dict):
        mask = {'counts': mask['counts'].decode('utf-8'), 'size': mask['size']}
    new_annotations.append({'image_id': img_id, 'category_id': 1, 'segmentation': mask, 'id': ann_id, 'bbox': mask_util.toBbox(mask).tolist(), 'area': mask_util.area(mask).item()})

new_coco['images'] = new_images
new_coco['annotations'] = new_annotations

with open(output_json, 'w') as f:
    json.dump(new_coco, f)


    
   