# -*- coding: utf-8 -*-  
import json
from tqdm import *

def get_test_annotation():
    ## annotation data with coco format for evaluation
    annotation_file = '../data/TextCaps_0.1_val.json'
    out_gt_file = '../data/TC_val_coco.json'

    annotations = json.load(open(annotation_file, 'r'))['data']

    annotation = []
    image = []
    image_filter = []
    i = 0
    for ann in tqdm(annotations):
        gt_dict = {}
        caption = ann['caption_str']
        image_id = ann['image_name']

        if image_id not in image_filter:
            image_filter.append(image_id)
            im_dict = {}
            im_dict['id'] = image_id
            image.append(im_dict)

        # if all(ord(c) < 128 for c in caption):
        gt_dict['image_id'] = image_id
        gt_dict['id'] = i
        # print story.decode('utf8')
        gt_dict['caption'] = caption
        annotation.append(gt_dict)
        i += 1

    groundtruth = {}
    groundtruth['annotations'] = annotation
    groundtruth['images'] = image
    groundtruth['info'] = {'ex1':1}
    groundtruth['licenses'] = [1,2,3]
    groundtruth['type'] = 'captions'
    
    with open(out_gt_file, 'w') as out_gt_fid:
        json.dump(groundtruth, out_gt_fid)
    return groundtruth

get_test_annotation()