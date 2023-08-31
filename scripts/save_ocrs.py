"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
from collections import defaultdict
from clear_caption import clear_ocr, clear_cap
from tqdm import *

def build_vocab(imgs, params):

  # clear ocr tokens and captions
  img_ids = []
  all_imgs = []
  for img in tqdm(imgs):
    if img['image_id'] not in img_ids:
      img_dict = {}
      img_ids.append(img['image_id'])
      dup_ocr_tokens = np.load(str(params['input_ocr_root'])+'/'+img['image_id']+'_info.npy',\
                         allow_pickle=True)[()]['ocr_tokens']
      dup_ocr_tokens = [token.lower() for token in dup_ocr_tokens]

      # remove duplicated ocrs
      ocr_tokens = []
      ocr_ids = []
      for iii, token in enumerate(dup_ocr_tokens):
        token = clear_ocr(token)
        if len(ocr_tokens) < params['max_ocr_len'] and token not in ocr_tokens:
          ocr_tokens.append(token.lower())
          ocr_ids.append(iii)
      img_dict['ocr_ids'] = ocr_ids
      img_dict['new_ocr_tokens'] = ocr_tokens
      # print('\n')
      # print(dup_ocr_tokens)
      # print(ocr_tokens)
      img_dict['image_id'] = img['image_id']
      all_imgs.append(img_dict)

  return all_imgs

def main(params):

  val = json.load(open(params['input_json_PR']+'val.json', 'r'))
  imgs = val['data']

  seed(123) # make reproducible
  
  all_imgs = build_vocab(imgs, params)
  
  json.dump(all_imgs, open('ocr_tokens.json', 'w'))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json_PR', default='TextCaps_0.1_', required=True, help='prefix of input json file to process into hdf5')
  parser.add_argument('--output_json', default='TC.json', help='output json file')
  parser.add_argument('--output_h5', default='TC', help='output h5 file')
  parser.add_argument('--output_vocab', default='TC_vocab', help='output vocab file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--input_ocr_root', default='m4c_textvqa_ocr_en_frcn_features', required=True, help='')

  # options
  parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=10, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--max_ocr_len', default=30, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
