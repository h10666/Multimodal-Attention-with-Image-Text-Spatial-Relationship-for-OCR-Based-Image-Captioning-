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
import pickle

def norm(box, w, h):
  box[0] = box[0]/w
  box[1] = box[1]/h
  box[2] = box[2]/w
  box[3] = box[3]/h
  return box, box[2]-box[0], box[3]-box[1]

def get_io(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

  iou = interArea / float(boxAArea + boxBArea - interArea)

  ioA = interArea / float(boxAArea)
  ioB = interArea / float(boxBArea)
  
  score = ioA < 0.5 and ioB < 0.5
  _dist, dist = min_dist_sqrt(boxA, boxB)
  return iou > 0 and score, iou == 0 and _dist, ioA+ioB, dist

def min_dist_sqrt(boxA, boxB):
  wA = abs(boxA[0]-boxA[2])
  hA = abs(boxA[1]-boxA[3])
  wB = abs(boxB[0]-boxB[2])
  hB = abs(boxB[1]-boxB[3])

  # center coord
  xA = (boxA[0]+boxA[2])/2 
  yA = (boxA[1]+boxA[3])/2
  xB = (boxB[0]+boxB[2])/2
  yB = (boxB[1]+boxB[3])/2

  dx = abs(xA - xB)
  dy = abs(yA - yB)

  if dx < (wA+wB)/2 and dy >= (hA+hB)/2:
      dist = (dy - (hA+hB)/2) ** 2
  elif dx >= (wA+wB)/2 and dy < (hA+hB)/2:
      dist = (dx - (wA+wB)/2) ** 2
  elif dx >= (wA+wB)/2 and dy >= (hA+hB)/2:
      delta_x = dx - (wA+wB)/2
      delta_y = dy - (hA+hB)/2
      dist = delta_x**2 + delta_y**2
  else:
      dist = 0
  return 4*dist < wA**2+hA**2, 4*dist/(wA**2+hA**2)

def get_degree(boxA, boxB):
  def unit_vector(vector):
      """ Returns the unit vector of the vector.  """
      return vector / np.linalg.norm(vector)
  c1 = boxB[0]+boxB[2]-boxA[0]-boxA[2]
  c2 = boxB[1]+boxB[3]-boxA[1]-boxA[3]
  v1 = np.array((c1, c2))
  v2 = np.array((1,0))
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  degree = np.degrees(angle)
  # if degree == 180:
  #     degree = 179
  if c2 < 0:
      degree = 360 - degree
  if degree == 360:
      degree = 359
  d = degree // 22.5
  if d in [15,0]:
    ang = 2
  elif d in [7,8]:
    ang = 3
  elif d in [11,12]:
    ang = 4
  elif d in [3,4]:
    ang = 5
  else:
    ang = 0
  return ang, degree/360

def get_relation(boxes, w, h):
  box_num = boxes.shape[0]
  relation = np.zeros((box_num, box_num))
  for A in range(box_num):
    rel_score = {}
    for B in range(box_num):
      if A == B:
        relation[A,B] = 1
      else:
        boxA, Aw, Ah = norm(boxes[A].copy(),w,h)
        boxB, Bw, Bh = norm(boxes[B].copy(),w,h)
        if Bh < 0.5*Ah or Bh > 1.5*Ah:
          relation[A,B] = 0
        else:
          _io, _dist, io, dist = get_io(boxA, boxB) # iou>0 and io<0.5, iou = 0 and dist < 0.5
          if _io or _dist:
            angle, degree = get_degree(boxA, boxB)
            angle_score = degree+io+dist
            if angle == 0:
              relation[A,B] = 0
            else:
              if angle in rel_score.keys():
                if angle_score < rel_score[angle]:
                  relation[A,np.where(relation[A]==angle)[0]] = 0
                  relation[A,B] = angle
                  rel_score[angle] = angle_score
              else:
                relation[A,B] = angle
                rel_score[angle] = angle_score
    # print(relation[A])
    if np.where(relation[A]==2)[0] or np.where(relation[A]==3)[0]:
      relation[A,np.where(relation[A]==4)[0]] = 0
      relation[A,np.where(relation[A]==5)[0]] = 0
    # print(relation[A])
    # input()
  # if token B under token A has a left token, then remove the rel

  array1, array2 = np.where(relation==4)
  for i in range(array1.shape[0]):
    relB = relation[array2[i]]
    if np.where(relB==3)[0]:
      relation[array1[i], array2[i]] = 0

  return relation

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # clear ocr tokens and captions
  img_ids = []
  all_imgs = []
  all_imgs_test = []
  for img in tqdm(imgs):
    if img['image_id'] not in img_ids:
      img_ids.append(img['image_id'])
      split = img['set_name']
      if split == 'val':
        split = 'train'
      ocr_file = np.load(str(params['input_ocr_root'])+'/'+split+'_images/'+img['image_id']+'_info.npy',\
                         allow_pickle=True)[()]
      dup_ocr_tokens = ocr_file['ocr_tokens']
      # dup_ocr_tokens = [token.lower() for token in dup_ocr_tokens]
      dup_ocr_tokens = [clear_ocr(token).lower()for token in dup_ocr_tokens]
      ocr_tokens = dup_ocr_tokens[:params['max_ocr_len']]
      img['ocr_ids'] = list(range(len(ocr_tokens)))
      img['new_ocr_tokens'] = ocr_tokens
      ocr_boxes = ocr_file['ocr_boxes'][:params['max_ocr_len']]
      img['ocr_relations'] = get_relation(ocr_boxes, img['image_width'], img['image_height'])

      # # remove duplicated ocrs
      # ocr_tokens = []
      # ocr_ids = []
      # for iii, token in enumerate(dup_ocr_tokens):
      #   token = clear_ocr(token)
      #   if len(ocr_tokens) < params['max_ocr_len'] and token not in ocr_tokens:
      #     ocr_tokens.append(token.lower())
      #     ocr_ids.append(iii)
      # img['ocr_ids'] = ocr_ids
      # img['new_ocr_tokens'] = ocr_tokens
      # # print('\n')
      # # print(dup_ocr_tokens)
      # # print(ocr_tokens)

      if split == 'test':
        all_imgs_test.append(img)
      else:
        img['final_captions'] = []
        img['lower_final_captions'] = []
        for sent in img['reference_strs']:
          caption = clear_cap(sent, ocr_tokens)
          img['final_captions'].append(caption)
          img['lower_final_captions'].append([c.lower() for c in caption])
          # if not all(c.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', \
          # 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', \
          # 'x', 'y', 'z', ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
          # '"', ','] for c in sent.strip('.')):
          #   print('\n'+sent)
          #   print(caption)
          #   input()
        all_imgs.append(img)
  
  # mark ocr or vocab
  for img in tqdm(all_imgs):
    captions = img['final_captions']
    long_list = []
    for ix,sent in enumerate(captions):
      long_list += sent
    long_list = list(set(long_list))
    titled_words = [w.lower() for w in long_list if w.istitle()]
    titled_and_ocr = titled_words + img['new_ocr_tokens']

    img['token_marks'] = []
    for sent in captions:
      mark = []
      for token in sent:
        if token.istitle() or token.lower() in titled_and_ocr:
          mark.append('ocr')
        else:
          mark.append('vocab')
      img['token_marks'].append(mark)

  # count up the number of words
  counts = {}
  for img in all_imgs:
    for caption in img['lower_final_captions']:
      for w in caption:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))
  top20 = [w for (c,w) in cw[:20]]

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in all_imgs:
    for caption in img['lower_final_captions']:
      nw = len(caption)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
    vocab.append('ocrunk')

  return vocab, top20, all_imgs, all_imgs_test

def encode_captions(imgs, imgs_test, params, wtoi, top20):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['lower_final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  target_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1

  vocab_num = len(wtoi)+1

  for i,img in enumerate(imgs_test):
    ocr_tokens = img['new_ocr_tokens']
    ocrtoi = defaultdict(list)
    for i_ocr,ocr in enumerate(ocr_tokens):
      ocrtoi[ocr].append(i_ocr+vocab_num)
    # ocrtoi = {ocr:i_ocr+vocab_num for i_ocr,ocr in enumerate(ocr_tokens)}
    itoocr = {i_ocr+vocab_num:ocr for i_ocr,ocr in enumerate(ocr_tokens)}
    img['ocr_dict'] = itoocr

  for i,img in enumerate(imgs):
    # introduce ocr dict
    ocr_tokens = img['new_ocr_tokens']

    ocrtoi = defaultdict(list)
    for i_ocr,ocr in enumerate(ocr_tokens):
      ocrtoi[ocr].append(i_ocr+vocab_num)
    # ocrtoi = {ocr:i_ocr+vocab_num for i_ocr,ocr in enumerate(ocr_tokens)}
    itoocr = {i_ocr+vocab_num:ocr for i_ocr,ocr in enumerate(ocr_tokens)}
    img['ocr_dict'] = itoocr

    n = len(img['lower_final_captions'])
    assert n > 0, 'error: some image has no captions'

    # Li = np.zeros((n*50, max_length), dtype='uint32')
    caption_num = 0
    for j,s in enumerate(img['lower_final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      caption_matches = []
      for k,w in enumerate(s):
        if k < max_length:
          # decide word or ocr
          # if img['token_marks'][j][k] == 'ocr':
          #   if w not in ocrtoi.keys():
          #     w = 'ocrunk'
          #     Li[j,k] = wtoi[w]
          #   else:
          #     Li[j,k] = ocrtoi[w]
          # else:
          #   assert img['token_marks'][j][k] == 'vocab'
          #   if w not in wtoi.keys():
          #     w = 'UNK'
          #   Li[j,k] = wtoi[w]
          matched_inds = []
          if img['token_marks'][j][k] == 'ocr':
            if w not in ocrtoi.keys():
              w = 'ocrunk'
              matched_inds.append(wtoi[w])
            else:
              matched_inds.extend(ocrtoi[w])
          else:
            assert img['token_marks'][j][k] == 'vocab'
            if w not in wtoi.keys():
              w = 'UNK'
            matched_inds.append(wtoi[w])
          caption_matches.append(matched_inds)
      
      assert len(caption_matches) > 0
      matched_list = [()]
      for matched_inds in caption_matches:
        matched_list = [
          seq + (idx,)
          for seq in matched_list for idx in matched_inds
        ]
        if len(matched_list) > 100:
          matched_list = matched_list[:100]
      matched_array = np.zeros((len(matched_list), max_length), dtype='uint32')
      for m_ix, m in enumerate(matched_list):
        matched_array[m_ix, :len(m)] = np.array(m)
      
      target_array = np.zeros((len(matched_list), max_length, params['max_ocr_len']+1), dtype='uint32')
      for m_ix, matched_inds in enumerate(caption_matches):
        target_array[:,m_ix,:len(matched_inds)] = np.array(matched_inds)

      label_arrays.append(matched_array)
      target_arrays.append(target_array)
      caption_num += len(matched_list)
          
    # note: word indices are 1-indexed, and captions are padded with zeros
    # label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + caption_num - 1
    
    counter += caption_num
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  T = np.concatenate(target_arrays, axis=0)
  # assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert L.shape[0] == T.shape[0] == counter-1
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length, T

def main(params):

  train = json.load(open(params['input_json_PR']+'train.json', 'r'))
  val = json.load(open(params['input_json_PR']+'val.json', 'r'))
  test = json.load(open(params['input_json_PR']+'test.json', 'r'))
  imgs = train['data']
  imgs.extend(val['data'])
  imgs.extend(test['data'])

  seed(123) # make reproducible
  
  # create the vocab
  vocab, top20, all_imgs, all_imgs_test = build_vocab(imgs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
  
  # create folder
  f_name = params['output_vocab'].split('/')[-1]
  folder_name = params['output_vocab'].split(f_name)[0]
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)
  with open(params['output_vocab']+'.txt', 'w') as vocab_f:
    for vo in vocab:
      vocab_f.write(vo+'\n')
  json.dump(vocab, open(params['output_vocab']+'.json', 'w'))

  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length, T = encode_captions(all_imgs, all_imgs_test, params, wtoi, top20)

  # create output h5 file
  N = len(all_imgs)
  f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.create_dataset("target_labels", dtype='uint32', data=T)
  f_lb.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  for i,img in enumerate(all_imgs):
    
    jimg = {}
    jimg['split'] = img['set_name']
    jimg['file_path'] = img['image_path'] # copy it over, might need
    jimg['id'] = img['image_id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    jimg['width'] = img['image_width']
    jimg['height'] = img['image_height']
    jimg['distinct_ocr_ids'] = img['ocr_ids']
    jimg['distinct_ocr_tokens'] = img['new_ocr_tokens']
    jimg['ocr_relations'] = img['ocr_relations']
    jimg['ocr_dict'] = img['ocr_dict']
    out['images'].append(jimg)
  
  print('train+val image num: ', i)

  # create output test json file
  out_test = {}
  out_test['ix_to_word'] = itow # encode the (1-indexed) vocab
  out_test['images'] = []
  for i,img in enumerate(all_imgs_test):
    
    jimg = {}
    jimg['split'] = img['set_name']
    jimg['file_path'] = img['image_path'] # copy it over, might need
    jimg['id'] = img['image_id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    jimg['width'] = img['image_width']
    jimg['height'] = img['image_height']
    jimg['distinct_ocr_ids'] = img['ocr_ids']
    jimg['distinct_ocr_tokens'] = img['new_ocr_tokens']
    jimg['ocr_relations'] = img['ocr_relations']
    jimg['ocr_dict'] = img['ocr_dict']
    out_test['images'].append(jimg)
  
  # json.dump(out, open(params['output_json'], 'w'))
  pickle.dump(out, open(params['output_json'], 'wb'))
  pickle.dump(out_test, open(params['output_json_test'], 'wb'))
  print('wrote ', params['output_json'])
  print('test image num: ', i)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json_PR', default='TextCaps_0.1_', required=True, help='prefix of input json file to process into hdf5')
  parser.add_argument('--output_json', default='TC.json', help='output json file')
  parser.add_argument('--output_json_test', default='TC_test.json', help='output json file')
  parser.add_argument('--output_h5', default='TC', help='output h5 file')
  parser.add_argument('--output_vocab', default='TC_vocab', help='output vocab file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--input_ocr_root', default='m4c_textvqa_ocr_en_frcn_features', required=True, help='')

  # options
  parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=10, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--max_ocr_len', default=50, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
