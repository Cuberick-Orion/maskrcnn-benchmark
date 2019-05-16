## repurposed from Chris's code

'''
Extract features from VQA v2.0 dataset

Input: MSCOCO/train2014 and val2014
Output: each image corresponds a pkl file containing {"K" : K_current, "features": features_zip}
    where K is the number of objects (varies), and features is a scipy.csr_matrix of size 100x1024 (padded to K_max=100)

    The pkl file name is the image_id as the original MSCOCO dataset.
'''

import os, sys
import cv2
import json
import torch
import numpy as np
from scipy import sparse
import time
import pdb

import base64
import csv
import h5py
import pickle ## 2 -> 3
import utils

from modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg

from predictor0 import COCODemo

import argparse

# CUDA_VISIBLE_DEVICES=2 python extract_object_features_VQA2.py --img_dir "../data/images/COCO_val2014/" --feat_dir "../data/features/COCO_val2014_withboxlist/"

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate feature extraction for dataset using maskRCNN')

    parser.add_argument('--img_dir', dest='img_dir',
                        help='image directory (dataset)',
                        default="../data/images/COCO_train2014/", type=str)
    parser.add_argument('--feat_dir', dest='feat_dir',
                        help='feature file directory (output)',
                        default="../data/features/COCO_train2014_withboxlist/", type=str)    
    parser.add_argument('--model', dest='model',
                        help='model to use (config file)',
                        default="../configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml", type=str)
    # parser.add_argument('--model', dest='model',
    #                     help='model to use (config file)',
    #                     default="../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml", type=str)

    parser.add_argument('--K_max', dest='K_max',
                        help='K max',
                        default=100, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim',
                        help='dimension of the feature',
                        default=1024, type=int)
    parser.add_argument('--thres', dest='thres',
                        help='confidence threshold',
                        default=0.2, type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    ## define path
    ## Paths to frames, annotation("selected images") and features dst folder.
    pathTo = {}
    pathTo['frames']   = args.img_dir
    pathTo['features'] = args.feat_dir

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)
        print('[INFO] Create feature output directory (non exists)')
    config_file = args.model

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    min_image_size=800,
    confidence_threshold=0.2,


    ## build model
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=args.thres,
    )

    ## load images
    imglist = os.listdir(pathTo['frames'])
    num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    counter = 0

    print("[INFO] Starting processing images")
    while (num_images > 0):
        num_images -= 1

        img_file = os.path.join(pathTo['frames'], imglist[num_images])
        img_id = int(imglist[num_images][15:-4])

        dst_path = "{}/{}.pkl".format(pathTo['features'], str(img_id).replace(".jpg",""))
        ## check if file exists
        if os.path.isfile(dst_path):
            print("File exists for IMGID %i, skipped" % img_id)
        else:
            img_cv2 = cv2.imread(img_file)

            det_tic = time.time()
            result, top_predictions, predictions, features = coco_demo.run_on_opencv_image(img_cv2)
            det_toc = time.time()
            detect_time = det_toc - det_tic

            if len(predictions) != 0:
                features = features.cpu().numpy()
                K_current = features.shape[0]
                paddings = np.zeros((args.K_max-K_current, args.feature_dim))
                features_padded = np.vstack((features, paddings))
            else:
                K_current = 0
                print("[WARNING] image has ZERO features extracted, IMGID: %i" % img_id)
                features_padded = np.zeros((args.K_max, args.feature_dim)) ## 100x1024

            # pdb.set_trace()
            features_zip = sparse.csr_matrix(features_padded)
            pickle.dump({"K" : K_current, 'predictions': predictions, "features": features_zip}, \
                        open(dst_path,'wb'), protocol=3) ## NOTE protocol 2 for backwards compatibility
            counter += 1

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s | K={:d}   \r' \
                    .format(num_images + 1, len(imglist), detect_time, K_current))
            sys.stdout.flush()

    print("[DONE]!")