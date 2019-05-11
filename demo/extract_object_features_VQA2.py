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
# import time
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

# CUDA_VISIBLE_DEVICES=2 python extract_object_features_VQA2.py --img_dir "../data/images/COCO_val2014/" --feat_dir "../data/features/COCO_val2014/" --info_out_dir "../data/features/COCO_val2014/val2014_info.pkl" --imgids_dir "../data/features/COCO_val2014/val2014_ids.pkl" --imgindices_dir "../data/features/COCO_val2014/val2014_imgid2idx.pkl"

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
                        default="../data/features/COCO_train2014_v2/", type=str)
    parser.add_argument('--info_out_dir', dest='info_out_dir',
                        help='additional pickle output directory (output)',
                        default="../data/features/COCO_train2014_v2/train2014_info.pkl", type=str)
    parser.add_argument('--imgids_dir', dest='imgids_dir',
                        help='image ids directory (output)',
                        default="../data/features/COCO_train2014_v2/train2014_ids.pkl", type=str)
    parser.add_argument('--imgindices_dir', dest='imgindices_dir',
                        help='image ids directory (output)',
                        default="../data/features/COCO_train2014_v2/train2014_imgid2idx.pkl", type=str)
    parser.add_argument('--zero_feat_imgids_dir', dest='zero_feat_imgids_dir',
                        help='image ids with zero features extracted directory (output)',
                        default="../data/features/COCO_train2014_v2/train2014_imgidempty.pkl", type=str)


    
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

    ## load config
    # config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
    # config_file = "../configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
    # config_file = "../configs/caffe2/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml"
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

    ## init output dir
    # if not os.path.exists(pathTo["features"]):
    #     os.system("mkdir {}".format(pathTo['features']))
    
    ## init output h5
    # h_file = h5py.File(args.feat_file, "w")

    ## init image ids
    #### get train_ids.pkl and val_ids.pkl
    #### train image ids and val image ids

    # imgids = utils.load_imageid(args.img_dir) ## a simple function to iterate through all file ids in the folder
    # pickle.dump(imgids, open(args.imgids_dir, 'wb'))

    indices = {}
    # infos = {}

    # img_features = h_file.create_dataset(
    #     'image_features', (len(imgids), args.K_max, args.feature_dim), 'f') ## 82783 x 36 x 2048

    counter = 0

    print("[INFO] Starting processing images")
    while (num_images > 0):
        # total_tic = time.time()
        num_images -= 1

        img_file = os.path.join(pathTo['frames'], imglist[num_images])
        img_id = int(imglist[num_images][15:-4])

        dst_path = "{}/{}.pkl".format(pathTo['features'], str(img_id).replace(".jpg",""))
        ## check if file exists
        if os.path.isfile(dst_path):
            print("File exists for IMGID %i" % img_id)
            # imgids.remove(img_id)
        else:
            img_cv2 = cv2.imread(img_file)

            # pdb.set_trace()
            # det_tic = time.time()
            result, top_predictions, predictions, features = coco_demo.run_on_opencv_image(img_cv2)
            # det_toc = time.time()
            # detect_time = det_toc - det_tic

            # print(len(predictions))
            if len(predictions) != 0:
                features = features.cpu().numpy()
                K_current = features.shape[0]
                # assert K_current <= args.K_max
                # assert features.shape[1] == args.feature_dim

                # print("Shape of the feature rep. %s" % str(features.shape))

                paddings = np.zeros((args.K_max-K_current, args.feature_dim))
                # pdb.set_trace()
                features_padded = np.vstack((features, paddings))
                # assert features_padded.shape[0] == args.K_max
            else:
                K_current = 0
                print("[WARNING] image has ZERO features extracted, IMGID: %i" % img_id)
                features_padded = np.zeros((args.K_max, args.feature_dim)) ## 100x1024

            features_zip = sparse.csr_matrix(features_padded)

            # imgids.remove(img_id)
            # indices[img_id] = counter
            # infos[img_id] = {'K':K_current, 'predictions': predictions}
            # img_features[counter, :, :] = features_padded ## 100x1024
            
            pickle.dump({"K" : K_current, "features": features_zip}, \
                        open(dst_path,'wb'))

            # pdb.set_trace()
            counter += 1


            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s | K={:d}   \r' \
                    .format(num_images + 1, len(imglist), 0.000, K_current))
            sys.stdout.flush()

    # pickle.dump(indices, open(args.imgindices_dir, 'wb'))
    # pickle.dump(infos, open(args.info_out_dir, 'wb'))
    # pdb.set_trace()
    # h_file.close()

    # print(">>>>> All zero featured images IDs:")
    # print(imgids)
    # pickle.dump(imgids, open(args.zero_feat_imgids_dir, 'wb'))
    
    print("[DONE]!")