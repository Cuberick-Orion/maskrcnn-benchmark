## repurposed from Chris's code



import os, sys
import cv2
import json
import torch
import numpy as np
import time
import pdb

from modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg

from predictor0 import COCODemo

## Paths to frames, annotation("selected images") and features dst folder.
pathTo = {}
pathTo['frames']   = "../data/images"
pathTo['features'] = "../data/features"

# config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
# config_file = "../configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
# config_file = "../configs/caffe2/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml"
config_file = "../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"

cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
min_image_size=800,
confidence_threshold=0.5,

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

## load images
imglist = os.listdir(pathTo['frames'])
num_images = len(imglist)

print('Loaded Photo: {} images.'.format(num_images))

## init output dir
if not os.path.exists(pathTo["features"]):
    os.system("mkdir {}".format(pathTo['features']))

while (num_images > 0):
    total_tic = time.time()
    num_images -= 1

    im_file = os.path.join(pathTo['frames'], imglist[num_images])
    dst_path = os.path.join(pathTo['features'], imglist[num_images]+'.npy')
    img = cv2.imread(im_file)

    # pdb.set_trace()
    det_tic = time.time()
    result, top_predictions, predictions, features = coco_demo.run_on_opencv_image(img)
    det_toc = time.time()
    detect_time = det_toc - det_tic

    print(len(predictions), dst_path)
    if len(predictions) != 0:
        features = features.cpu().numpy()
        print("Shape of the feature rep. %s" % str(features.shape))
        print("predictions {}".format(predictions))
        np.save(dst_path, {"predictions":predictions, "features": features})
        # print(predictions, top_predictions, features.shape, img_path, dst_path)
        # pdb.set_trace()

    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
               .format(num_images + 1, len(imglist), detect_time))
    sys.stdout.flush()
