import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# config_file = "../configs/e2e_faster_rcnn_R_101_FPN_1x.yaml"
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# from http://cocodataset.org/#explore?id=345434
image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
plt.imsave("test.jpg", image)


# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
plt.imsave("test_pred.jpg", predictions)

# load image and then run prediction
# pil_image = Image.open("demo_e2e_mask_rcnn_R_50_FPN_1x.png").convert("RGB")
# image = np.array(pil_image)[:, :, [2, 1, 0]]

# image = load_image("demo_e2e_mask_rcnn_R_50_FPN_1x.png")

# predictions = coco_demo.run_on_opencv_image(image)