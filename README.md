# MaskRCNN Implementation from FAI

**Forked for feature extraction**

## Environment Requirement
 - pytorch Nightly
 - CUDA 9.0
 - GCC 5.5.0 (>=4.9)

## Setup
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd ~github
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

**Additional setup procedures**
```bash
# building dependencies
pip install -r requirements.txt
# demo_enlist dependencies
pip install requests
conda install -c conda-forge opencv
```
**For servers**
```bash
# Setting gcc version through bashrc symbolic link
```
Note we should set `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN` follow the rule in Single-GPU training.

## Evaluation
You can test your model directly on single or multiple gpus. Here is an example for Mask R-CNN R-50 FPN with the 1x schedule on 8 GPUS:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" TEST.IMS_PER_BATCH 16
```
To calculate mAP for each class, you can simply modify a few lines in [coco_eval.py](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/evaluation/coco/coco_eval.py). See [#524](https://github.com/facebookresearch/maskrcnn-benchmark/issues/524#issuecomment-475118810) for more details.

