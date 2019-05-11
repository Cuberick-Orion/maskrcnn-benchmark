# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms_inference
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

import pdb
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False
    ):

        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        '''
        (Pdb) box_regression.shape
        torch.Size([1000, 324])
        (Pdb) class_logits.shape
        torch.Size([1000, 81])
        '''

        class_logits, box_regression = x 
        class_prob = F.softmax(class_logits, -1)
        
        
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes] ## [(800, 801)]
        boxes_per_image = [len(box) for box in boxes] ## [1000]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0) ## <-> boxes[0].bbox ==> [1000,4]

        if self.cls_agnostic_bbox_reg: ## False
            box_regression = box_regression[:, -4:] ## box_regression -> torch.Size([1000, 324])
            ''' If cls_agnostic_bbox_reg then indice the 1000,4'''
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) ## some sort of decode
        if self.cls_agnostic_bbox_reg: ## False
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1] ## 81

        # pdb.set_trace()

        proposals = proposals.split(boxes_per_image, dim=0) ## split for each image, in the case of inference, no fucking difference...
        class_prob = class_prob.split(boxes_per_image, dim=0) ## same here

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        boxes = boxlist.bbox.reshape(-1, num_classes * 4) ## 1000, 324
        scores = boxlist.get_field("scores").reshape(-1, num_classes) ## 1000, 81

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh ## 0.05, total o 1118 in 1000,81
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1) ## extract nonzero indexs within the row
            scores_j = scores[inds, j] ## corresponding scores (use idx to extract value)
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4] ## corresponding box bound for the class

            '''
        (Pdb) inds_all[:,j].nonzero().squeeze(1)
        tensor([  1,   2,   3,   4,   8,   9,  14,  16,  19,  21,  26,  27,  29,  34,
                35,  48,  50,  51,  58,  60,  70,  73,  78,  94, 102, 103, 111, 117,
                150, 151, 158, 165, 168, 200, 255, 277, 387, 440, 519, 543, 685, 826],
            device='cuda:0')
        (Pdb) scores_j
        tensor([0.8351, 0.5963, 0.1416, 0.9766, 0.2436, 0.9077, 0.9410, 0.0868, 0.8560,
                0.8369, 0.4492, 0.7877, 0.1829, 0.8692, 0.4672, 0.5099, 0.0852, 0.0718,
                0.7845, 0.3410, 0.8635, 0.0678, 0.1538, 0.6063, 0.5969, 0.0561, 0.1027,
                0.6940, 0.4596, 0.2457, 0.3354, 0.0904, 0.2066, 0.0765, 0.0894, 0.1484,
                0.3581, 0.0611, 0.1441, 0.1787, 0.0848, 0.5040], device='cuda:0')
        (Pdb) boxes_j
        tensor([[292.5679, 358.4301, 702.7862, 679.2198],
                [284.0809, 354.4844, 715.9680, 688.7250],
                [208.1719, 277.7682, 712.2347, 733.3550],
                [301.0667, 375.4473, 703.9703, 657.2903],
                [238.4261, 335.3055, 718.5346, 718.3773],
                [303.8419, 357.7995, 704.4770, 673.5759],
                [314.2845, 367.7926, 695.3793, 659.8552],
                [236.4809, 309.3256, 719.0093, 735.2736],
                [300.9787, 377.5573, 703.8750, 642.1729],
                [308.4949, 365.9230, 673.5913, 672.4384],
                [311.5718, 366.4764, 690.5406, 624.3467],
                [303.0940, 378.4481, 694.2054, 649.5673],
                [297.3215, 360.7489, 626.4065, 643.7121],
                [298.0553, 368.0692, 699.8431, 672.4728],
                [308.8369, 391.8201, 697.8859, 654.0883],
                [314.7489, 376.5642, 709.4797, 646.4972],
                [271.0072, 380.6682, 654.8160, 656.8567],
                [279.1275, 291.4538, 682.3453, 730.8010],
                [309.2395, 371.4013, 708.0017, 654.3240],
                [330.9998, 377.2258, 698.7076, 635.0524],
                [306.4465, 358.4348, 690.7463, 695.8696],
                [320.0174, 353.9033, 663.6014, 654.0599],
                [348.4945, 365.5082, 708.4135, 686.0593],
                [327.9101, 369.9449, 689.8553, 635.1447],
                [326.7773, 364.1052, 701.8678, 672.2239],
                [268.0594, 374.2564, 596.6932, 675.7466],
                [340.9244, 386.6096, 689.0173, 601.6307],
                [309.1302, 354.8578, 642.1836, 655.9590],
                [309.2238, 360.3986, 670.9274, 631.3458],
                [293.9551, 375.4770, 703.0724, 705.6344],
                [280.2108, 388.4203, 692.8959, 674.3420],
                [266.7839, 366.9989, 718.5910, 738.8284],
                [352.5945, 369.1713, 698.8074, 626.6541],
                [277.4543, 335.3736, 657.1885, 638.1774],
                [301.1632, 350.7199, 503.9427, 658.0142],
                [307.8394, 340.1129, 579.8877, 645.4824],
                [289.1397, 368.0304, 704.0181, 703.2656],
                [351.4515, 372.6910, 691.4120, 617.6402],
                [310.0724, 378.7103, 618.0771, 682.5920],
                [336.0062, 385.3442, 693.1871, 620.2101],
                [294.1665, 357.9558, 720.8423, 728.6946],
                [299.7781, 373.7291, 682.2679, 687.0191]], device='cuda:0')
            '''

            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy") ## create empty
            boxlist_for_class.add_field("scores", scores_j) ## append the found boxes
            
            # pdb.set_trace()
            # print('In inference.py, self.nms = %f' % self.nms)

            boxlist_for_class = boxlist_nms_inference(
                boxlist_for_class, self.nms ## second layer nms thresholding, set to -1 for no-suppression
            )

            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)


        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        '''NOTE this is over all classes, i.e. the reduction does not case about classes
        Simply pull out all scores and sort them
        Then take the first 1000 (max cap)
        '''
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN ## True

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH ## default 0.05
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS ## default 0.5
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG ## default 100 Max number of detections
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG ## False default

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
