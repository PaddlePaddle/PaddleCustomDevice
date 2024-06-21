# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import unittest
import numpy as np
import math
import paddle
import copy
from op_test import OpTest

paddle.enable_static()


def anchor_generator_in_python(
    input_feat, anchor_sizes, aspect_ratios, variances, stride, offset
):
    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    layer_h = input_feat.shape[2]
    layer_w = input_feat.shape[3]
    out_dim = (layer_h, layer_w, num_anchors, 4)
    out_anchors = np.zeros(out_dim).astype("float32")

    for h_idx in range(layer_h):
        for w_idx in range(layer_w):
            x_ctr = (w_idx * stride[0]) + offset * (stride[0] - 1)
            y_ctr = (h_idx * stride[1]) + offset * (stride[1] - 1)
            idx = 0
            for r in range(len(aspect_ratios)):
                ar = aspect_ratios[r]
                for s in range(len(anchor_sizes)):
                    anchor_size = anchor_sizes[s]
                    area = stride[0] * stride[1]
                    area_ratios = area / ar
                    base_w = np.round(np.sqrt(area_ratios))
                    base_h = np.round(base_w * ar)
                    scale_w = anchor_size / stride[0]
                    scale_h = anchor_size / stride[1]
                    w = scale_w * base_w
                    h = scale_h * base_h
                    out_anchors[h_idx, w_idx, idx, :] = [
                        (x_ctr - 0.5 * (w - 1)),
                        (y_ctr - 0.5 * (h - 1)),
                        (x_ctr + 0.5 * (w - 1)),
                        (y_ctr + 0.5 * (h - 1)),
                    ]
                    idx += 1

    # set the variance.
    out_var = np.tile(variances, (layer_h, layer_w, num_anchors, 1))
    out_anchors = out_anchors.astype("float32")
    out_var = out_var.astype("float32")
    return out_anchors, out_var


def clip_tiled_boxes(boxes, im_shape, pixel_offset=True):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert (
        boxes.shape[1] % 4 == 0
    ), "boxes.shape[1] is {:d}, but must be divisible by 4.".format(boxes.shape[1])
    offset = 1 if pixel_offset else 0
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - offset), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - offset), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - offset), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - offset), 0)
    return boxes


def box_coder(all_anchors, bbox_deltas, variances, pixel_offset=True):
    """
    Decode proposals by anchors and bbox_deltas from RPN
    """
    offset = 1 if pixel_offset else 0
    # proposals: xmin, ymin, xmax, ymax
    proposals = np.zeros_like(bbox_deltas, dtype=np.float32)

    # anchor_loc: width, height, center_x, center_y
    anchor_loc = np.zeros_like(bbox_deltas, dtype=np.float32)

    anchor_loc[:, 0] = all_anchors[:, 2] - all_anchors[:, 0] + offset
    anchor_loc[:, 1] = all_anchors[:, 3] - all_anchors[:, 1] + offset
    anchor_loc[:, 2] = all_anchors[:, 0] + 0.5 * anchor_loc[:, 0]
    anchor_loc[:, 3] = all_anchors[:, 1] + 0.5 * anchor_loc[:, 1]

    # predicted bbox: bbox_center_x, bbox_center_y, bbox_width, bbox_height
    pred_bbox = np.zeros_like(bbox_deltas, dtype=np.float32)
    if variances is not None:
        for i in range(bbox_deltas.shape[0]):
            pred_bbox[i, 0] = (
                variances[i, 0] * bbox_deltas[i, 0] * anchor_loc[i, 0]
                + anchor_loc[i, 2]
            )
            pred_bbox[i, 1] = (
                variances[i, 1] * bbox_deltas[i, 1] * anchor_loc[i, 1]
                + anchor_loc[i, 3]
            )
            pred_bbox[i, 2] = (
                math.exp(
                    min(variances[i, 2] * bbox_deltas[i, 2], math.log(1000 / 16.0))
                )
                * anchor_loc[i, 0]
            )
            pred_bbox[i, 3] = (
                math.exp(
                    min(variances[i, 3] * bbox_deltas[i, 3], math.log(1000 / 16.0))
                )
                * anchor_loc[i, 1]
            )
    else:
        for i in range(bbox_deltas.shape[0]):
            pred_bbox[i, 0] = bbox_deltas[i, 0] * anchor_loc[i, 0] + anchor_loc[i, 2]
            pred_bbox[i, 1] = bbox_deltas[i, 1] * anchor_loc[i, 1] + anchor_loc[i, 3]
            pred_bbox[i, 2] = (
                math.exp(min(bbox_deltas[i, 2], math.log(1000 / 16.0)))
                * anchor_loc[i, 0]
            )
            pred_bbox[i, 3] = (
                math.exp(min(bbox_deltas[i, 3], math.log(1000 / 16.0)))
                * anchor_loc[i, 1]
            )
    proposals[:, 0] = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
    proposals[:, 1] = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
    proposals[:, 2] = pred_bbox[:, 0] + pred_bbox[:, 2] / 2 - offset
    proposals[:, 3] = pred_bbox[:, 1] + pred_bbox[:, 3] / 2 - offset

    return proposals


def iou(box_a, box_b, pixel_offset=True):
    """
    Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])
    offset = 1 if pixel_offset else 0
    area_a = (ymax_a - ymin_a + offset) * (xmax_a - xmin_a + offset)
    area_b = (ymax_b - ymin_b + offset) * (xmax_b - xmin_b + offset)
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa + offset, 0.0) * max(yb - ya + offset, 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)

    return iou_ratio


def nms(boxes, scores, nms_threshold, eta=1.0, pixel_offset=True):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        nms_threshold: (float) The overlap thresh for suppressing unnecessary
            boxes.
        eta: (float) The parameter for adaptive NMS.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()

    sorted_indices = np.argsort(-all_scores, axis=0, kind="mergesort")
    sorted_scores = all_scores[sorted_indices]
    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx], pixel_offset=pixel_offset)
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold *= eta
    return selected_indices


def python_generate_proposals_v2(
    scores,
    bbox_deltas,
    img_size,
    anchors,
    variances,
    pre_nms_top_n=6000,
    post_nms_top_n=1000,
    nms_thresh=0.5,
    min_size=0.1,
    eta=1.0,
    pixel_offset=False,
    return_rois_num=True,
):
    rpn_rois, rpn_roi_probs, rpn_rois_num = paddle.vision.ops.generate_proposals(
        scores,
        bbox_deltas,
        img_size,
        anchors,
        variances,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        eta=eta,
        pixel_offset=pixel_offset,
        return_rois_num=return_rois_num,
    )
    return rpn_rois, rpn_roi_probs


def generate_proposals_v2_in_python(
    scores,
    bbox_deltas,
    im_shape,
    anchors,
    variances,
    pre_nms_topN,
    post_nms_topN,
    nms_thresh,
    min_size,
    eta,
    pixel_offset,
):
    all_anchors = anchors.reshape(-1, 4)
    rois = np.empty((0, 5), dtype=np.float32)
    roi_probs = np.empty((0, 1), dtype=np.float32)

    rpn_rois = []
    rpn_roi_probs = []
    rois_num = []
    num_images = scores.shape[0]
    for img_idx in range(num_images):
        img_i_boxes, img_i_probs = proposal_for_one_image(
            im_shape[img_idx, :],
            all_anchors,
            variances,
            bbox_deltas[img_idx, :, :, :],
            scores[img_idx, :, :, :],
            pre_nms_topN,
            post_nms_topN,
            nms_thresh,
            min_size,
            eta,
            pixel_offset,
        )
        rois_num.append(img_i_probs.shape[0])
        rpn_rois.append(img_i_boxes)
        rpn_roi_probs.append(img_i_probs)
    # print('rois_num',rois_num)
    return rpn_rois, rpn_roi_probs, rois_num


def proposal_for_one_image(
    im_shape,
    all_anchors,
    variances,
    bbox_deltas,
    scores,
    pre_nms_topN,
    post_nms_topN,
    nms_thresh,
    min_size,
    eta,
    pixel_offset,
):
    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #   - bbox deltas will be (4 * A, H, W) format from conv output
    #   - transpose to (H, W, 4 * A)
    #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
    #     in slowest to fastest order to match the enumerated anchors
    bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape(-1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    variances = variances.reshape(-1, 4)
    # Same story for the scores:
    #   - scores are (A, H, W) format from conv output
    #   - transpose to (H, W, A)
    #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
    #     to match the order of anchors and bbox_deltas
    scores = scores.transpose((1, 2, 0)).reshape(-1, 1)

    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN (e.g. 6000)
    if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
        order = np.argsort(-scores.squeeze())
    else:
        # Avoid sorting possibly large arrays;
        # First partition to get top K unsorted
        # and then sort just those
        inds = np.argpartition(-scores.squeeze(), pre_nms_topN)[:pre_nms_topN]
        order = np.argsort(-scores[inds].squeeze())
        order = inds[order]
    scores = scores[order, :]
    bbox_deltas = bbox_deltas[order, :]
    all_anchors = all_anchors[order, :]
    # print('all_anchors ',all_anchors.shape,bbox_deltas.shape)
    proposals = box_coder(all_anchors, bbox_deltas, variances, pixel_offset)

    # print('box_coder proposals ',proposals.shape,proposals)
    # clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    proposals = clip_tiled_boxes(proposals, im_shape, pixel_offset)
    # print('clip proposals ',proposals)
    # remove predicted boxes with height or width < min_size
    keep = filter_boxes(proposals, min_size, im_shape, pixel_offset)
    # print('keep index',keep)
    # print('need proposals num',len(keep))
    if len(keep) == 0:
        proposals = np.zeros((1, 4)).astype("float32")
        scores = np.zeros((1, 1)).astype("float32")
        return proposals, scores
    proposals = proposals[keep, :]
    scores = scores[keep, :]

    # apply loose nms (e.g. threshold = 0.7)
    # take post_nms_topN (e.g. 1000)
    # return the top proposals
    if nms_thresh > 0:
        keep = nms(
            boxes=proposals,
            scores=scores,
            nms_threshold=nms_thresh,
            eta=eta,
            pixel_offset=pixel_offset,
        )
        if post_nms_topN > 0 and post_nms_topN < len(keep):
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep, :]
    return proposals, scores


def filter_boxes(boxes, min_size, im_shape, pixel_offset=True):
    """Only keep boxes with both sides >= min_size and center within the image."""
    # Scale min_size to match image scale
    min_size = max(min_size, 1.0)
    offset = 1 if pixel_offset else 0
    ws = boxes[:, 2] - boxes[:, 0] + offset
    hs = boxes[:, 3] - boxes[:, 1] + offset
    if pixel_offset:
        x_ctr = boxes[:, 0] + ws / 2.0
        y_ctr = boxes[:, 1] + hs / 2.0
        keep = np.where(
            (ws >= min_size)
            & (hs >= min_size)
            & (x_ctr < im_shape[1])
            & (y_ctr < im_shape[0])
        )[0]
    else:
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


class TestGenerateProposalsV2Op(OpTest):
    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {
            "Scores": self.scores,
            "BboxDeltas": self.bbox_deltas,
            "ImShape": self.im_shape.astype(np.float32),
            "Anchors": self.anchors,
            "Variances": self.variances,
        }

        self.attrs = {
            "pre_nms_topN": self.pre_nms_topN,
            "post_nms_topN": self.post_nms_topN,
            "nms_thresh": self.nms_thresh,
            "min_size": self.min_size,
            "eta": self.eta,
            "pixel_offset": self.pixel_offset,
        }

        self.outputs = {
            "RpnRois": self.rpn_rois[0],
            "RpnRoiProbs": self.rpn_roi_probs[0],
        }
        # print('Scores',self.scores.shape,flush=True)
        # print('BboxDeltas',self.bbox_deltas.shape,flush=True)
        # print('anchors',self.anchors.shape,flush=True)
        # print('Variances',self.variances.shape,flush=True)
        # print('RpnRois',self.rpn_rois[0].shape,flush=True)
        # print('RpnRoiProbs',self.rpn_roi_probs[0].shape,flush=True)
        # print('RpnRois',self.rpn_rois[1].shape,flush=True)
        # print('RpnRoiProbs',self.rpn_roi_probs[1].shape,flush=True)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_sdaa(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

    def setUp(self):
        self.op_type = "generate_proposals_v2"
        self.python_api = python_generate_proposals_v2
        self.init_sdaa()
        self.set_data()

    def init_test_params(self):
        self.pre_nms_topN = 12000  # train 12000, test 2000
        self.post_nms_topN = 5000  # train 6000, test 1000
        self.nms_thresh = 0.7
        self.min_size = 3.0
        self.eta = 1.0
        self.pixel_offset = True

    def init_test_input(self):
        batch_size = 1
        input_channels = 20
        layer_h = 16
        layer_w = 16
        input_feat = np.random.random(
            (batch_size, input_channels, layer_h, layer_w)
        ).astype("float32")
        self.anchors, self.variances = anchor_generator_in_python(
            input_feat=input_feat,
            anchor_sizes=[16.0, 32.0],
            aspect_ratios=[0.5, 1.0],
            variances=[1.0, 1.0, 1.0, 1.0],
            stride=[16.0, 16.0],
            offset=0.5,
        )
        self.im_shape = np.array([[64, 64]]).astype("float32")
        num_anchors = self.anchors.shape[2]

        self.scores = np.random.random(
            (batch_size, num_anchors, layer_h, layer_w)
        ).astype("float32")
        self.bbox_deltas = np.random.random(
            (batch_size, num_anchors * 4, layer_h, layer_w)
        ).astype("float32")

    def init_test_output(self):
        (
            self.rpn_rois,
            self.rpn_roi_probs,
            self.rois_num,
        ) = generate_proposals_v2_in_python(
            self.scores,
            self.bbox_deltas,
            self.im_shape,
            self.anchors,
            self.variances,
            self.pre_nms_topN,
            self.post_nms_topN,
            self.nms_thresh,
            self.min_size,
            self.eta,
            self.pixel_offset,
        )


class TestGenerateProposalsV2OpNoOffset(TestGenerateProposalsV2Op):
    def init_test_params(self):
        self.pre_nms_topN = 12000  # train 12000, test 2000
        self.post_nms_topN = 5000  # train 6000, test 1000
        self.nms_thresh = 0.7
        self.min_size = 3.0
        self.eta = 1.0
        self.pixel_offset = False


if __name__ == "__main__":
    unittest.main()
