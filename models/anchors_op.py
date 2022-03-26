#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Version : 0.0.0
''' Inherent libs '''
import os
''' Third libs '''
import numpy as np
import torch
from mmdet.core import AnchorGenerator as mmdet_AnchorGenerator
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import RoIAlign
from torchvision.ops import MultiScaleRoIAlign
from scipy.spatial import distance_matrix
''' Local libs '''


# this class includes all the required operations for the anchors
class AnchorsOperator(object):
    def __init__(self):

        self.roi_based_featmap_name = ["avg_pool"]

    def define_anchor_generator(self,
                                strides,
                                ratios,
                                scales,
                                base_sizes,
                                center_offset=0.):
        """[define and obtain an anchor generator to generate anchors for any feature map]

        Args:
            strides (list[int]): [a list that each item is the strides between the feature map and the 
                                image in which anchors are generated,
                                for example: 
                                    feature map: is <5 x 5> that is the size <height x width> of the output of pool9 layer
                                                in the ResNet.
                                    strides: is 16 that shows the size changes between input image and the pool9 features map  in ResNet.]
                                        
            ratios (list[float]): [a list that each item present the ratio between the height and width
                                of anchors in the corresponding single level (feature map).]
            scales (list[int]): [a list that each item shows the anchor scale for anchors in a single level (feature map).]
            base_sizes (list[int]): [a list that each item provides the basic size of the anchor. 
                                    Each item is a single integar as it assumes the anchor are in equal height and width. 
                                    We can change the size of anchors by using ratios and scales]
            center_offset (list[tuple[float, float]] | None): The centers of the anchor
                            relative to the feature grid center in multiple feature levels.
                            By default it is set to be None and not used. If a list of tuple of
                            float is given, they will be used to shift the centers of anchors.
        """

        anchor_generator = mmdet_AnchorGenerator(strides=strides,
                                                 ratios=ratios,
                                                 scales=scales,
                                                 base_sizes=base_sizes,
                                                 center_offset=center_offset)

        return anchor_generator

    def define_tv_anchor_generator(self,
                                   anchor_sizes=[(32, )],
                                   aspect_ratios=[(1.0)],
                                   anchor_intervals=[(64, 64)]):
        """[define and obtain an anchor generator to generate anchors for any feature map]

        Args:
            base_sizes (list[tuple(int,)]): [a list that each item provides the basic size of the anchor. 
                                    Each item is a single integar as it assumes the anchor are in equal height and width. 
                                    We can change the size of anchors by using ratios and scales]
                            for example: anchor_sizes = [(32,), (64,), (128,), (256,), (512,)]

            ratios (list[float]): [a list that each item present the ratio between the height and width
                                of anchors in the corresponding single level (feature map).]
                            for example: aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_intervals (list[tuple(int,int)])
        Return:
            rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
                                                    maps.
            
            Note: To generate the anchors, you need to use rpn_anchor_generator(images, features)
                        - images (ImageList): images for which we want to compute the predictions
                        - features (OrderedDict[Tensor]): features computed from the images that are
                            used for computing the predictions. Each tensor in the list
                            correspond to different feature levels
                        anchors = rpn_anchor_generator(images, features)
        """
        self.anchor_intervals = anchor_intervals
        aspect_ratios = aspect_ratios * len(anchor_sizes)

        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        print("-"*10 + "This defined anchor generator can produce the number of " \
                + str(rpn_anchor_generator.num_anchors_per_location())\
                 + " anchor per location")
        return rpn_anchor_generator

    def define_box_roi_align(self,
                             featmap_names=["avg_pool"],
                             output_size=[5],
                             sampling_ratio=2):
        '''
            Note: The shape of rois is expected to be (m: num-rois, 5), and each roi is set as below:
                    (batch_index, x_min, y_min, x_max, y_max). RoIAlign
            Example:
                >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], [3], 2)
                >>> i = OrderedDict()
                >>> i['feat1'] = torch.rand(1, 5, 64, 64)
                >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
                >>> i['feat3'] = torch.rand(1, 5, 16, 16)
                >>> # create some random bounding boxes
                >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
                >>> # original image size, before computing the feature maps
                >>> image_sizes = [(512, 512)]
                >>> output = m(i, [boxes], image_sizes)
                >>> print(output.shape)
                >>> torch.Size([6, 5, 3, 3])
        '''
        self.roi_based_featmap_name = featmap_names
        roi_aligner = MultiScaleRoIAlign(featmap_names=featmap_names,
                                         output_size=output_size,
                                         sampling_ratio=sampling_ratio)

        return roi_aligner

    def generate_anchors(self, anchor_generator, feature_maps_size):
        """[Generate anchors]

        Args:
            anchor_generator (mmdet.core.anchorGenerator): [an defined anchor generator]
            feature_maps_size (list(tuple[int, int])): [the size of the feature map from which the centers of anchors are mapped to
                                                the target map (defined by the strides) to generate anchors]
        Return:
            target_anchors (list[torch.Tensor]): [Anchors in multiple feature levels. \
                                                The sizes of each tensor should be [N, 4], where \
                                                N = width * height * num_base_anchors, width and height \
                                                are the sizes of the corresponding feature level, \
                                                num_base_anchors is the number of anchors for that level.]
        """
        target_anchors = anchor_generator.grid_anchors(feature_maps_size,
                                                       device='cpu')

        return target_anchors

    def remove_anchors_with_intervals(self, trandformed_batch_images,
                                      anchors_h_w, images_anchors,
                                      images_anchors_interval, strides):
        """[remove anchors based on the required interval between anchors]

        Args:
            trandformed_batch_images (ImageList): the batch of image after the transformation, 
                                                    ImageList: can be visited by ImageList.tensor, ImageList.image_sizes
            anchors_h_w (list[list([int, int])]): the h and w of the anchor matrix, N(=wxh) x 4
            images_anchors (list[torch.Tensor]): [each item is a tensor containing the boxes of the image]
            images_anchors_interval ([list(Tuple(int, int))]): [each item is an integer presenting the required interval between anchors]
            strides  ([list(Tuple(int, int))]): [each item shows the strides of anchors]
        """
        lefted_images_anchors = list()

        for i, (image_height, image_width) in enumerate(
                trandformed_batch_images.image_sizes):

            image_anchors_h, image_anchors_w = anchors_h_w[i]

            stride_h, stride_w = strides[i]

            if len(images_anchors_interval) == 1:  # we only use one
                image_anchors_interval_h, image_anchors_interval_w = images_anchors_interval[
                    0]
            else:
                image_anchors_interval_h, image_anchors_interval_w = images_anchors_interval[
                    i]

            image_anchors = images_anchors[i]

            w_skips = int(image_anchors_interval_w / stride_w)
            h_skips = int(image_anchors_interval_h / stride_h)

            wbased_maintained_idxs = list()
            for y_j in range(image_anchors_h):
                baseline_idx = y_j * image_anchors_w
                keep_idxs = list(
                    range(baseline_idx, baseline_idx + image_anchors_w,
                          w_skips))

                wbased_maintained_idxs = wbased_maintained_idxs + keep_idxs

                # border_idx = baseline_idx+image_feature_w-(w_skips-1)
                # if border_idx not in wbased_maintained_idxs:
                #     wbased_maintained_idxs.append(border_idx)

            hbased_maintained_idxs = list()
            for x_i in range(image_anchors_w):
                baseline_idx = x_i
                keep_idxs = list(range(0, image_anchors_h + 1, h_skips))

                # border_idx = image_anchors_h-(h_skips-1)
                # if border_idx not in keep_idxs:
                #     keep_idxs.append(border_idx)

                keep_idxs = np.array(
                    keep_idxs) * image_anchors_w + baseline_idx

                hbased_maintained_idxs = hbased_maintained_idxs + keep_idxs.tolist(
                )

            maintained_idxs = set(wbased_maintained_idxs).intersection(
                hbased_maintained_idxs)
            maintained_idxs = torch.tensor(list(maintained_idxs))

            #print("wbased_maintained_idxs: ", wbased_maintained_idxs)

            lefted_images_anchors.append(image_anchors[maintained_idxs])

        return lefted_images_anchors

    def filter_anchors(self, images_size, images_anchors, allowed_border=0):
        """[Filter out the anchors based on the size of the board]

        Args:
            images_size (list(Tuple[int, int])): [the size of each image]
            images_anchors (list[torch.Tensor]): [description]
        """
        filtered_images_anchors = images_anchors.copy()

        maintained_anchors_id = list()
        for anchors_i in range(len(images_anchors)):

            anchors = images_anchors[anchors_i]
            img_size = images_size[anchors_i]

            keep = ((anchors[:, 0] >= -allowed_border) &
                    (anchors[:, 1] >= -allowed_border) &
                    (anchors[:, 2] < img_size[1] + allowed_border) &
                    (anchors[:, 3] < img_size[0] + allowed_border))

            maintained_anchors_id.append(np.where(keep.numpy() == True)[0])

            inds_inside = torch.nonzero(keep).view(-1)

            # keep only inside anchors
            remained_anchors = anchors[inds_inside, :]
            filtered_images_anchors[anchors_i] = remained_anchors

        return filtered_images_anchors, maintained_anchors_id

    def obtain_h_w_from_anchors_id(self, maintained_anchors_id):
        """[obtain the w, h of the anchors matrix (N=wxh) x 4]

        Args:
            maintained_anchors_id ([list]): [each item contains the idxs of the maintained anchors]
        """

        anchors_h_w = list()
        for m_anchors_idx in maintained_anchors_id:
            counted_number = 0
            anchor_id_holder = -1
            number_of_anchors = len(m_anchors_idx)
            for anchor_idx in m_anchors_idx:
                if anchor_id_holder == -1:
                    counted_number += 1
                    anchor_id_holder = anchor_idx
                else:
                    if anchor_id_holder + 1 == anchor_idx:
                        counted_number += 1
                        anchor_id_holder = anchor_idx
                    else:
                        break
            anchors_h_w.append(
                [int(number_of_anchors / counted_number), counted_number])

        return anchors_h_w


if __name__ == "__main__":
    anchor_op = AnchorsOperator()

    anchor_generator = anchor_op.define_anchor_generator(strides=[16],
                                                         ratios=[1.],
                                                         scales=[1.],
                                                         base_sizes=[64])
    anchors = anchor_generator.grid_anchors([(7, 7)], device='cpu')

    filtered_anchors = anchor_op.filter_anchors(board_size=(128, 128),
                                                all_anchors=anchors)

    filtered_anchors = filtered_anchors[0]
    filtered_anchors = torch.cat(
        [torch.zeros((filtered_anchors.size(0), 1)), filtered_anchors], 1)

    print("filtered_anchors: ", filtered_anchors)

    # roi_extractor = anchor_op.define_bbox_roi_extractor(out_size=7, out_channels=256, featmap_strides=[32])
    # print(roi_extractor)

    feats = torch.tensor(
        [[[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28],
           [29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42],
           [43, 44, 45, 46, 47, 48, 49]]]],
        dtype=torch.float)
    print("feats: ", feats)
    print("feats: ", feats.shape)

    roi_aligner = anchor_op.define_box_roi_align()
    roi_aligner(feats, filtered_anchors)