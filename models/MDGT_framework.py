#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Inherent libs '''
import os
from collections import OrderedDict
''' Third libs '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional

import torchvision.models as models
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
''' Local libs '''


class MDGTFramework(nn.Module):
    """
    A standard multi-modal dynamic graph transformer framework for progressive learning on visual grounding. Base for this and many other modules.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    """
    def __init__(self,
                 backbone_visual_net_name,
                 language_module,
                 anchor_operator,
                 physical_batch_graph_operator,
                 mm_graph,
                 transformation_head,
                 flags,
                 is_eval=True):
        super(MDGTFramework, self).__init__()
        self.backbone_visual_net_name = backbone_visual_net_name

        if self.backbone_visual_net_name is "resnset18":
            resnet18 = models.resnet18(pretrained=True)
            self.backbone_visual_net = torch.nn.Sequential(
                *(list(resnet18.children())[:-2]))  # remove the final fc layer
            # print("self.backbone_visual_net: ", self.backbone_visual_net)
        else:
            raise NotImplementedError()

        # here we set the mean and std to 0 because the normialization has been operated in the preprocssing part
        self.transform = GeneralizedRCNNTransform(flags.min_size,
                                                  flags.max_size,
                                                  image_mean=flags.image_mean,
                                                  image_std=flags.image_std)

        self.language_module = language_module
        self.anchor_operator = anchor_operator
        self.physical_batch_graph_operator = physical_batch_graph_operator

        self.an_generator = self.anchor_operator.define_tv_anchor_generator(
            anchor_sizes=flags.anchor_sizes,
            aspect_ratios=flags.aspect_ratios,
            anchor_intervals=flags.anchor_intervals)
        self.an_roi_aligner = self.anchor_operator.define_box_roi_align(
            featmap_names=flags.featmap_names,
            output_size=flags.roi_output_size,
            sampling_ratio=flags.roi_sampling_ratio)
        if not is_eval:
            self.batch_size = flags.batch_size
        else:
            self.batch_size = 1

        self.mm_graph = mm_graph
        self.transformation_head = transformation_head

        self.original_image_size = torch.jit.annotate(
            List[Tuple[int, int]], [])  # List[Tuple[int, int]]

    def get_orinigal_image_size(self, batch_images):
        images_size = torch.jit.annotate(List[Tuple[int, int]],
                                         [])  # List[Tuple[int, int]]
        for img in batch_images:
            val = img.shape[-2:]
            assert len(val) == 2
            images_size.append((val[0], val[1]))  # List[Tuple[int, int]]
        return images_size

    def compute_transformation_coefficients(self, src_boxes, trg_boxes):
        hs = self.original_image_size[0][0]
        ws = self.original_image_size[0][1]
        transformation_coefficients = self.transformation_head.get_expected_trans_coeffs(
            boxes=src_boxes, groundtruth_boxes=trg_boxes, Hs=hs, Ws=ws)

        return transformation_coefficients

    def backbone_forward(self, batch_images, batch_images_boxes,
                         batch_captions, batch_captions_phrases):
        """[Obtain the multi-modal features through the backbone networks]

        Args:
            batch_images (list[Tensor]): [a list of tensors, each tensor contains the image data with
                                            shape <H, W, 3>, ] 
            batch_images_boxes (list[torch.Tensor]): [~~]
            batch_captions ([list]): [a list in which each item is the list that contains the caption of the image,
                                Thus, this list contains the captions of one batch of data]
            batch_captions_phrases ([list]): [a nested list in which each item is a list that contains the corresponding phrases of one image]

        Return:
            trandformed_batch_images (ImageList): the batch of image after the transformation, 
                                                    ImageList: can be visited by ImageList.tensors, ImageList.image_sizes
            trandformed_batch_boxes (OrderedDict[boxes: torch.Tensor]): the ground-truth boxes after the transformation
            batch_images_features (OrderedDict[featmap_name: torch.Tensor]): the features of the batch of images,
                                                                there can be serveral feature maps if we set more
            [integrated_captions_phrases_embds]: [a torch with shape <batch_size, max_number_of_phrases, embd_dim>
                                                    padding with zeros]
            [captions_phrases_mask]: [a nested list that each item is a list containing the padding information of the corresponding caption]
        """
        # convert to <3, H, W>
        batch_images = [img_data.permute(2, 0, 1) for img_data in batch_images]

        self.original_image_size = self.get_orinigal_image_size(batch_images)

        trandformed_batch_images, trandformed_batch_boxes = self.transform(
            batch_images, [{
                "boxes": image_boxes
            } for image_boxes in batch_images_boxes])

        batch_images_features = self.backbone_visual_net(
            trandformed_batch_images.tensors)
        if isinstance(batch_images_features, torch.Tensor):
            batch_images_features = OrderedDict([
                (self.anchor_operator.roi_based_featmap_name[0],
                 batch_images_features)
            ])

        # <batch_size, max_number_of_phrases, embd_dim>,
        integrated_captions_phrases_embds, captions_phrases_mask = self.language_module(
            batch_captions, batch_captions_phrases)

        return trandformed_batch_images, trandformed_batch_boxes, batch_images_features, integrated_captions_phrases_embds, captions_phrases_mask

    def build_physical_batch_graphs(self,
                                    images_boxes,
                                    graphs_node_neighbors_circle_level=2):
        """[Build the physical graphs from the image boxes]

        Args:
            images_boxes (List(torch.Tensor)) : [each item is a tensor containing the anchors of the corresponding image]
            graphs_node_neighbors_circle_level (int): [the neighbors are existed in this circle level]
        """
        for image_i in range(len(images_boxes)):
            image_boxes = images_boxes[image_i]
            self.physical_batch_graph_operator.define_graph(graph_id=image_i)
            image_graph = self.physical_batch_graph_operator.obtain_graph(
                graph_id=image_i)

            image_graph.assign_nodes(boxes=image_boxes.numpy())
            image_graph.generate_edges(
                circle_level=graphs_node_neighbors_circle_level)
            image_graph.construct_graph()

            self.physical_batch_graph_operator.set_graph(image_graph, image_i)
        #print(self.physical_batch_graph_operator.get_batch_nodes_boxes())
        self.physical_batch_graph_operator.construct_batch_graph()

        return self.physical_batch_graph_operator.get_graphs()

    def initialize_anchors(self, trandformed_batch_images,
                           batch_images_features):
        """[Initialize the anchors for images]
            Note: the generated boxes are derived based on the order: row-wise
        Args:
            trandformed_batch_images (ImageList): the batch of image after the transformation, 
                                                    ImageList: can be visited by ImageList.tensor, ImageList.image_sizes
            batch_images_features (OrderDict(str: torch.Tensor)): [featmap_name: feature map tensor]

        Returns:
            lefted_images_anchors (List(torch.Tensor)) : [each item is a tensor containing the anchors of the corresponding image]
        """
        batch_images_features_tensor = list(batch_images_features.values())
        initialization_images_anchors = self.an_generator(
            trandformed_batch_images, batch_images_features_tensor)

        strides = [(int(image_anchors[1, 0] - image_anchors[0, 0]), int(image_anchors[1, 0] - image_anchors[0, 0])) \
                        for image_anchors in initialization_images_anchors]

        filtered_init_boxes, maintained_anchors_id = self.anchor_operator.filter_anchors(
            trandformed_batch_images.image_sizes,
            initialization_images_anchors)
        anchors_h_w = self.anchor_operator.obtain_h_w_from_anchors_id(
            maintained_anchors_id)

        lefted_images_anchors = self.anchor_operator.remove_anchors_with_intervals(
            trandformed_batch_images,
            anchors_h_w,
            filtered_init_boxes,
            images_anchors_interval=self.anchor_operator.anchor_intervals,
            strides=strides)

        return lefted_images_anchors, filtered_init_boxes

    def forward(self, trandformed_batch_images, trandformed_batch_boxes,
                batch_images_features, integrated_captions_phrases_embds,
                captions_phrases_mask, built_graph):
        hs = self.original_image_size[0][0]
        ws = self.original_image_size[0][1]

        rois_feas = self.an_roi_aligner(batch_images_features,
                                        [trandformed_batch_boxes[0]],
                                        self.original_image_size)
        built_graph.assign_node_feas(rois_feas)

        out_nodes_features, edge_index = self.mm_graph([
            built_graph.nodes_feas, trandformed_batch_boxes,
            built_graph.batch_edges_index, hs, ss
        ], integrated_captions_phrases_embds)

        trans_coefficients = self.transformation_head(out_nodes_features)

        normed_transformed_boxes = self.transformation_head.operate_transformation(
            trandformed_batch_boxes, trans_coefficients, hs, ws)

        return normed_transformed_boxes, trans_coefficients

    def convert_boxes_to_original_image(self, transformed_batch_images,
                                        images_boxes):
        """[Map the boxes in the resized/transformed images to the original image]

        Args:
            transformed_batch_images (ImageList): the batch of image after the transformation, ImageList: can be visited by ImageList.tensor
            images_boxes (List[torch.Tensor]): [description]
                    It need to be converted to the following type: For example:
                                [
                                    {
                                        "boxes": boxes[i],
                                        "labels": labels[i],
                                        "scores": scores[i],
                                    }
                                ]
        """
        original_image_sizes = self.original_image_size
        # convert the boxes from list(torch.Tensor) to the List[Dict[str, Tensor]]
        required_boxes = [{"boxes": boxes} for boxes in images_boxes]
        self.transform.training = False
        post_processed_boxes = self.transform.postprocess(
            required_boxes, transformed_batch_images.image_sizes,
            original_image_sizes)
        return post_processed_boxes