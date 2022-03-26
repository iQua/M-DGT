#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Inherent Python '''
import os
''' Third Libs '''
import torch
import numpy as np
from absl import app
from absl import flags
''' Local Libs '''
from datasets import referitgame_base
from datasets import ReferItGame_provider

from preprocess import image_detection_processer
from models.language import LanguageModel
from models.anchors_op import AnchorsOperator
from models.base_graph_op import BatchGraphOperator
from models.node_transformer import CrossModalNodeEncoderLayer, CrossModalContextMultiHeadedAttention, TwoDTransformationHead
from models.multi_modal_graph import CMGATLayerImp
from models.MDGT_framework import MDGTFramework

from learning import evaluation

from common_flags import CURRENT_PROJECT_DIR

from visualization import grounding_visualizer
from visualization import graph_visualizer

FLAGS = flags.FLAGS

flags.DEFINE_string('visualizations_dir',
                    os.path.join(CURRENT_PROJECT_DIR, "visualization",
                                 "Refcoco_demo"),
                    help="the visualization dir")

flags.DEFINE_string('backbone_visual_net_name',
                    "resnset18",
                    help="the backbone network used for the image")
flags.DEFINE_string('backbone_text_net_name',
                    "bert",
                    help="the backbone network used for the text")

flags.DEFINE_float("min_size", 800,
                   "the minimum size of the image in the transformer")
flags.DEFINE_float("max_size", 1333,
                   "the maximum size of the image in the transformer")
flags.DEFINE_list("image_mean", [0.485, 0.456, 0.406],
                  "the mean size of the image in the transformer")
flags.DEFINE_list("image_std", [0.229, 0.224, 0.225],
                  "the std size of the image in the transformer")

flags.DEFINE_list("anchor_sizes", [(256, )],
                  "the base anchor size used to generate boxes")
flags.DEFINE_list("anchor_intervals", [(256, 256)],
                  "the distance between the generated anchors")
flags.DEFINE_list("aspect_ratios", [(1.0)],
                  "the ratio of h/w used to generate different boxes")

flags.DEFINE_integer(
    "graphs_node_neighbors_circle_level", 2,
    "the edges of the neighbot circle in the initialization graph")

flags.DEFINE_list('featmap_names', ["avg_pool"],
                  help="the names of the feature maps that obtain the ROIs")
flags.DEFINE_list("roi_output_size", [5], "output size of the roi alignment")
flags.DEFINE_float("roi_sampling_ratio", 2, "the sampling ratio")

flags.DEFINE_string(
    "eval_log_dir", os.path.join(CURRENT_PROJECT_DIR, "experiments",
                                 "Refcoco"),
    "the number of batches for one epoch")

flags.DEFINE_string(
    'visualization_dir',
    os.path.join(CURRENT_PROJECT_DIR, "visualization", "Refcoco"),
    "The path of the visualization")

flags.DEFINE_float("eps", 0.0, "epsilon of label smoothing")


def unpack_batch_data(batches_data):
    images_name = [bt_data[0] for bt_data in batches_data]
    original_images = [bt_data[1] for bt_data in batches_data]  # uint8 array
    processed_images = [bt_data[2]
                        for bt_data in batches_data]  # uint8, tensor
    images_caption = [bt_data[3].caption for bt_data in batches_data]
    images_caption_phrases = [
        bt_data[3].caption_phrases for bt_data in batches_data
    ]
    images_caption_phrase_bboxs = [
        bt_data[3].caption_phrase_bboxs for bt_data in batches_data
    ]
    images_caption_phrases_cate = [
        bt_data[3].caption_phrases_cate for bt_data in batches_data
    ]
    images_caption_phrases_cate_id = [
        bt_data[3].caption_phrases_cate_id for bt_data in batches_data
    ]

    return images_name, original_images, processed_images, images_caption_phrases, \
            images_caption_phrase_bboxs, images_caption, images_caption_phrases_cate, images_caption_phrases_cate_id


def _main(argv):
    referitgame_bs = referitgame_base.REFERITGMBase(
        dataset_dir=FLAGS.ReferItGame_source_path,
        source_images_dir=os.path.join(FLAGS.COCO_source_images_path,
                                       "train2017"),
        data_name="refcoco+",  # refcoco, refcoco+ and refcocog
        split_type="unc")  # google or unc
        
    img_dec_processor = image_detection_processer.ImageDetectionProcessor(
        dataset_name='refcoco')
    image_dec_transform_func = img_dec_processor.create_image_dec_processor(
        resize_shape=[None, None],
        target_shape=[None, None],
        phase="val",
        randomization=False,
        normalization=True)

    refig_provider = ReferItGame_provider.ReferItGameProvider(
        base_data=referitgame_bs,
        batch_size=1,
        epoch_size=1,
        num_data_loading_workers=FLAGS.num_data_loading_workers,
        transform_image_dec_func=image_dec_transform_func,
        transform_text_func=None,
        phase="val")

    lang_module = LanguageModel(
        language_model_name=FLAGS.backbone_text_net_name)
    anchors_operator = AnchorsOperator()
    batch_phy_graph_op = BatchGraphOperator()

    cross_modal_node_encode_layer = CrossModalNodeEncoderLayer(
        size=512, node_feed_forward=None, dropout=0.2)
    cross_modal_attn_model = CrossModalContextMultiHeadedAttention(com_dim=512,
                                                                   dropout=0.2)
    mm_graph = CMGATLayerImp(
        cross_modal_node_encode_layer=cross_modal_node_encode_layer,
        cross_modal_attn_model=cross_modal_attn_model,
        num_in_node_features=512,
        num_in_phrase_features=512,
        num_out_node_features=1024,
        num_out_phrase_features=512,
        num_of_heads=1)  # the number of heads do not impact the performance
    transformation_head = TwoDTransformationHead(input_dims=1024,
                                                 coefficients_dim=4)
    dgf_model = MDGTFramework(
        backbone_visual_net_name=FLAGS.backbone_visual_net_name,
        language_module=lang_module,
        anchor_operator=anchors_operator,
        physical_batch_graph_operator=batch_phy_graph_op,
        mm_graph=mm_graph,
        transformation_head=transformation_head,
        flags=FLAGS)

    # 6. visualization
    gds_visualizer = grounding_visualizer.GroundingVisualizer(
        visualization_dir=FLAGS.visualizations_dir, is_unique_created=True)
    gph_visualizer = graph_visualizer.GraphVisualizer(
        visualization_dir=FLAGS.visualizations_dir, is_unique_created=False)

    # 5. build trainer
    model_evaler = evaluation.ModelStrategyEvaluator(
        eval_FLAGS=FLAGS,
        num_iters=9,
        visualizer=gds_visualizer,
        graph_visualizer=gph_visualizer)

    dgf_model = model_evaler.load_model(defined_model=dgf_model,
                                        model_path="cvpr22_refcoco_20.ph")
    model_evaler.eval_onece(epoch_num=1,
                            main_model=dgf_model,
                            data_loader=refig_provider,
                            unpack_batch_data=unpack_batch_data)


if __name__ == "__main__":
    app.run(_main)
