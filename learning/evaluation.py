import os
import time
from random import randint
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import torch

import torch.nn.functional as F
import torchvision
from models.boxes_utils import bbox_overlaps_diou

from learning.utils import numbering_directory, integrate_phrase_boxes, noise_add, noise_add_shift


def build_iters_graphs(iters_trans_results, main_model):

    built_graphs = OrderedDict()
    main_model.physical_batch_graph_operator.reset()

    boxes_id = None
    total_remove_box_idex = None
    total_remove_edges = None

    for iter_i in list(iters_trans_results.keys()):
        # print("\n")
        # print("iter_i: ", iter_i)
        if not isinstance(iter_i, int):
            continue
        iter_trans_results = iters_trans_results[iter_i]
        generated_boxes = iter_trans_results["generated_boxes"]
        boxes_appraoch_phrases_label = iter_trans_results[
            "boxes_appraoch_phrases_label"]
        boxes_appraoch_phrases_iou = iter_trans_results[
            "boxes_appraoch_phrases_iou"]

        #print("boxes_appraoch_phrases_label: ", boxes_appraoch_phrases_label)

        main_model.physical_batch_graph_operator.define_graph(graph_id=iter_i)

        image_graph = main_model.physical_batch_graph_operator.obtain_graph(
            graph_id=iter_i)

        if iter_i == 0:
            num_of_boxes = len(generated_boxes)
            boxes_id = np.arange(start=0, stop=num_of_boxes)
            image_graph.assign_nodes(
                boxes=generated_boxes,
                boxes_label=boxes_appraoch_phrases_label[:, 0],
                boxes_ids=boxes_id,
                boxes_score=boxes_appraoch_phrases_iou)
            image_graph.generate_edges(num_edges_pre_node=5,
                                       is_aligned_test=True)
        else:
            # print(main_model.physical_batch_graph_operator.batch_graphs[0].get_edges().keys())
            # print(main_model.physical_batch_graph_operator.batch_graphs[0].nodes.get_nodes_id())
            # print("the graph id: ", main_model.physical_batch_graph_operator.batch_graphs[0].graph_id)
            # print("image_graph id: ", image_graph.graph_id)
            initialize_graph = main_model.physical_batch_graph_operator.obtain_graph(
                graph_id=0)

            image_graph.inherit_edges(edges=initialize_graph.get_edges())

            remove_box_idex = iters_trans_results[iter_i -
                                                  1]["remove_box_idex"]
            #print("remove_box_idex: ", remove_box_idex)
            if total_remove_box_idex is None:
                total_remove_box_idex = remove_box_idex
            else:
                total_remove_box_idex = np.concatenate(
                    [total_remove_box_idex, remove_box_idex], axis=None)

            #print("total_remove_box_idex: ", total_remove_box_idex)
            image_graph.assign_nodes(
                boxes=generated_boxes,
                boxes_label=boxes_appraoch_phrases_label[:, 0],
                boxes_ids=boxes_id,
                boxes_score=boxes_appraoch_phrases_iou)

            image_graph.remove_nodes(remove_nodes_id=total_remove_box_idex)

            rm_edges = image_graph.generate_remove_edges()
            if total_remove_edges is None:
                if rm_edges is None:
                    pass
                else:
                    total_remove_edges = rm_edges

            else:
                if rm_edges is not None:
                    total_remove_edges = np.concatenate(
                        [total_remove_edges, rm_edges], axis=0)

            if total_remove_edges is not None:
                image_graph.remove_edges(total_remove_edges)

        image_graph.construct_graph()

        main_model.physical_batch_graph_operator.set_graph(image_graph, iter_i)

    built_graphs = main_model.physical_batch_graph_operator.get_graphs()

    return built_graphs


class ModelStrategyEvaluator(object):
    def __init__(self,
                 eval_FLAGS,
                 num_iters=9,
                 visualizer=None,
                 graph_visualizer=None):
        self._eval_FLAGS = eval_FLAGS  # log dir is where to load and save the model
        self._eval_log_dir = eval_FLAGS.eval_log_dir

        self.visualizer = visualizer
        self.graph_visualizer = graph_visualizer

        if not os.path.exists(self._eval_log_dir):
            os.makedirs(self._eval_log_dir, exist_ok=True)
        else:
            self._eval_log_dir = numbering_directory(self._eval_log_dir)
            os.makedirs(self._eval_log_dir)

        self._log_file_path = os.path.join(self._eval_log_dir, "eval_log.txt")

        self.batch_size = 1
        self.graph_edges_n = self._eval_FLAGS.graphs_node_neighbors_circle_level
        self.num_iters = num_iters
        self.eps = self._eval_FLAGS.eps

        self.iters_produces = OrderedDict()

    def load_model(self, defined_model, model_path):
        defined_model.load_state_dict(torch.load(model_path))
        return defined_model

    def eval_onece(self, epoch_num, main_model, data_loader,
                   unpack_batch_data):

        for step_i, batch in enumerate(tqdm(data_loader(epoch_num)), 1):
            # print(batch)
            images_name, original_images, \
                processed_images, images_phrases, images_phrases_boxes, \
                    images_caption, images_caption_phrases_cate, images_caption_phrases_cate_id = unpack_batch_data(batch)

            integrated_images_boxes = integrate_phrase_boxes(
                images_phrases_boxes)

            trandformed_batch_images, trandformed_batch_boxes, \
                batch_images_features, integrated_captions_phrases_embds, \
                    captions_phrases_mask  = main_model.backbone_forward(batch_images=processed_images,
                                                                batch_images_boxes=integrated_images_boxes,
                                                                batch_captions=images_caption,
                                                                batch_captions_phrases=images_phrases)
            target_trandformed_batch_boxes = trandformed_batch_boxes

            gen_anchors, filtered_init_gen_anchors = main_model.initialize_anchors(
                trandformed_batch_images, batch_images_features)
            normed_transformed_boxes = None

            for iter_i in range(self.num_iters):

                if iter_i == 0:
                    generated_boxes = gen_anchors
                else:
                    generated_boxes = normed_transformed_boxes

                built_phy_graphs = main_model.build_physical_batch_graphs(
                    images_boxes=generated_boxes,
                    graphs_node_neighbors_circle_level=self.graph_edges_n)
                normed_transformed_boxes, trans_coefficients = main_model(
                    trandformed_batch_images, generated_boxes,
                    batch_images_features, integrated_captions_phrases_embds,
                    captions_phrases_mask, built_phy_graphs)

                pred_rois_feas = main_model.an_roi_aligner(
                    normed_transformed_boxes)
                gh_rois_feas = main_model.an_roi_aligner(
                    trandformed_batch_boxes)

                gh_coeffs = main_mode.compute_transformation_coefficients(
                    normed_transformed_boxes, target_trandformed_batch_boxes)

                diou_scores = bbox_overlaps_diou(
                    normed_transformed_boxes, target_trandformed_batch_boxes)

                decided_boxes = torchvision.ops.nms(normed_transformed_boxes,
                                                    diou_scores,
                                                    iou_threshold=0.6)

                self.iters_produces[iter_i] = OrderedDict()
                self.iters_produces[iter_i][
                    "generated_boxes"] = generated_boxes  # array with shape <num_of_boxes, 4>
                self.iters_produces[iter_i][
                    "decided_boxes"] = decided_boxes  # array with shape <num_of_boxes, 4>
                self.iters_produces[iter_i][
                    "boxes_appraoch_phrases_iou"] = diou_scores

            # visual the boxes at the end of each epoch
            pp_gen_anchors = main_model.convert_boxes_to_original_image(
                transformed_batch_images=self.iters_produces[-1],
                images_boxes=gen_anchors)

            iters_trans_results = self.iters_produces

            built_graphs = build_iters_graphs(iters_trans_results, main_model)

            self.graph_visualizer.set_base_save_dir(
                self.visualizer.save_visual_dir)
            self.visualizer.set_save_dir(epoch_number=epoch_num, step_i=step_i)
            self.graph_visualizer.set_save_dir(epoch_number=epoch_num,
                                               step_i=step_i)

            self.visualizer.log_samples(images_name=images_name,
                                        original_images=original_images,
                                        processed_images=processed_images,
                                        images_phrases=images_phrases,
                                        images_caption=images_caption)
            self.visualizer.log_images_boxes_with_label(
                images_name=images_name,
                processed_images=processed_images,
                images_phrases=images_phrases,
                images_phrases_boxes=images_phrases_boxes,
                log_name="groundtruth_boxes")

            images_boxes = [
                img_boxes["boxes"].tolist() for img_boxes in pp_gen_anchors
            ]
            self.visualizer.log_images_boxes(images_name=images_name,
                                             processed_images=processed_images,
                                             images_boxes=images_boxes,
                                             log_name="images_anchors")

            self.visualizer.log_images_labled_itered_boxes(
                images_name=images_name,
                processed_images=processed_images,
                iters_trans_results=iters_trans_results,
                log_name="iteration_boxes")

            self.visualizer.log_final_boxes(
                images_name=images_name,
                processed_images=processed_images,
                images_phrases=images_phrases,
                images_phrases_boxes=images_phrases_boxes,
                iters_trans_results=iters_trans_results,
                log_name="final_boxes")

            self.graph_visualizer.graphs_visualization(
                built_phy_graphs, pre_save_name="0", log_name="sample_graphs")
            self.graph_visualizer.graphs_visualization(built_graphs,
                                                       pre_save_name="_iters_",
                                                       log_name="iters_graphs")

            self.visualizer.reset_save_dir()
            self.graph_visualizer.set_base_save_dir(
                self.visualizer.save_visual_dir)
