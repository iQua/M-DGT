#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' python inherent libs '''
import os
from collections import OrderedDict
''' third parts libs '''
import numpy as np
import scipy as scip
import networkx as nx
from scipy.linalg import block_diag
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.spatial import distance
''' Local libs '''


class Nodes(object):
    def __init__(self):
        # the order of nodes id, nodes_spatial_box, nodes_spatial_center should be consistent with each other
        self.nodes_info = OrderedDict()

    def add_node(self, spatial_box, node_id, node_label, node_score):
        """[add one node to the nodes]

        Args:
            spatial_box ([list or 1d array]): [the coordinate of the box]
            node_id ([int]): [the identity of the box]
        """
        box_w = spatial_box[2] - spatial_box[0]
        box_h = spatial_box[3] - spatial_box[1]
        center_x = spatial_box[0] + int(box_w / 2)
        center_y = spatial_box[1] + int(box_h / 2)

        if node_id not in self.nodes_info.keys():
            self.nodes_info[node_id] = OrderedDict()

        self.nodes_info[node_id]["spatial_box"] = spatial_box
        self.nodes_info[node_id]["spatial_center"] = (center_x, center_y)
        self.nodes_info[node_id]["node_id"] = node_id
        self.nodes_info[node_id]["node_label"] = node_label
        self.nodes_info[node_id]["node_score"] = node_score

    def get_nodes_info(self):
        nodes_spatial_box = list()
        nodes_id = list()
        nodes_label = list()
        nodes_center = list()
        nodes_scores = list()
        for node_id in self.nodes_info.keys():
            nodes_spatial_box.append(self.nodes_info[node_id]["spatial_box"])
            nodes_id.append(self.nodes_info[node_id]["node_id"])
            nodes_label.append(self.nodes_info[node_id]["node_label"])
            nodes_center.append(self.nodes_info[node_id]["spatial_center"])
            nodes_scores.append(self.nodes_info[node_id]["node_score"])

        return nodes_spatial_box, nodes_id, nodes_label, nodes_center, nodes_scores

    def compute_nodes_distance_matrix(self):
        nodes_spatial_box, nodes_id, _, _, _ = self.get_nodes_info()
        nodes_dis_mt = distance_matrix(nodes_spatial_box, nodes_spatial_box)

        return nodes_dis_mt, nodes_id

    def get_nodes_id(self):
        return list(self.nodes_info.keys())

    def get_nodes(self):
        return self.nodes_info

    def get_number_of_nodes(self):
        return len(list(self.nodes_info.keys()))

    def get_nodes_boxes(self):
        boxes = [
            self.nodes_info[node_id]["spatial_box"]
            for node_id in self.nodes_info.keys()
        ]
        boxes = np.array(boxes)
        boxes = boxes.reshape((-1, 4))
        return boxes

    def get_nodes_label(self):
        nodes_label = [
            self.nodes_info[node_id]["node_label"]
            for node_id in self.nodes_info.keys()
        ]
        return nodes_label

    def get_nodes_scores(self):
        nodes_scores = [
            self.nodes_info[node_id]["node_score"]
            for node_id in self.nodes_info.keys()
        ]
        return nodes_scores

    def get_nodes_center(self):
        nodes_center = [
            self.nodes_info[node_id]["spatial_center"]
            for node_id in self.nodes_info.keys()
        ]

        return nodes_center

    def set_nodes_info(self, new_nodes_info):
        self.nodes_info = new_nodes_info


class PhysicalGraph(object):
    def __init__(self, graph_id):
        self.spatial_graph = nx.Graph()
        self.spatial_graph.position = None
        self.spatial_graph.population = None

        self.nodes = Nodes()
        self.adjacency_matrix = list()
        self.edges_matrix = list()
        self.nodes_edges_mapper = OrderedDict(
        )  # map the id of boxes to the nodes id that are edges of this node
        self.nodes_index_id_mapper = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix
        self.nodes_id_index_mapper = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix

        self.graph_id = graph_id

    def assign_nodes(self,
                     boxes,
                     boxes_ids=None,
                     boxes_label=None,
                     boxes_score=None):
        # assign the boxes (np.array): [a array containing all boxes for the image] to this graph
        boxes = boxes
        num_of_boxes = boxes.shape[0]
        if boxes_ids is not None:
            boxes_ids = boxes_ids
        else:
            boxes_ids = np.arange(start=0, stop=num_of_boxes)

        if boxes_label is not None:
            boxes_label = boxes_label
        else:
            boxes_label = np.zeros(
                num_of_boxes)  # all boxes have the same label

        if boxes_score is not None:
            boxes_score = boxes_score
        else:
            boxes_score = np.ones(num_of_boxes) * 0.1  # all boxes set to 0.1

        for box_i in range(num_of_boxes):
            box_coor = boxes[box_i]
            box_id = boxes_ids[box_i]
            box_label = boxes_label[box_i]
            box_score = boxes_score[box_i]

            self.nodes.add_node(box_coor, box_id, box_label, box_score)

    def is_aligned_edges(self, node_id, edge_node_id):
        all_node_infos = self.nodes.get_nodes()

        node_center = all_node_infos[node_id]["spatial_center"]
        edge_node_center = all_node_infos[edge_node_id]["spatial_center"]

        if node_center[0] == edge_node_center[0] or node_center[
                1] == edge_node_center[1]:
            return True

        return False

    def inherit_edges(self, edges):
        # the edges is an OrderDict [node_id: node_edges[np.array])]
        self.nodes_edges_mapper = edges.copy()
        #print("inherit_edges: ", self.nodes_edges_mapper.keys())
        self.create_the_nodes_id_converter()

    def generate_edges(self, circle_level=1, is_aligned_test=True):
        """[generate edges between node based on the spatial connection]

        Args:
            circle_level (int): the neighbors of each node lies in the defined circle_level
                                the default is to include itself
        """
        # number_of_nodes = len(self.nodes.get_nodes_id())

        # get the sort index of boxes, [number_of_boxes, number_of_boxes]
        nodes_dis_mt, nodes_id = self.nodes.compute_nodes_distance_matrix()

        for i in range(len(nodes_dis_mt)):
            node_id = nodes_id[i]
            node_idxi_nodes_dis = nodes_dis_mt[i]

            # as this is for the initialization of the nodes
            total_levels_dis = np.unique(node_idxi_nodes_dis)
            total_levels_dis.sort()  # from nearest to the largest

            assert circle_level < len(total_levels_dis)

            most_outer_level_dis = total_levels_dis[circle_level]
            boxes_idx_inner_circle = [
                j for j in range(len(node_idxi_nodes_dis))
                if node_idxi_nodes_dis[j] <= most_outer_level_dis
            ]

            circle_included_nodes_id = [
                nodes_id[c_b_idx] for c_b_idx in boxes_idx_inner_circle
            ]

            self.nodes_edges_mapper[node_id] = np.array(
                circle_included_nodes_id)

        self.create_the_nodes_id_converter()

    def build_adjacency_matrix(self):
        self.create_the_nodes_id_converter()
        number_of_nodes = self.nodes.get_number_of_nodes()
        self.adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
        self.edges_matrix = list()
        for node_id in list(self.nodes_edges_mapper.keys()):
            node_id_index = self.nodes_id_index_mapper[node_id]
            node_edges = self.nodes_edges_mapper[node_id]
            for edge_id in node_edges:
                edge_node_index = self.nodes_id_index_mapper[edge_id]
                self.adjacency_matrix[node_id_index, edge_node_index] = 1
                self.edges_matrix.append([node_id_index, edge_node_index])

        self.edges_matrix = np.array(self.edges_matrix).reshape((2, -1))

        return self.adjacency_matrix, self.edges_matrix

    def construct_graph(self):
        nodes_spatial_box, nodes_id, nodes_label, nodes_center, nodes_scores = self.nodes.get_nodes_info(
        )
        # print("construct nodes_id: ", nodes_id)
        # print("self.nodes_edges_mapper: ", self.nodes_edges_mapper.keys())
        # # create required node type # G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])

        # Note that the node_id here will be the drawed lable in the graph
        graph_nodes = list()
        for i in range(len(nodes_id)):
            node_id = nodes_id[i]
            node_box = nodes_spatial_box[i]
            node_label = nodes_label[i]
            node_center = nodes_center[i]
            node_score = nodes_scores[i] * 500
            node_info = {
                "box": node_box,
                "label": node_label,
                "pos": node_center,
                "score": node_score
            }
            graph_nodes.append((node_id, node_info))

        #print("graph_nodes, ", graph_nodes)

        #self.spatial_graph.position = OrderedDict(zip(nodes_id, nodes_center))
        #print("construct self.nodes_edges_mapper: ", self.nodes_edges_mapper)
        graph_edges = list()
        for node_id in nodes_id:
            if node_id in list(self.nodes_edges_mapper.keys()):
                for node_connect_id in self.nodes_edges_mapper[node_id]:
                    graph_edges.append((node_id, node_connect_id))

        required_graph_nodes = list()
        for graph_node in graph_nodes:
            node_id = graph_node[0]

            is_bad_node = True
            for edge_node_id in list(self.nodes_edges_mapper.keys()):
                if node_id == edge_node_id:
                    is_bad_node = False
                if node_id in self.nodes_edges_mapper[edge_node_id]:
                    is_bad_node = False

            if not is_bad_node:
                required_graph_nodes.append(graph_node)

        #print("graph_edges: ", graph_edges)
        self.spatial_graph.add_nodes_from(required_graph_nodes)
        self.spatial_graph.add_edges_from(graph_edges)

        # nodes_size = [50] * len(nodes_id)
        # self.spatial_graph.population = OrderedDict(zip(nodes_id, nodes_size))

    def remove_nodes(self, remove_nodes_id):
        # remove the corresponding box
        nodes_info = self.nodes.get_nodes()
        # print(remove_nodes_id)
        # print(nodes_info)
        nodes_box = self.nodes.get_nodes_boxes()
        nodes_id = self.nodes.get_nodes_id()
        for rm_node_id in remove_nodes_id:
            # remove this key
            if rm_node_id in nodes_info.keys():
                del nodes_info[rm_node_id]

            edge_info_node = list(self.nodes_edges_mapper.keys())
            if rm_node_id in edge_info_node:
                del self.nodes_edges_mapper[rm_node_id]
            # remove the edge for this node node edge from other nodes
            for cur_node_id in list(self.nodes_edges_mapper.keys()):
                cur_node_edges = self.nodes_edges_mapper[cur_node_id]
                if rm_node_id in cur_node_edges:
                    new_cur_node_edges = [
                        nd_id for nd_id in cur_node_edges
                        if nd_id != rm_node_id
                    ]
                    self.nodes_edges_mapper[cur_node_id] = np.array(
                        new_cur_node_edges)

        self.nodes.set_nodes_info(nodes_info)
        #print(self.nodes_edges_mapper)

    def generate_remove_edges(self):
        # remove the bad edges
        nodes_info = self.nodes.get_nodes()
        nodes_spatial_box, nodes_id, nodes_label, nodes_center, nodes_scores = self.nodes.get_nodes_info(
        )
        nodes_edges_mapper = self.nodes_edges_mapper

        total_remove_edges = list()
        uni_labels = np.unique(nodes_label)
        for lb in uni_labels:
            lb_edge_whole_lengths = list()
            all_edges_length_record = list()
            all_edges_record = list()

            for node_id in nodes_id:
                if nodes_info[node_id]["node_label"] == lb:
                    node_edges = nodes_edges_mapper[node_id]
                    for edge_node_id in node_edges:
                        edge_center = nodes_info[edge_node_id][
                            "spatial_center"]
                        node_center = nodes_info[node_id]["spatial_center"]
                        dis_edge = distance.euclidean(edge_center, node_center)

                        if nodes_info[edge_node_id]["node_label"] == lb:
                            lb_edge_whole_lengths.append(dis_edge)

                        all_edges_record.append((node_id, edge_node_id))
                        all_edges_length_record.append(dis_edge)

            lb_edge_whole_lengths_arr = np.array(lb_edge_whole_lengths)
            all_edges_length_record_arr = np.array(all_edges_length_record)
            all_edges_record_arr = np.array(all_edges_record)

            lb_avg_dis = np.average(lb_edge_whole_lengths_arr)

            is_separete = all_edges_length_record_arr > 3 * lb_avg_dis
            separete_edge = all_edges_record_arr[is_separete]

            total_remove_edges.append(separete_edge.reshape((-1, 2)))

        is_flag = False
        #print("total_remove_edges: ", total_remove_edges)
        for edge in total_remove_edges:
            if edge.size != 0:
                is_flag = True
        if is_flag:
            total_remove_edges = np.concatenate(total_remove_edges, axis=0)
        else:
            total_remove_edges = None
        return total_remove_edges

    def remove_edges(self, total_remove_edges):

        # print("total_remove_edges: ", total_remove_edges)
        # print("self.nodes_edges_mapper: ", self.nodes_edges_mapper)
        for edge_i in total_remove_edges:
            edge_start = edge_i[0]
            edge_end = edge_i[1]
            if edge_start in list(self.nodes_edges_mapper.keys()):
                remained_edges = [
                    edge_node_id
                    for edge_node_id in self.nodes_edges_mapper[edge_start]
                    if edge_node_id != edge_end
                ]
                if len(remained_edges) == 0:
                    del self.nodes_edges_mapper[edge_start]

                else:
                    self.nodes_edges_mapper[edge_start] = np.array(
                        remained_edges)

            if edge_end in list(self.nodes_edges_mapper.keys()):
                remained_edges = [
                    edge_node_id
                    for edge_node_id in self.nodes_edges_mapper[edge_end]
                    if edge_node_id != edge_start
                ]
                if len(remained_edges) == 0:
                    del self.nodes_edges_mapper[edge_end]

                else:
                    self.nodes_edges_mapper[edge_end] = np.array(
                        remained_edges)
        # remove separate nodes
        nodes_info = self.nodes.get_nodes()

        for node_id in list(nodes_info.keys()):
            is_fremove_lag = True
            for edge_i in list(self.nodes_edges_mapper.keys()):
                if node_id == edge_i:
                    is_fremove_lag = False
                if node_id in self.nodes_edges_mapper[edge_i]:
                    is_fremove_lag = False

            if is_fremove_lag:
                del nodes_info[node_id]

        self.nodes.set_nodes_info(nodes_info)
        #print("self.nodes_edges_mapper: ", self.nodes_edges_mapper)

    def create_the_nodes_id_converter(self):
        """[Convert the nodes id to the standard index that ranges from 0]


            Why this function is required?
                - because of the dynamic graph, some nodes and edges will be removed from the graph. 
                    Thus, the node ids can be 1, 5, 7, 9, 10, which is not consistent with the node feature matrix A = R^{5xd}
                    Then, in the node indexing, such as A[7], A[10], can occur errors. A[7] should be the A[2]. 
            Therefore, we need to convert the 1, 5, 7, 9, 10 to 0, 1, 2, 3, 4. 
        Returns:
            [type]: [description]
        """
        nodes_info = self.nodes.get_nodes(
        )  # get the nodes that are stored as the orderdict
        nodes_id = list(
            nodes_info.keys()
        )  # the node id should be in order as they are stored in the OrderDict
        nodes_id_index = list(range(len(nodes_id)))

        self.nodes_index_id_mapper = OrderedDict(zip(nodes_id_index, nodes_id))
        self.nodes_id_index_mapper = OrderedDict(zip(nodes_id, nodes_id_index))

    def get_edges(self):
        return self.nodes_edges_mapper


# in this class, all boxes and ids of nodes will be stored in a big matrix. such as :
#   [A_1          ]
#   [   A_2       ]
#   [       A_2   ]
#   [           A_3], in which A_i is the adjacency matrice of the i-th graph
# the nodes' boxes will be store as a a list [box_coor, ....,] whose length is the
# the nodes'id will be strore as a
class BatchGraphOperator(object):
    def __init__(self):
        self.batch_graphs = OrderedDict()
        self.reset()
        self.nodes_feas = None

    def assign_node_feas(self, nodes_feas):
        self.nodes_feas = nodes_feas

    def define_graph(self, graph_id):
        phy_graph = PhysicalGraph(graph_id)

        self.batch_graphs[graph_id] = phy_graph

    def set_graph(self, graph, graph_id):
        self.batch_graphs[graph_id] = graph

    def obtain_graph(self, graph_id):
        return self.batch_graphs[graph_id]

    def obtain_all_graphs_list(self):
        return [
            self.batch_graphs[graph_id]
            for graph_id in list(self.batch_graphs.keys())
        ]

    def get_graphs(self):
        return self.batch_graphs

    def get_batch_nodes_boxes(self):
        batch_boxes = [
            graph.nodes.get_nodes_boxes()
            for graph in self.obtain_all_graphs_list()
        ]

        return batch_boxes

    def reset(self):
        self.batch_graphs = OrderedDict()

        # this is a big graph which holds all graphs in one batch
        self.batch_nodes = Nodes()
        self.batch_adjacency_matrix = list()
        self.batch_edges_index = list()
        self.batch_nodes_edges_mapper = OrderedDict(
        )  # map the id of boxes to the nodes id that are edges of this node
        self.batch_nodes_index_id_mapperer = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix
        self.batch_nodes_id_index_mapperer = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix

    def construct_batch_graph(self):
        """[Construct a batch of graphs]

       
        batch_graphs ([OrderedDict]): [the built graphs stored in the dict such as graph_id: PhysicalGraph],
        """
        self.batch_edges_index = list()
        self.batch_nodes_edges_mapper = OrderedDict(
        )  # map the id of boxes to the nodes id that are edges of this node
        self.batch_nodes_index_id_mapper = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix
        self.batch_nodes_id_index_mapper = OrderedDict(
        )  # convert the node id to the index corresponding to the node feature matrix

        total_number_of_nodes = 0
        self.batch_adjacency_matrix = list()  # clean the matrix
        self.batch_edges_index = list()

        for built_graph_id in list(self.batch_graphs.keys()):
            built_graph = self.batch_graphs[built_graph_id]
            gh_nodes_boxes = built_graph.nodes.get_nodes_boxes()
            gh_nodes_scores = built_graph.nodes.get_nodes_scores()
            number_of_boxes = len(gh_nodes_boxes)

            # add the corresponding nodes
            for i in range(number_of_boxes):
                gh_box = gh_nodes_boxes[i]
                gh_box_score = gh_nodes_scores[i]
                self.batch_nodes.add_node(spatial_box=gh_box,
                                          node_id=i + total_number_of_nodes,
                                          node_label=built_graph_id,
                                          node_score=gh_box_score)

            built_graph_nodes_edges_mapper = built_graph.nodes_edges_mapper
            built_graph_nodes_index_id_mapper = built_graph.nodes_index_id_mapper
            built_graph_nodes_id_index_mapper = built_graph.nodes_id_index_mapper

            def insert_graph_tp_batch_graph(graph_mapper, batch_mapper):
                for k, v in graph_mapper.items():
                    k = k + total_number_of_nodes
                    v = v + total_number_of_nodes
                    batch_mapper[k] = v
                return batch_mapper

            insert_graph_tp_batch_graph(built_graph_nodes_edges_mapper,
                                        self.batch_nodes_edges_mapper)
            insert_graph_tp_batch_graph(built_graph_nodes_index_id_mapper,
                                        self.batch_nodes_index_id_mapper)
            insert_graph_tp_batch_graph(built_graph_nodes_id_index_mapper,
                                        self.batch_nodes_id_index_mapper)

            adjacency_matrix, edges_matrix = built_graph.build_adjacency_matrix(
            )
            increasted_edges_matrix = edges_matrix + number_of_boxes  # increase the node id for each graph on the batch

            self.batch_edges_index.append(increasted_edges_matrix)
            self.batch_adjacency_matrix.append(adjacency_matrix)

            total_number_of_nodes += number_of_boxes

        self.batch_adjacency_matrix = block_diag(*self.batch_adjacency_matrix)
        self.batch_edges_index = np.concatenate(self.batch_edges_index, axis=1)


if __name__ == "__main__":
    pass