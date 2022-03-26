#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy as scip
import networkx as nx
import torch
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
from base_mm_graph_layer import CMGATLayerBase
from base_graph_op import Nodes


class CMGATLayerImp(CMGATLayerBase):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self,
                 cross_modal_node_encode_layer, 
                 cross_modal_attn_model,
                 num_in_node_features,
                 num_in_phrase_features
                 num_out_node_features,
                 num_out_phrase_features,
                 num_of_heads,
                 concat=True,
                 activation=nn.ELU(),
                 dropout_prob=0.6,
                 add_skip_connection=True,
                 bias=True,
                 log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(cross_modal_attn_model, num_in_features,
                         num_out_features, num_of_heads, concat, activation,
                         dropout_prob, add_skip_connection, bias,
                         log_attention_weights)


        self.cross_modal_node_encode_layer = cross_modal_node_encode_layer

    def node_transform(self, multi_head_nodes_features,
                       multi_head_phrases_features, mask):
        """[Obtain the matching scores of the nodes and the phrases]

        Args:
            multi_head_nodes_features (troch.tensor): [tensor with shape [-1, num_of_heads, num_of_visual_features]]
            multi_head_phrases_features (troch.tensor): [tensor with shape [-1, num_of_heads, num_of_text_features]]
            mask (troch.tensor): [description]

            Note, we expect the num_of_visual_features == num_of_text_features
        
        Returns:
            attn_text2visual_concepts_fs [torch.tensor]: [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
            v2t_scores [torch.tensor]: [a tensor with shape <batch_size, number_of_heads, num_phrases, num_nodes>]
        """

        attn_text2visual_fs, v2t_scores = self.cross_modal_attn_model(
            multi_head_nodes_features, multi_head_phrases_features, mask)

        return attn_text2visual_fs, v2t_scores

    def forward(self, graph_data, text_data, mask):
        #
        # Step 1: Linear Projection + regularization
        #
        in_phrase_features = text_data
        in_nodes_features, in_edge_features, edge_index = graph_data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[
            0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)
        in_phrase_features = self.dropout(in_phrase_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.node_linear_proj(in_nodes_features).view(
            -1, self.num_of_heads, self.num_out_node_features)
        phrases_features_proj = self.phrase_linear_proj(
            in_phrase_features).view(-1, self.num_of_heads,
                                     self.num_out_phrase_features)

        nodes_features_proj = self.dropout(
            nodes_features_proj
        )  # in the official GAT imp they did dropout here as well
        phrases_features_proj = self.dropout(
            phrases_features_proj
        )  

        #
        # Step 2: Edge attention calculation
        #

        # nodes_phrases_response_scores is obtained by using the phrases as the query
        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        attn_text2visual_fs, nodes_phrases_response_scores = self.node_transform(nodes_features_proj, phrases_features_proj, mask)

        # convert scores to the shape <batch_size, num_nodes, num_heads>
        nodes_phrases_response_scores = torch.squeeze(nodes_phrases_response_scores, 2)
        nodes_phrases_response_scores = torch.transpose(nodes_phrases_response_scores, 1, 2)

        # fusion the text to the visual to generate the enhanced node features
        encoded_cm_nodes_features_proj=self.cross_modal_node_encode_layer(visual_nodes_feas=nodes_features_proj, 
                                                                            text_phrases_feas=attn_text2visual_fs)


        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        nodes_scores_source, nodes_features_proj_lifted = self.lift(nodes_phrases_response_scores, encoded_cm_nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(nodes_scores_source)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(
            scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(
            nodes_features_proj_lifted_weighted, edge_index, in_nodes_features,
            num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge,
                                                   in_nodes_features,
                                                   out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index,
                                   num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (
            neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge,
                                           trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(
            trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape
                    )  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size,
                                        dtype=exp_scores_per_edge.dtype,
                                        device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted,
                                       exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted,
                            edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape
                    )  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size,
                                         dtype=in_nodes_features.dtype,
                                         device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim],
            nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted,
                                        nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, nodes_response_scores, nodes_features_matrix_proj,
             edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        # trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        nodes_scores_source = nodes_response_scores.index_select(self.nodes_dim,
                                                   src_nodes_index)
        # scores_target = scores_target.index_select(self.nodes_dim,
        #                                            trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(
            self.nodes_dim, src_nodes_index)

        return nodes_scores_source, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
