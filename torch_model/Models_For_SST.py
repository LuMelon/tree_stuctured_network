import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeLSTM_2ary(nn.Module):

    def __init__(self, hiddendim=50, Nclass=5, max_degree=2):
        super(TreeLSTM_2ary, self).__init__()
        self.hiddendim = hiddendim
        self.Nclass = Nclass
        self.degree = max_degree

        # parameters for the model
        self.W_i = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.hiddendim]))
        self.U_i = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_i = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_f = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.hiddendim]))
        self.U_f = nn.parameter.Parameter(self.init_matrix([self.degree, self.degree, self.hiddendim, self.hiddendim]))
        self.b_f = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_o = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.hiddendim]))
        self.U_o = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_o = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_u = nn.parameter.Parameter(self.init_matrix([self.hiddendim, self.hiddendim]))
        self.U_u = nn.parameter.Parameter(self.init_matrix([self.degree, self.hiddendim, self.hiddendim]))
        self.b_u = nn.parameter.Parameter(self.init_vector([self.hiddendim]))

        self.W_s = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hiddendim]))
        self.b_s = nn.parameter.Parameter(self.init_vector([self.Nclass]))

        self.drop = nn.Dropout(p=0.5)

    def recursive_unit(self, parent_word, child_hs, child_cells):
        def manyU_mul_manyH(manyU):
            sum = manyU[0].mul(child_hs[0]).sum(dim=1)
            for i in range(1, self.degree):
                sum += manyU[i].mul(child_hs[i]).sum(dim=1)
            return sum
        def forget_manycells(f_gates):
            sum = f_gates[0].mul(child_cells[0]).sum(dim=1)
            for i in range(1, self.degree):
                sum += f_gates[i].mul(child_cells[i]).sum(dim=1)
            return sum
        input = F.sigmoid( self.W_i.mul(parent_word) + manyU_mul_manyH(self.U_i) + self.b_i)
        output = F.sigmoid(self.W_o.mul(parent_word) + manyU_mul_manyH(self.U_o) + self.b_o)
        utility = F.tanh(self.W_u.mul(parent_word) + manyU_mul_manyH(self.U_u) + self.b_u)
        forgets = [F.sigmoid(self.W_f.mul(parent_word) + manyU_mul_manyH(this_U_f) + self.b_f) for this_U_f in self.U_f]
        parent_cell = input.mul(utility) + forget_manycells(forgets)
        parent_h = output.mul(F.tanh(parent_cell))
        return parent_h, parent_cell

    def forward(self, tree):
        self.init_tree(tree)
        self.computeTree(tree)
        self.PredictUpTree(tree)
        loss = self.computeLoss(tree)
        return loss

    def computeLoss(self, tree):

        return

    def PredictUpTree(self, tree):
        def PredictUpNode(node_h):
            node_h = self.drop(node_h)
            distrib = F.softmax( self.W_s.mul(node_h).sum(dim=1) + self.b_s)
            pred = distrib.argmax()
            return pred, distrib[pred]
        nodes = tree.tree.nodes
        for node in nodes:
            pred, prob = PredictUpNode(nodes[node]['node_h'])
            nodes[node]['pred'] = pred
            node [node]['prob'] = prob
        return

    def init_tree(self, tree):
        nodes = tree.tree.nodes
        for node in nodes:
            nodes[node]['node_h'] = self.init_vector([self.hiddendim])
            nodes[node]['cell'] = self.init_vector([self.hiddendim])
        return

    def computeTree(self, tree):
        leaf_nodes = tree.LeafNodes()
        def handle_leaf_node(leaf):

            leaf_h, leaf_cell = self.recursive_unit()
            return

        return

    def init_vector(self, shape):
        return torch.zeros(shape)

    def init_matrix(self, shape):
        return  torch.from_numpy(np.random.normal(scale=0.1, size=shape))

    def initWordVecMatrix(self, words):
        self.Words = nn.parameter.Parameter(words)