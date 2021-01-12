import math
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter

class ScenePriorsModel(nn.Module):
    """Scene Priors implementation"""
    def __init__(
        self, 
        action_sz, 
        state_sz = 8192, 
        target_sz = 300,
        ):
        super(ScenePriorsModel, self).__init__()
        target_embed_sz = 300
        self.fc_target = nn.Linear(target_sz, target_embed_sz)
        # Observation layer
        self.fc_state = nn.Linear(state_sz, 512)

        # GCN layer
        self.gcn = GCN()

        # Merge word_embedding(300) + observation(512) + gcn(512)
        self.navi_net = nn.Linear(
            target_embed_sz + 512 + 512, 512)
        self.navi_hid = nn.Linear(512,512)
        #output
        self.actor_linear = nn.Linear(512, action_sz)
        self.critic_linear = nn.Linear(512, 1)

    def forward(self, model_input):

        x = model_input['fc|4'].reshape(-1, 8192)
        y = model_input['glove']
        z = model_input['score']

        #x = x.view(-1)
        #print(x.shape)
        x = self.fc_state(x)
        x = F.relu(x, True)

        #y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        z = self.gcn(z)

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat((x, y, z), dim = 1)
        xyz = self.navi_net(xyz)
        xyz = F.relu(xyz, True)
        xyz = F.relu(self.navi_hid(xyz), True)
        return dict(
            policy=self.actor_linear(xyz),
            value=self.critic_linear(xyz)
            )

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        # Load adj matrix for GCN
        A_raw = torch.load("../thordata/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        objects = open("../thordata/gcn/objects.txt").readlines()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        self.register_buffer('all_glove', torch.zeros(self.n, 300))

        glove = h5py.File("../thordata/word_embedding/thor_glove/glove_map300d.hdf5","r",)
        for i in range(self.n):
            self.all_glove[i, :] = torch.from_numpy(glove[objects[i]][:])

        glove.close()

        nhid = 1024
        # Convert word embedding to input for gcn
        self.word_to_gcn = nn.Linear(300, 512)

        # Convert resnet feature to input for gcn
        self.resnet_to_gcn = nn.Linear(1000, 512)

        # GCN net
        self.gc1 = GraphConvolution(512 + 512, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, 1)

        self.mapping = nn.Linear(self.n, 512)

    def gcn_embed(self, x, params):
        if params == None:
            resnet_embed = self.resnet_to_gcn(x)
            word_embedding = self.word_to_gcn(self.all_glove)

            n_steps = resnet_embed.shape[0]
            resnet_embed = resnet_embed.repeat(self.n,1,1)

            output = torch.cat(
                (resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)), 
                dim=2
                )
        else:
            resnet_embed = F.linear(
                x,
                weight=params["resnet_to_gcn.weight"],
                bias=params["resnet_to_gcn.bias"],
            )
            word_embedding = F.linear(
                self.all_glove,
                weight=params["word_to_gcn.weight"],
                bias=params["word_to_gcn.bias"],
            )

            n_steps = resnet_embed.shape[0]
            resnet_embed = resnet_embed.repeat(self.n,1,1)

            output = torch.cat(
                (resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)), 
                dim=2
                )
        return output

    def forward(self, x, params = None):

        # x = (current_obs)
        # Convert input to gcn input
        x = self.gcn_embed(x, params)
        if params == None:
            x = F.relu(self.gc1(x, self.A))
            x = F.relu(self.gc2(x, self.A))
            x = F.relu(self.gc3(x, self.A))
            x.squeeze_(-1)
            x = self.mapping(x)
        else:
            gc_p = [
                dict(
                    weight = params[f'gc{x}.weight'], bias = params[f'gc{x}.bias']
                    )
                for x in [1,2,3]
                ]
            x = F.relu(self.gc1(x, self.A, gc_p[0]))
            x = F.relu(self.gc2(x, self.A, gc_p[1]))
            x = F.relu(self.gc3(x, self.A, gc_p[2]))
            x.squeeze_(-1)
            x = F.linear(
                    x,
                    weight=params["mapping.weight"],
                    bias=params["mapping.bias"],
                )

        return x

# Code borrowed from https://github.com/tkipf/pygcn
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, params = None):
        if params == None:
            support = torch.matmul(input, self.weight)
            output = torch.matmul(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output
        else:
            support = torch.matmul(input, params['weight'])
            output = torch.matmul(adj, support)
            if self.bias is not None:
                return output + params['bias']
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

if __name__ == "__main__":
    model = ScenePriorsModel(9)
    input1 = torch.randn(4,8192)
    input2 = torch.randn(4,300)
    input3 = torch.randn(4,300,400,3)
    #print(input2)
    out = model.forward({'fc|4':input1, 'glove':input2, 'RGB':input3})
    print(out['policy'])
    print(out['value'])

