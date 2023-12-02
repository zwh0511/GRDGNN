from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import warnings
import scipy.sparse as ssp
from units import weights_init, _param_init, glorot_uniform
warnings.filterwarnings("ignore")
from torch.nn import init
import torch.nn.functional as F
from set2set import Set2Set
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class DGCNConv(nn.Module):
    """
    DGCNConv achieved the multi_layer directed graph Convolution，
    zin = f(D_in(-1/2)(A_ + A_.T)D_in(-1/2)XW_in)
    zout = f(D_out(-1/2)(A_ + A_.T)D_out(-1/2)XW_out)
    Z = (alph*zin + beta*zout)/2
    """
    def __init__(self, output_dim, node_feats_dim, latent_dim=[32, 32, 32, 1], k=60):
        print('Initializing DGCNConv')
        super(DGCNConv, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.k = k
        self.num_node_feats = node_feats_dim
        self.total_latent_dim = np.sum(self.latent_dim)  # 控制特征链接的方式
        print("self.num_node_feats:", self.num_node_feats)

        """setting learnable parameters"""
        self.conv_params1 = nn.ModuleList()
        self.conv_params1.append(nn.Linear(self.num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params1.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        self.conv_params2 = nn.ModuleList()
        self.conv_params2.append(nn.Linear(self.num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params2.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        self.conv_params3 = nn.ModuleList()
        self.conv_params3.append(nn.Linear(latent_dim[0], latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params3.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))
        self.conv_params4 = nn.ModuleList()
        self.conv_params4.append(nn.Linear(latent_dim[0], latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params4.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        weights_init(self)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        A_T, D_in, D_out = self.PrepareMatrices(x, edge_index)
        x = x.to(device)
        A_T = A_T.to(device)
        D_in = D_in.to(device)
        D_out = D_out.to(device)
        x = Variable(x)
        A_T = Variable(A_T)
        D_in = Variable(D_in)
        D_out = Variable(D_out)

        h = self.GraphEmbedding(x, A_T, D_in, D_out, graphs=data)
        return h


    def GraphEmbedding(self,node_feat, A_T=None, D_in=None, D_out=None, graphs=None):
        lv = 0
        cur_message_layer1 = node_feat
        cur_message_layer1 = cur_message_layer1.to(torch.float32)
        cur_message_layer2 = node_feat
        cur_message_layer2 = cur_message_layer2.to(torch.float32)
        cat_message_layers1_2 = []
        while lv < len(self.latent_dim):
            """z = f(D_in(-1/2)(A_ + A_.T)D_in(-1/2)XW)"""
            n2npool1 = torch.mm(D_in, A_T)
            A_1 = torch.mm(n2npool1, D_in)
            n2npool1 = torch.mm(A_1, cur_message_layer1)  # + cur_message_layer1
            node_linear1 = self.conv_params1[lv](n2npool1)
            cur_message_layer1 = torch.tanh(node_linear1)
            """z = f(D_out(-1/2)(A_ + A_.T)D_out(-1/2)XW)"""
            n2npool2 = torch.mm(D_out, A_T)
            A_2 = torch.mm(n2npool2, D_out)
            n2npool2 = torch.mm(A_2, cur_message_layer2)  # + cur_message_layer2
            node_linear2 = self.conv_params2[lv](n2npool2)
            cur_message_layer2 = torch.tanh(node_linear2)
            cur_message_layer_out = 0.5 * (0.1 * cur_message_layer1 + 0.9 * cur_message_layer2)
            """The second GCN layer"""
            n2npool3 = torch.mm(A_1, cur_message_layer_out)
            node_linear3 = self.conv_params3[lv](n2npool3)
            cur_message_layer3 = torch.tanh(node_linear3)
            n2npool4 = torch.mm(A_2, cur_message_layer_out)
            node_linear4 = self.conv_params3[lv](n2npool4)
            cur_message_layer4 = torch.tanh(node_linear4)
            cat_message_layers1_2.append(0.5 * (0.1 * cur_message_layer3 + 0.9 * cur_message_layer4))
            lv += 1
        cur_message_layer = torch.cat(cat_message_layers1_2, 1)

        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(graphs.num_graphs, self.k, self.total_latent_dim)
        batch_sortpooling_graphs = batch_sortpooling_graphs.to(device)
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(graphs.num_graphs):
            to_sort = sort_channel[accum_count: accum_count + graphs[i].num_nodes]
            k = self.k if self.k <= graphs[i].num_nodes else graphs[i].num_nodes
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                to_pad = to_pad.to(device)
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graphs[i].num_nodes

        return batch_sortpooling_graphs

    def PrepareMatrices(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row_idxes, col_vals = edge_index.cpu()
        data_val = torch.ones(len(row_idxes))
        A = ssp.csc_matrix((data_val, (row_idxes, col_vals)), shape=(x.shape[0], x.shape[0]))
        A = A.todense()
        A_T = A.T + A
        A_self = torch.from_numpy(A)
        D_in = torch.sum(A_self, dim=0)
        D_out = torch.sum(A_self, dim=1)
        A_T = torch.from_numpy(A_T)
        D_in = torch.pow(D_in, -0.5)
        D_out = torch.pow(D_out, -0.5)
        D_in = torch.diag(D_in)
        D_out = torch.diag(D_out)
        return A_T, D_in, D_out


class FocalLoss(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, alpha=0.25, gamma=2, use_alpha=True, size_average=True,
                 with_dropout=False):
        super(FocalLoss, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).to(device)
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        h1 = self.h1_weights(pred)
        h1 = F.sigmoid(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = self.softmax(logits)

        prob = logits
        prob = prob.clamp(min=0.0001, max=1.0)

        if target is not None:
            target_ = torch.zeros(target.size(0), self.num_class).to(device)
            target_.scatter_(1, target.view(-1, 1).long(), 1.)
            target = Variable(target)
            pred1 = logits.data.max(1, keepdim=True)[1]
            acc = pred1.eq(target.data.view_as(pred1)).cpu().sum().item() / float(target.size()[0])
            # select only false results
            mask = ~pred1.eq(target.data.view_as(pred1)).cpu()
            if self.use_alpha:
                batch_loss = - self.alpha.double() * torch.pow(1 - prob,
                                                               self.gamma).double() * prob.log().double() * target_.double()
            else:
                batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

            batch_loss = batch_loss.sum(dim=1)

            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return logits, loss, acc, mask
        else:
            return logits

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, alph=0.6, beta=0.4, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout).to(device)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(device))
        self.weight2 = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(device))
        self.alph = nn.Parameter(torch.tensor(alph, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(device))
        else:
            self.bias = None

    def forward(self, x, adj_in, adj_out):

        # if self.dropout > 0.001:
        #     x = self.dropout_layer(x)

        y1 = torch.matmul(adj_in, x)
        y2 = torch.matmul(adj_out, x)
        y1 = torch.matmul(y1, self.weight1)
        y2 = torch.matmul(y2, self.weight1)
        y = 0.5 * (self.alph * y1 + self.beta * y2)
        # y = 0.5 * (0.5 * y1 + 0.5 * y2)
        if self.bias is not None:
            y = y + self.bias

        y = F.relu(y)

        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight1.data = init.xavier_uniform(m.weight1.data, gain=nn.init.calculate_gain('relu'))
                m.weight2.data = init.xavier_uniform(m.weight2.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(device)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(device)
        return bn_module(x)

    def gcn_forward(self, x, adj_in, adj_out, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj_in, adj_out)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj_in, adj_out)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj_in, adj_out)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj_in, adj_out, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj_in.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj_in, adj_out)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj_in, adj_out)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj_in, adj_out)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred

    def loss(self, pred, label, type='margin'):
        # softmax + CE
        if type == 'softmax':
            return F.multilabel_margin_loss(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().to(device)
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj_in, adj_out, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj_in.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj_in, adj_out,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [1024], assign_dim, num_aggs=1)


            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        # self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims,
        #         label_dim, num_aggs=self.num_aggs)
        self.pred_model = FocalLoss(assign_input_dim*2, hidden_dim, label_dim, with_dropout=True)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight1.data = init.xavier_uniform(m.weight1.data, gain=nn.init.calculate_gain('relu'))
                m.weight2.data = init.xavier_uniform(m.weight2.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj_in, adj_out, labels, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj_in.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []
        embedding_tensor = self.gcn_forward(x, adj_in, adj_out,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj_in, adj_out,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj_in = torch.transpose(self.assign_tensor, 1, 2) @ adj_in @ self.assign_tensor
            adj_out = torch.transpose(self.assign_tensor, 1, 2) @ adj_out @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj_in, adj_out,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output, labels)

        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss





