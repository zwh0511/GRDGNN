from __future__ import print_function

from itertools import cycle
import seaborn as sns
import tqdm
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
import matplotlib.pyplot as plt
from torch.autograd import Variable
import networkx as nx
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import argparse
import scipy.sparse as ssp
from sklearn.decomposition import PCA
import torch
from gensim.models import Word2Vec
from code.Inductive.MultiClassification import node2vec
import warnings
import numpy as np
from sklearn import metrics
from tqdm import *
import torch.nn as nn
import math
import os
import random
from sklearn.preprocessing import StandardScaler, scale
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
seed_value = 43
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-feat_dim', type=int, default=4, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=4, help='#classes')
cmd_args, _ = cmd_opt.parse_known_args()

class GNNGraph(object):
    def __init__(self, g, label, node_features=None, Nodes: tuple = (0, 1)):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = g.number_of_nodes()  # 子图的节点数目
        self.label = label  # 子图对应的标签
        self.node_features = node_features  # 节点特征：numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())  # 子图的度大小
        self.graph = g  # 子图对应的图
        self.nodes = Nodes  # 子图的中心节点

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.sigmoid(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.sigmoid(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            # select only false results
            mask = ~pred.eq(y.data.view_as(pred)).cpu()
            return logits, loss, acc, mask
        else:
            return logits


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
            self.alpha = torch.tensor(alpha).cuda()
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
            target_ = torch.zeros(target.size(0), self.num_class).cuda()
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

# Generate explicit features for inductive learning, get trends features
def genenet_attribute_dream(allx):
    if allx.shape[1] < cmd_args.feat_dim:
        raise Exception("The dim of Gene Expression is less than feat_dim, please resetup the feat_dim!!!!!")
    if cmd_args.feat_dim > 2:
        Relative_variation = np.zeros((allx.shape[0], cmd_args.feat_dim - 2))
        for i in range(cmd_args.feat_dim - 2):
            Relative_variation[:, i] = allx[:, i + 1] - allx[:, i]
        Relative_variation = scale(Relative_variation)

        # Relative_variation = np.zeros((allx.shape[0], allx.shape[1] - 1))
        # for i in range(allx.shape[1] - 1):
        #     Relative_variation[:, i] = allx[:, i + 1] - allx[:, i]
        #
        # #   相对变化量进行Z-score归一化
        # Relative_variation = scale(Relative_variation)
        #
        # #   提取cmd_args.feat_dim - 2 主成分作为节点特征向量
        # pca = PCA(n_components=cmd_args.feat_dim - 2)
        # Relative_variation = pca.fit_transform(Relative_variation[:, :cmd_args.feat_dim - 2])

        # 1: average to one dimension
        allx_ = StandardScaler().fit_transform(allx)
        trainAttributes = np.average(allx_, axis=1).reshape((len(allx), 1))
        trainAttributes = np.concatenate([trainAttributes, Relative_variation], axis=1)
        return trainAttributes
    else:
        return np.average(StandardScaler().fit_transform(allx), axis=1).reshape((len(allx), 1))
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    print("generate node2vec embeddings......")
    row, col, _ = ssp.find(A)
    A[row, col] = 1
    A[col, row] = 1
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1,workers=8)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings

def genenet_attribute_singleCell(allx):
    if allx.shape[1] < cmd_args.feat_dim:
        raise Exception("The dim of Gene Expression is less than feat_dim, please resetup the feat_dim!!!!!")
    if cmd_args.feat_dim > 2:
        # Relative_variation = np.zeros((allx.shape[0], allx.shape[1] - 1))
        # for i in range(allx.shape[1] - 1):
        #     Relative_variation[:, i] = allx[:, i + 1] - allx[:, i]

        #   相对变化量进行Z-score归一化
        Relative_variation = scale(allx)
        #   提取cmd_args.feat_dim - 2 主成分作为节点特征向量
        pca = PCA(n_components=cmd_args.feat_dim - 2)
        Relative_variation = pca.fit_transform(Relative_variation[:, :cmd_args.feat_dim - 2])
        # 1: average to one dimension
        allx_ = StandardScaler().fit_transform(allx)
        trainAttributes = np.average(allx_, axis=1).reshape((len(allx), 1))
        trainAttributes = np.concatenate([trainAttributes, Relative_variation], axis=1)
        return trainAttributes
    else:
        return np.average(StandardScaler().fit_transform(allx), axis=1).reshape((len(allx), 1))


def node_label(subgraph):
    """
    an implementation of the proposed double-radius node labeling (DRNL)

    加入有向图的节点标记

    DRNL + 有向图节点标记， 以此得到更加丰富的节点标记， 用于获取更多的信息


    :param subgraph: 需要进行节点标记的子图稀疏矩阵，该图为无向图

    :return: 返回图中节点的标签；
    """
    row, col, _ = ssp.find(subgraph)
    Af = np.zeros((subgraph.shape[0], subgraph.shape[0]))
    #   转换为无向图
    for i in range(row.size):
        Af[row[i], col[i]] = 1
        Af[col[i], row[i]] = 1
    subgraph = ssp.csc_matrix(Af)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0] + list(range(2, K)), :][:, [0] + list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels
# original version
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        _, nei1, _ = ssp.find(A[node, :])
        nei = set(nei)
        nei1 = set(nei1)
        res = res.union(nei)
        res = res.union(nei1)
    return res
def subgraph_extraction_labeling(ind, A, h: int = 1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # labels = np.zeros(len(nodes))
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    features = np.concatenate([features, labels.reshape(features.shape[0], 1)], axis=1)
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph, create_using=nx.DiGraph())
    # nx.draw(g, with_labels=True, pos=nx.circular_layout(g))
    # plt.show()
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    if g.has_edge(1, 0):
        g.remove_edge(1, 0)
    return g, features
def caculate_accuracy(trainNet_ori, trainNet_agent0, addnoisy=False, Threshold=0.70):
    """
    计算初始网络结构得准确率
    :param trainNet_ori: 基因调控网络得金标准网络，稀疏矩阵
    :param trainNet_agent0: 通过辨识方法得到得初始网络，稀疏矩阵
    """
    True_edges = trainNet_ori.todense()
    Predict_edges = trainNet_agent0.todense()
    True_row, True_col, _ = ssp.find(True_edges)
    True_count = len(True_row)
    Pro_row, Pro_col, _ = ssp.find(Predict_edges)
    bidirectional_link = 0

    if addnoisy:
        chan_count = int(True_count * Threshold)
        for i in range(True_count):
            if i % 2 == 0:

                Predict_edges[Pro_row[i], Pro_col[i]] = 0
                Predict_edges[Pro_col[i], Pro_row[i]] = 0
                chan_count -= 1
                if chan_count == 0:
                    break
            m, n = random.randint(0, trainNet_ori.shape[0] - 1), random.randint(0, trainNet_ori.shape[0] - 1)
            Predict_edges[m, n] = 1
        Pro_row, Pro_col, _ = ssp.find(Predict_edges)
    Predict_count = len(Pro_row)
    Right_count = 0
    if True_count < True_edges.shape[0]:
        for i in range(len(True_row)):
            for j in range(len(Pro_row)):
                if True_row[i] == Pro_row[j] and True_col[i] == Pro_col[j]:
                    Right_count += 1
                    # if Predict_edges[True_row[i], True_col[j]] == Predict_edges[True_col[j], True_row[i]] and Predict_edges[True_row[i], True_col[j]] == 1:
                    #     bidirectional_link += 1
    else:
        for i in range(True_edges.shape[0]):
            for j in range(True_edges.shape[1]):
                if True_edges[i, j] == Predict_edges[i, j] and True_edges[i, j] != 0:
                    Right_count += 1
                    # if Predict_edges[True_row[i], True_col[j]] == Predict_edges[True_col[j], True_row[i]] and Predict_edges[True_row[i], True_col[j]] == 1:
                    #     bidirectional_link += 1
    print()
    print("***************************************")
    print("*    The GoldEdges are: ", True_count)
    print("*    The PredictEdges are: ", Predict_count)
    print("*    The RightPredict Edges are: ", Right_count)
    print("*    The Predicted average degree of node: ", Predict_count / trainNet_agent0.shape[0])
    # print("*    The Right bidirectional_links are: ", bidirectional_link )
    print("*    The Rate of TruePositive of Precdict is: ", (Right_count / Predict_count) * 100, "%")
    print("*    The Rate of FalsePositive of Precdict is: ", ((Predict_count - Right_count) / Predict_count) * 100, "%")
    # print("*    The accuracy of Precdict is: ", (Right_count / True_count) * 100, "%")
    print("***************************************")
    print()
    if addnoisy:
        return ssp.csc_matrix(Predict_edges)


def TP_FP_TN_FN(cnf_matrix_agent0, total_num, Agent_name):
    # result = []
    # n_classes = cnf_matrix_agent0.shape[0]
    # metrics_result = []
    # for i in range(n_classes):
    #     # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
    #     ALL = np.sum(confusion_matrix)
    #     # 对角线上是正确预测的
    #     TP = confusion_matrix[i, i]
    #     # 列加和减去正确预测是该类的假阳
    #     FP = np.sum(confusion_matrix[:, i]) - TP
    #     # 行加和减去正确预测是该类的假阴
    #     FN = np.sum(confusion_matrix[i, :]) - TP
    #     # 全部减去前面三个就是真阴
    #     TN = ALL - TP - FP - FN
    #     metrics_result.append([TP / (TP + FP), TP / (TP + FN), TN / (TN + FP)])

    FP = cnf_matrix_agent0.sum(axis=0) - np.diag(cnf_matrix_agent0)
    FN = cnf_matrix_agent0.sum(axis=1) - np.diag(cnf_matrix_agent0)
    TP = np.diag(cnf_matrix_agent0)
    TN = cnf_matrix_agent0.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    Recall = TP / (TP + FN)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # Overall accuracy
    F1 = 2 * Precision * Recall / (Precision + Recall)
    Precision_total = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    Recall_total = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    MCC = (TP * TN - FP * FN) / np.power((TP + FP) * (FN + TP) * (FN + TN) * (FP + TN), 0.5)

    print("* " + Agent_name + ":       tp      fp      tn      fn")
    print("********************************************************************")
    print("*==》0:\t" + str(TP[0]) + "\t" + str(FP[0]) + "\t" + str(TN[0]) + "\t" + str(FN[0]))
    print("*==》1:\t" + str(TP[1]) + "\t" + str(FP[1]) + "\t" + str(TN[1]) + "\t" + str(FN[1]))
    print("*==》2:\t" + str(TP[2]) + "\t" + str(FP[2]) + "\t" + str(TN[2]) + "\t" + str(FN[2]))
    print("*==》3:\t" + str(TP[3]) + "\t" + str(FP[3]) + "\t" + str(TN[3]) + "\t" + str(FN[3]))
    print("*each_class__MCC: " + str(MCC))

    # print("*each_class__ACC: " + str(ACC))
    # print("*Average acc of calss: " + str(np.sum(ACC)/len(ACC)))
    # print("*Total acc of calss: " + str(acc))
    # print("*MicroF1: " + str(MicroF1))
    # print("*MacroF1: " + str(MacroF1))
    print("********************************************************************")


def RocPrint(y_label, y_score, dirtory="Dream5", testdataName="net1", Agent="Agent"):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_label = np.array(y_label)
    y_score = np.array(np.exp(y_score))
    n_classes = cmd_args.num_class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #
    # # micro（方法二）
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='orangered', linestyle='-', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='deeppink', linestyle='-', linewidth=2)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC--{}'.format(Agent))
    plt.legend(loc="lower right")
    # plt.savefig("/home/ubuntu/home/Projects/Final_DirGRGNN/DirGRGNN/Result/{}/{}/PR_{}.png".format(dirtory, testdataName, Agent), dpi=300)
    # plt.show()
    return roc_auc["micro"], roc_auc["macro"]


def PrPrint(label_onehot, score_array, dirtory="Dream5", testdataName="net1", Agent="Agent"):
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    score_array = np.exp(score_array)
    for i in range(cmd_args.num_class):
        precision_dict[i], recall_dict[i], _ = metrics.precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = metrics.average_precision_score(label_onehot[:, i], score_array[:, i])
        # print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # macro
    precision_dict["macro"], recall_dict["macro"], _ = metrics.precision_recall_curve(label_onehot.ravel(),
                                                                                      score_array.ravel())
    average_precision_dict["macro"] = metrics.average_precision_score(label_onehot, score_array, average="macro")
    # micro
    precision_dict["micro"], recall_dict["micro"], _ = metrics.precision_recall_curve(label_onehot.ravel(),
                                                                                      score_array.ravel())
    average_precision_dict["micro"] = metrics.average_precision_score(label_onehot, score_array, average="micro")
    lw = 2
    plt.figure()
    plt.step(recall_dict[0], precision_dict[0],
             label='0-average PR curve(AP = {0:0.2f})'
                   ''.format(average_precision_dict[0]),
             color='purple', linestyle=':', linewidth=1)
    plt.step(recall_dict[1], precision_dict[1],
             label='1-average PR curve(AP = {0:0.2f})'
                   ''.format(average_precision_dict[1]),
             color='orangered', linestyle='--', linewidth=1)
    plt.step(recall_dict[2], precision_dict[2],
             label='2-average PR curve(AP = {0:0.2f})'
                   ''.format(average_precision_dict[2]),
             color='deeppink', linestyle='-.', linewidth=1)
    plt.step(recall_dict[3], precision_dict[3],
             label='3-average PR curve(AP = {0:0.2f})'
                   ''.format(average_precision_dict[3]),
             color='seagreen', linestyle='-', linewidth=1)
    plt.step(recall_dict['micro'], precision_dict['micro'],
             label='all-average PR curve micro (AP = {0:0.2f})'
                   ''.format(average_precision_dict['micro']),
             color='red', linestyle='-', linewidth=2)
    plt.step(recall_dict['macro'], precision_dict['macro'],
             label='all-average PR curve macro(AP = {0:0.2f})'
                   ''.format(average_precision_dict['macro']),
             color='orangered', linestyle='-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PR--{}'.format(Agent))
    plt.legend(loc="lower right")
    # plt.savefig("/home/ubuntu/home/Projects/Final_DirGRGNN/DirGRGNN/Result/{}/{}_PR_{}.png".format(dirtory, testdataName, Agent), dpi=300)

    # plt.show()
    return average_precision_dict["micro"], average_precision_dict["macro"]


def network_info(network, TF_num=None):
    row, col, _ = ssp.find(network)
    label_1 = 0
    label_2 = 0
    label_3 = 0
    count_self = 0
    for i in range(len(row)):
        if network[row[i], col[i]] == network[col[i], row[i]]:
            if row[i] != col[i]:
                label_2 += 1
            else:
                count_self += 1
        elif row[i] < col[i]:
            label_1 += 1
        else:
            label_3 += 1
    A = network.todense()
    degree_of_nodes1 = np.sum(A, axis=1)
    degree_of_nodes0 = np.sum(A, axis=0)
    degree_of_nodes0 = degree_of_nodes0.T

    degree0 = []
    ij = ([], [])

    for i in range(network.shape[0]):
        if int(degree_of_nodes1[i]) + degree_of_nodes0[i] == 0:
            degree0.append(i)

    for i in range(len(row)):
        if int(degree_of_nodes1[row[i]]) + degree_of_nodes0[col[i]] < 3:
            ij[0].append(row[i])
            ij[1].append(col[i])

    maxcount = min(label_2 / 2, label_3, label_1)

    print("****************************************************")
    print("*    The proportion: " + str(int(label_1 / maxcount)) + ":" + str(int(label_2 / maxcount / 2)) + ":" + str(
        int(label_3 / maxcount)))
    print("*    The number of GeneA-->GeneB: ", label_1)
    print("*    The number of GeneA<-->GeneB: ", label_2 / 2)
    print("*    The number of GeneA<--GeneB: ", label_3)
    print("*    The total number of edges: ", len(row) - count_self)
    print("*    The no edge nodes are: ", len(degree0))
    # print(degree0)
    print("*    The only one edge node targets numbers are: ", len(ij[0]))
    # print(ij)
    print("****************************************************")

def sample_neg_balance(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, TF_num=333, balance=True):
    undirected_daj = net.todense()
    row, col, _ = ssp.find(net)
    for i in range(len(row)):
        undirected_daj[row[i], col[i]] = 1
        undirected_daj[col[i], row[i]] = 1
    net_triu = ssp.csc_matrix(undirected_daj)
    net_triu = ssp.triu(net_triu, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)

    # 将row、col打乱
    perm = random.sample(range(0, len(row)), len(row))
    random.shuffle(perm)
    row = row[perm]
    col = col[perm]

    label1 = ([], [])
    label2 = ([], [])
    label3 = ([], [])
    net = net.todense()
    for i in range(len(row)):
        if net[row[i], col[i]] == net[col[i], row[i]]:
            label2[0].append(row[i])
            label2[1].append(col[i])
        else:
            if net[col[i], row[i]] == 1 and net[row[i], col[i]] == 0:
                label1[0].append(row[i])
                label1[1].append(col[i])
            else:
                label3[0].append(row[i])
                label3[1].append(col[i])

    # 数据类别样本平衡处理
    # 获取label1、label2、label3中样本数量最大的数量
    max_num_Sample = int(np.max([len(label1[0]), len(label2[0]), len(label3[0])]))

    # sample negative links for train/test
    label0 = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict = {}
    while len(label0[0]) < max_num_Sample:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and net[i, j] == 0 and str(i) + "_" + str(j) not in recordDict:
            label0[0].append(i)
            label0[1].append(j)
            recordDict[str(i) + "_" + str(j)] = ''
        else:
            continue
    train_num_label0 = int(math.ceil(len(label0[0]) * (1 - test_ratio)))
    train_num_label1 = int(math.ceil(len(label1[0]) * (1 - test_ratio)))
    train_num_label2 = int(math.ceil(len(label2[0]) * (1 - test_ratio)))
    train_num_label3 = int(math.ceil(len(label3[0]) * (1 - test_ratio)))

    test_lable0 = (label0[0][train_num_label0:], label0[1][train_num_label0:])
    test_lable1 = (label1[0][train_num_label1:], label1[1][train_num_label1:])
    test_lable2 = (label2[0][train_num_label2:], label2[1][train_num_label2:])
    test_lable3 = (label3[0][train_num_label3:], label3[1][train_num_label3:])

    label0 = (label0[0][:train_num_label0], label0[1][:train_num_label0])
    label1 = (label1[0][:train_num_label0], label1[1][:train_num_label0])
    label2 = (label2[0][:train_num_label0], label2[1][:train_num_label0])
    label3 = (label3[0][:train_num_label0], label3[1][:train_num_label0])

    if balance:
        while len(label1[0]) < max_num_Sample / 2:
            if len(label1[0]) == 0:
                break
            i = np.random.random_integers(0, len(label1[0]) - 1)
            label1[0].append(label1[0][i])
            label1[1].append(label1[1][i])

        while len(label2[0]) < max_num_Sample / 2:
            if len(label2[0]) == 0:
                break
            i = np.random.random_integers(0, len(label2[0]) - 1)
            label2[0].append(label2[0][i])
            label2[1].append(label2[1][i])

        while len(label3[0]) < max_num_Sample / 2:
            if len(label3[0]) == 0:
                break
            i = np.random.random_integers(0, len(label3[0]) - 1)
            label3[0].append(label3[0][i])
            label3[1].append(label3[1][i])

    # 将label0、label1、label2、label3划分为训练集
    train_label0 = (label0[0][:], label0[1][:])
    train_lable1 = (label1[0][:], label1[1][:])
    train_lable2 = (label2[0][:], label2[1][:])
    train_lable3 = (label3[0][:], label3[1][:])

    return train_lable3, train_label0, test_lable3, test_lable0, train_lable1, train_lable2, test_lable1, test_lable2


def sample_all_gene_pairs(gold_net, TF_num=334, balance=False):
    labels = {i: ([], []) for i in range(4)}
    net = gold_net.todense()
    undirected_net = gold_net.todense()
    row, col, _ = ssp.find(gold_net)
    TF_dict = {}
    for i in range(len(row)):
        undirected_net[row[i], col[i]] = 1
        undirected_net[col[i], row[i]] = 1
    net_triu = ssp.csc_matrix(undirected_net)
    net_triu = ssp.triu(net_triu, k=1)
    gold_row, gold_col, _ = ssp.find(net_triu)

    for i in range(len(gold_row)):
        TF_dict[str(gold_row[i]) + "_" + str(gold_col[i])] = ''


    for i in tqdm(range(TF_num)):
        for j in range(gold_net.shape[0]):
            if i != j:
                if str(i) + "_" + str(j) in TF_dict:
                    if net[j, i] == net[i, j]:
                        labels[2][0].append(i)
                        labels[2][1].append(j)
                    else:
                        if net[j, i] == 1 and net[i, j] == 0:
                            labels[1][0].append(i)
                            labels[1][1].append(j)
                        else:
                            labels[3][0].append(i)
                            labels[3][1].append(j)
                else:
                    labels[0][0].append(i)
                    labels[0][1].append(j)

    if balance:
        for i in range(1, 4):
            while len(labels[i][0]) < len(labels[0][0]) // 2:
                if len(labels[i][0]) == 0:
                    break
                index = np.random.randint(0, len(labels[i][0]))
                labels[i][0].append(labels[i][0][index])
                labels[i][1].append(labels[i][1][index])

    train_labels = {i: (labels[i][0][:], labels[i][1][:]) for i in range(4)}

    return train_labels[3], train_labels[0], train_labels[1], train_labels[2]

def Cross_3_V_sample_neg_balance(net, test_ratio=0.1, max_train_num=None, balance=False, Partion=3):
    undirected_daj = net.todense()
    row, col, _ = ssp.find(net)
    for i in range(len(row)):
        undirected_daj[row[i], col[i]] = 1
        undirected_daj[col[i], row[i]] = 1
    net_triu = ssp.csc_matrix(undirected_daj)
    net_triu = ssp.triu(net_triu, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)

    # 将row、col打乱
    perm = random.sample(range(0, len(row)), len(row))
    random.shuffle(perm)
    row = row[perm]
    col = col[perm]

    label1 = ([], [])
    label2 = ([], [])
    label3 = ([], [])
    net = net.todense()
    for i in range(len(row)):
        if net[row[i], col[i]] == net[col[i], row[i]]:
            label2[0].append(row[i])
            label2[1].append(col[i])
        else:
            if net[col[i], row[i]] == 1 and net[row[i], col[i]] == 0:
                label1[0].append(row[i])
                label1[1].append(col[i])
            else:
                label3[0].append(row[i])
                label3[1].append(col[i])

    # 数据类别样本平衡处理
        # 获取label1、label2、label3中样本数量最大的数量
    max_num_Sample = int(np.max([len(label1[0]), len(label2[0]), len(label3[0])]))

    # sample negative links for train/test
    label0 = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict = {}
    while len(label0[0]) < max_num_Sample:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and net[i, j] == 0 and str(i) + "_" + str(j) not in recordDict:
            label0[0].append(i)
            label0[1].append(j)
            recordDict[str(i) + "_" + str(j)] = ''
        else:
            continue

    size_test_neg_start = math.floor(len(label0[0]) * (Partion) / 3)
    size_test_pos_start = math.floor(len(label1[0]) * (Partion) / 3)
    size_test_1_start = math.floor(len(label2[0]) * (Partion) / 3)
    size_test_2_start = math.floor(len(label3[0]) * (Partion) / 3)

    size_test_neg_end = math.floor(len(label0[0]) * (Partion + 1) / 3)
    size_test_pos_end = math.floor(len(label1[0]) * (Partion + 1) / 3)
    size_test_1_end = math.floor(len(label2[0]) * (Partion + 1) / 3)
    size_test_2_end = math.floor(len(label3[0]) * (Partion + 1) / 3)

    if Partion == 0:
        test_lable0 = (label0[0][:size_test_neg_end], label0[1][:size_test_neg_end])
        test_lable1 = (label1[0][:size_test_pos_end], label1[1][:size_test_pos_end])
        test_lable2 = (label2[0][:size_test_1_end], label2[1][:size_test_1_end])
        test_lable3 = (label3[0][:size_test_2_end], label3[1][:size_test_2_end])

        label0 = (label0[0][size_test_neg_end:], label0[1][size_test_neg_end:])
        label1 = (label1[0][size_test_pos_end:], label1[1][size_test_pos_end:])
        label2 = (label2[0][size_test_1_end:], label2[1][size_test_1_end:])
        label3 = (label3[0][size_test_2_end:], label3[1][size_test_2_end:])

    elif Partion == 1:

        test_lable0 = (label0[0][size_test_neg_start:size_test_neg_end], label0[1][size_test_neg_start:size_test_neg_end])
        test_lable1 = (label1[0][size_test_pos_start:size_test_pos_end], label1[1][size_test_pos_start:size_test_pos_end])
        test_lable2 = (label2[0][size_test_1_start:size_test_1_end], label2[1][size_test_1_start:size_test_1_end])
        test_lable3 = (label3[0][size_test_2_start:size_test_2_end], label3[1][size_test_2_start:size_test_2_end])

        label0 = (label0[0][:size_test_neg_start] + label0[0][size_test_neg_end:],
                     label0[1][:size_test_neg_start] + label0[1][size_test_neg_end:])
        label1 = (label1[0][:size_test_pos_start] + label1[0][size_test_pos_end:],
                     label1[1][:size_test_pos_start] + label1[1][size_test_pos_end:])
        label2 = (label2[0][:size_test_1_start] + label2[0][size_test_1_end:],
                         label2[1][:size_test_1_start] + label2[1][size_test_1_end:])
        label3 = (label3[0][:size_test_2_start] + label3[0][size_test_2_end:],
                         label3[1][:size_test_2_start] + label3[1][size_test_2_end:])
    elif Partion == 2:

        test_lable0 = (label0[0][size_test_neg_start:], label0[1][size_test_neg_start:])
        test_lable1 = (label1[0][size_test_pos_start:], label1[1][size_test_pos_start:])
        test_lable2 = (label2[0][size_test_1_start:], label2[1][size_test_1_start:])
        test_lable3 = (label3[0][size_test_2_start:], label3[1][size_test_2_start:])

        label0 = (label0[0][:size_test_neg_start], label0[1][:size_test_neg_start])
        label1 = (label1[0][:size_test_pos_start], label1[1][:size_test_pos_start])
        label2 = (label2[0][:size_test_1_start], label2[1][:size_test_1_start])
        label3 = (label3[0][:size_test_2_start], label3[1][:size_test_2_start])
    else:
        raise Exception("The partion was wrong!!!! Expected partion in {0,1,2}....")

    if balance:
        while len(label1[0]) < max_num_Sample / 2:
            if len(label1[0]) == 0:
                break
            i = np.random.random_integers(0, len(label1[0]) - 1)
            label1[0].append(label1[0][i])
            label1[1].append(label1[1][i])

        while len(label2[0]) < max_num_Sample / 2:
            if len(label2[0]) == 0:
                break
            i = np.random.random_integers(0, len(label2[0]) - 1)
            label2[0].append(label2[0][i])
            label2[1].append(label2[1][i])

        while len(label3[0]) < max_num_Sample / 2:
            if len(label3[0]) == 0:
                break
            i = np.random.random_integers(0, len(label3[0]) - 1)
            label3[0].append(label3[0][i])
            label3[1].append(label3[1][i])

    # 将label0、label1、label2、label3划分为训练集
    train_label0 = (label0[0][:], label0[1][:])
    train_lable1 = (label1[0][:], label1[1][:])
    train_lable2 = (label2[0][:], label2[1][:])
    train_lable3 = (label3[0][:], label3[1][:])

    return train_lable3, train_label0, test_lable3, test_lable0, train_lable1, train_lable2, test_lable1, test_lable2



def extractLinks2subgraphs_balance(Atrain, train_pos,train_neg,
                                   tranin_lable1, tranin_lable2,
                                   h: int = 1, max_nodes_per_hop=None,
                                   train_node_information=None):
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, node_information):
        g_list = []
        count = 0
        for i, j in tqdm(zip(links[0], links[1])):
            count += 1
            g, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            """对应标签子图可视化"""
            Nodes = (i, j)
            if g_label == 5:
                node_names = "Nodes: " + "(" + str(i) + "," + str(j) + ")" + "  Label: " + str(g_label)
                plt.title(node_names)
                nx.draw_networkx(g, with_labels=True, pos=nx.circular_layout(g))
                plt.show()
                # plt.savefig('./img/pic-{}.png'.format(epoch + 1))
            g_list.append(GNNGraph(g, g_label, n_features, Nodes))
        return g_list

    print('Extract enclosed subgraph...')
    train_graphs = helper(Atrain, train_neg, 0, train_node_information) + \
                   helper(Atrain, tranin_lable1, 1, train_node_information) + \
                   helper(Atrain, tranin_lable2, 2, train_node_information) + \
                   helper(Atrain, train_pos, 3, train_node_information)

    print(max_n_label)
    return train_graphs, max_n_label['value']

def extractLinks2subgraphs_balance_SVM(Atrain, train_pos,train_neg,
                                   tranin_lable1, tranin_lable2,
                                   h: int = 1, max_nodes_per_hop=None,
                                   train_node_information=None):
    # extract enclosing subgraphs
    def helper(A, links, g_label, node_information):
        g_list = []
        label_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            _, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            g_list.append(np.concatenate((n_features[0, :], n_features[1, :])))
            label_list.append(g_label)
        return g_list, label_list

    print('Extract enclosed subgraph...')
    train_graphs0, train_labels0 = helper(Atrain, train_neg, 0, train_node_information)
    train_graphs1, train_labels1 = helper(Atrain, tranin_lable1, 1, train_node_information)
    train_graphs2, train_labels2 = helper(Atrain, tranin_lable2, 2, train_node_information)
    train_graphs3, train_labels3 = helper(Atrain, train_pos, 3, train_node_information)
    train_graphs = train_graphs0 + train_graphs1 + train_graphs2 + train_graphs3
    train_labels = train_labels0 + train_labels1 + train_labels2 + train_labels3

    return train_graphs, train_labels


def Calculate_ACC_P_R(true_lable, prediction, agnet_name):
    print("**********【Accuracy，Precision，Recall，F1_score】*************")
    print(agnet_name)
    measure_result = metrics.classification_report(true_lable, prediction)
    print('measure_result = \n', measure_result)

    Precision_score = []
    print("----------------------------- precision（精确率）-----------------------------")
    precision_score_average_None = metrics.precision_score(true_lable, prediction, average=None)
    precision_score_average_micro = metrics.precision_score(true_lable, prediction, average='micro')
    precision_score_average_macro = metrics.precision_score(true_lable, prediction, average='macro')
    precision_score_average_weighted = metrics.precision_score(true_lable, prediction, average='weighted')

    # Precision_score.append(precision_score_average_None)
    Precision_score.append(precision_score_average_micro)
    Precision_score.append(precision_score_average_macro)
    Precision_score.append(precision_score_average_weighted)
    print('precision_score_average_None = ', precision_score_average_None)
    print('precision_score_average_micro = ', precision_score_average_micro)
    print('precision_score_average_macro = ', precision_score_average_macro)
    print('precision_score_average_weighted = ', precision_score_average_weighted)

    Recall_score = []
    print("\n\n----------------------------- recall（召回率）-----------------------------")
    recall_score_average_None = metrics.recall_score(true_lable, prediction, average=None)
    recall_score_average_micro = metrics.recall_score(true_lable, prediction, average='micro')
    recall_score_average_macro = metrics.recall_score(true_lable, prediction, average='macro')
    recall_score_average_weighted = metrics.recall_score(true_lable, prediction, average='weighted')

    # Recall_score.append(recall_score_average_None)
    Recall_score.append(recall_score_average_micro)
    Recall_score.append(recall_score_average_macro)
    Recall_score.append(recall_score_average_weighted)
    print('recall_score_average_None = ', recall_score_average_None)
    print('recall_score_average_micro = ', recall_score_average_micro)
    print('recall_score_average_macro = ', recall_score_average_macro)
    print('recall_score_average_weighted = ', recall_score_average_weighted)

    F1_score = []
    print("\n\n----------------------------- F1-value-----------------------------")
    f1_score_average_None = metrics.f1_score(true_lable, prediction, average=None)
    f1_score_average_micro = metrics.f1_score(true_lable, prediction, average='micro')
    f1_score_average_macro = metrics.f1_score(true_lable, prediction, average='macro')
    f1_score_average_weighted = metrics.f1_score(true_lable, prediction, average='weighted')

    # F1_score.append(f1_score_average_None)
    F1_score.append(f1_score_average_micro)
    F1_score.append(f1_score_average_macro)
    F1_score.append(f1_score_average_weighted)

    print('f1_score_average_None = ', f1_score_average_None)
    print('f1_score_average_micro = ', f1_score_average_micro)
    print('f1_score_average_macro = ', f1_score_average_macro)
    print('f1_score_average_weighted = ', f1_score_average_weighted)

    acc = metrics.accuracy_score(true_lable, prediction)
    print("The accuracy_score: ", acc)

    return acc, Precision_score, Recall_score, F1_score
def PrPrint(label_onehot, score_array, Agent="MI"):
    encoder = OneHotEncoder(sparse=False)
    label_onehot = encoder.fit_transform(np.array(label_onehot).reshape(-1, 1))
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    score_array = np.exp(score_array)
    for i in range(cmd_args.num_class):
        precision_dict[i], recall_dict[i], _ = metrics.precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = metrics.average_precision_score(label_onehot[:, i], score_array[:, i])
        # print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # macro
    precision_dict["macro"], recall_dict["macro"], _ = metrics.precision_recall_curve(label_onehot.ravel(),
                                                                                      score_array.ravel())
    average_precision_dict["macro"] = metrics.average_precision_score(label_onehot, score_array, average="macro")
    # micro
    precision_dict["micro"], recall_dict["micro"], _ = metrics.precision_recall_curve(label_onehot.ravel(),
                                                                              score_array.ravel())
    average_precision_dict["micro"] = metrics.average_precision_score(label_onehot, score_array, average="micro")
    # lw = 2
    # plt.figure()
    # plt.step(recall_dict[0], precision_dict[0],
    #          label='0-average PR curve(AP = {0:0.2f})'
    #                ''.format(average_precision_dict[0]),
    #          color='purple', linestyle=':', linewidth=1)
    # plt.step(recall_dict[1], precision_dict[1],
    #          label='1-average PR curve(AP = {0:0.2f})'
    #                ''.format(average_precision_dict[1]),
    #          color='orangered', linestyle='--', linewidth=1)
    # plt.step(recall_dict[2], precision_dict[2],
    #          label='2-average PR curve(AP = {0:0.2f})'
    #                ''.format(average_precision_dict[2]),
    #          color='deeppink', linestyle='-.', linewidth=1)
    # plt.step(recall_dict[3], precision_dict[3],
    #          label='3-average PR curve(AP = {0:0.2f})'
    #                ''.format(average_precision_dict[3]),
    #          color='seagreen', linestyle='-', linewidth=1)
    # plt.step(recall_dict['micro'], precision_dict['micro'],
    #          label='all-average PR curve micro (AP = {0:0.2f})'
    #                ''.format(average_precision_dict['micro']),
    #          color='red', linestyle='-', linewidth=2)
    # plt.step(recall_dict['macro'], precision_dict['macro'],
    #          label='all-average PR curve macro(AP = {0:0.2f})'
    #                ''.format(average_precision_dict['macro']),
    #          color='orangered', linestyle='-', linewidth=2)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('PR--{}'.format(Agent))
    # plt.legend(loc="lower right")
    # plt.savefig("/home/ubuntu/home/Projects/Final_DirGRGNN/DirGRGNN/Result/{}/{}_PR_{}.png".format(dirtory, testdataName, Agent), dpi=300)

    # plt.show()
    return average_precision_dict["micro"], average_precision_dict["macro"]

def readfeature(file, number_of_node):
    data = np.zeros((number_of_node, 128))
    count = -1
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if count > -1:
                line = line.strip()
                words = line.split()
                ncount = 0
                for word in words:
                    if word != str(count - 1):
                        data[count, ncount - 1] = word
                    ncount = ncount + 1
            count = count + 1
    return data


def draw_heatmap(allx, Name="heatmap"):
    plt.figure(dpi=300)
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(allx, cmap=cmap, annot=False)
    plt.title('{}_feature'.format(Name))
    plt.show()

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name:  # top-level parameters
            _param_init(p)

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(max_nodes=1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=0.8,
                        batch_size=50,
                        num_epochs=150,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
                        input_dim=10,
                        hidden_dim=1028,
                        output_dim=20,
                        num_classes=4,
                        num_gc_layers=5,
                        dropout=0.1,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()

def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

def draw_heatmap(allx, Name="heatmap"):

    plt.figure(dpi=300)
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(allx, cmap=cmap, annot=False)
    plt.title('{}_feature'.format(Name))
    plt.show()

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def SVM_RF_genenet_attribute(allx, tfNum):
    # 1: average to one dimension
    allx_ = StandardScaler().fit_transform(allx)
    trainAttributes = np.average(allx_, axis=1).reshape((len(allx), 1))
    allx_ = StandardScaler().fit_transform(allx)
    pca = PCA(n_components=2)
    pcaAttr = pca.fit_transform(allx_)
    trainAttributes = np.concatenate([trainAttributes, pcaAttr], axis=1)
    return trainAttributes

def writeReslut2File_dream(acc, precision, recall, F1, roc, pr, dataset_type='dream', network_type="MI", test_dataset="net3", threshold=0.3, hop=1, location="last", cross_val=True, experiment_count=0):
    if dataset_type == "dream":
        file_path = "/root/undergraduate/guozydata/expression2GRN/" + dataset_type + "/result/" + test_dataset + "/"
        if cross_val:
            filename = "Transductive_{}_location_{}_th_{}_hop_{}_{}.tsv".format(test_dataset, location, str(threshold), str(hop), network_type)

        else:
            filename = "Inductive_{}_th_{}_hop_{}_{}.tsv".format(test_dataset, str(threshold), str(hop), network_type)

        os.makedirs(os.path.dirname(file_path + filename), exist_ok=True)
        file = open(file_path + filename, "a")
        if experiment_count ==0:
            str0 = "Times\tAccuary\tp_micro\tp_macro\tp_weigh\tr_micro\tr_macro\tr_weigh\tF1_micro\tF1_macro\tF1_weigh\troc_micro\troc_macro\tpr_micro\tpr_macro"
            file.write(str0 + "\n")
        file.write(str(experiment_count) + "\t" + str(acc) + "\t")
        strP = ""
        strR = ""
        strF1 = ""
        for j in range(3):
            strP += str(precision[j]) + "\t"
            strR += str(recall[j]) + "\t"
            strF1 += str(F1[j]) + "\t"
        str2 = strP + strR + strF1
        file.write(str2)
        file.write(str(roc[0]) + "\t")
        file.write(str(roc[1]) + "\t")
        file.write(str(pr[0]) + "\t")
        file.write(str(pr[1]) + "\n")
        file.close()
def writeReslut2File(acc, precision, recall, F1, roc, pr, dataset_type='DREAM', train_dataset="net4",
                    network_type="MI", test_dataset="net3", threshold=0.3, scGRN_type = "STRING",
                    hop=1, location="last", cross_val=True, experiment_count=0,
                    gene_num="TFs+1000"):
    if dataset_type == "DREAM":
        file_path = "/root/sdc/guozy/GRDGNN/data/" + dataset_type + "/result/" + test_dataset + "/"
    else:
        file_path = "/root/sdc/guozy/GRDGNN/data/" + dataset_type + "/result/" + scGRN_type + "/" + test_dataset + "/" + gene_num + "/"
    if cross_val:
        filename = "Transductive_test_{}_location_{}_th_{}_hop_{}_{}.tsv".format(test_dataset, location, str(threshold),
                                                                            str(hop), network_type)

    else:
        filename = "Inductive_test_{}_train_{}_th_{}_hop_{}_{}.tsv".format(test_dataset, train_dataset, str(threshold), str(hop), network_type)
    os.makedirs(os.path.dirname(file_path + filename), exist_ok=True)
    file = open(file_path + filename, "a")
    if experiment_count ==0:
        str0 = "Times\tAccuary\tp_micro\tp_macro\tp_weigh\tr_micro\tr_macro\tr_weigh\tF1_micro\tF1_macro\tF1_weigh\troc_micro\troc_macro\tpr_micro\tpr_macro"
        file.write(str0 + "\n")
    file.write(str(experiment_count) + "\t" + str(acc) + "\t")
    strP = ""
    strR = ""
    strF1 = ""
    for j in range(3):
        strP += str(precision[j]) + "\t"
        strR += str(recall[j]) + "\t"
        strF1 += str(F1[j]) + "\t"
    str2 = strP + strR + strF1
    file.write(str2)
    file.write(str(roc[0]) + "\t")
    file.write(str(roc[1]) + "\t")
    file.write(str(pr[0]) + "\t")
    file.write(str(pr[1]) + "\n")
    file.close()
def writeallGeneResult2file(pairs, total_result, test_dataset):
    datanamedict = {
        "net1": "1",
        "net3": "3",
        "net4": "4"
    }
    file_path = "/root/undergraduate/guozydata/expression2GRN/dream/result/" + test_dataset + "/"
    filename = "DREAM5_NetworkInference_myteam_Network{}.txt".format(datanamedict[test_dataset])
    os.makedirs(os.path.dirname(file_path + filename), exist_ok=True)
    file = open(file_path + filename, "w+")
    TF = []
    Gene = []
    for k in range(len(pairs)):
        TF = TF + pairs[k][0].tolist()
        Gene = Gene + pairs[k][1].tolist()
    for i in range(total_result.shape[0]):
        label = np.argmax(total_result[i, :], axis=0)

        if label == 0:
            str1 = str(total_result[i][label])
        else:
            str1 = str(1 - total_result[i][0])

        str0 = "G" + str(TF[i]) + "\tG" + str(Gene[i]) + "\t"
        file.write(str0 + str1 + "\n")

        if label == 2:
            str0 = "G" + str(Gene[i]) + "\tG" + str(TF[i]) + "\t"
            file.write(str0 + str1 + "\n")
    file.close()
def get_project_dir(dir) -> str:
    """返回包含项目代码的目录"""
    # 获取当前模块所在的目录
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # 逐层向上查找，直到找到包含 myproject 子目录的目录为止
    while not os.path.exists(os.path.join(current_dir, dir)):
        current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if current_dir == "/":
            # 已经到达根目录，仍未找到包含 myproject 的目录
            raise Exception("找不到项目目录")
    return os.path.join(current_dir, dir + "/")

def directed2undirected(A):
    net = A.todense()
    row, col, _ = ssp.find(A)
    net[row, col] = 1
    net[col, row] = 1
    return ssp.csc_matrix(net)