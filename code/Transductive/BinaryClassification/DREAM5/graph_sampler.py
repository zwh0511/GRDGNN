import os

import pandas as pd
import scipy.sparse as ssp
import networkx as nx
import numpy as np
import torch
import torch.utils.data
from torch_geometric.data import Dataset, Data
from code.Transductive.BinaryClassification.DREAM5.units import cmd_args
from code.Transductive.BinaryClassification.DREAM5 import units as uf
import tqdm
seed_value = 43
np.random.seed(seed_value)
torch.manual_seed(seed_value)

data_dim ={
    "hESC":758,
    "hHEP":425,
    "mDC":383,
    "mESC":421,
    "mHSC-E":1071,
    "mHSC-GM":889,
    "mHSC-L":847
}
class GRDGrahp(Dataset):
    def __init__(self, root='/root/sdc/guozy/GRDGNN/data', dataset_type='DREAM', network_type="Pearson", scGRN_type = "STRING",
                 cross_val_location="last", use_embedding=False, embedding_dim = 1, directed2undirected=False, use_gold_grn = True,
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None, all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False, gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        self.filepath = root + os.sep + dataset_type + os.sep + "processed/"
        self.filenames = os.listdir(self.filepath)
        self.network_type = network_type
        self.scGRN_type = scGRN_type
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.threshold = threshold
        self.hop = hop
        self.test_ratio = test_ratio
        self.cross_val = cross_val
        self.max_nodes_per_hop = max_nodes_per_hop
        self.balance = balance
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.gene_num = gene_num
        self.avgDegree = avgDegree
        self.all_gene_pairs = all_TF_gene_pairs
        self.TFs_num = TFs_num
        self.test_flag = test_flag
        self.cross_val_location = cross_val_location
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.use_gold_grn = use_gold_grn
        self.directed2undirected = directed2undirected


        # if max_num_nodes == 0:
        #     self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        # else:
        #     self.max_num_nodes = max_num_nodes

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        # self.slices：一个切片字典，用于从该对象重构单个示例

    @property
    def raw_dir(self):
        """默认也是self.root/raw"""
        return self.filepath

    @property
    def processed_dir(self):
        """默认是self.root/processed"""
        if self.dataset_type == "DREAM":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.dataset_name)
        elif self.dataset_type == "SingleCell":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.scGRN_type + "/" + self.dataset_name + "/" + self.gene_num )
        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")
    @property
    def raw_file_names(self):
        """"原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self):
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        file_name_parts = [self.dataset_name, self.network_type, self.scGRN_type, "Th" + str(self.threshold), "hop_" + str(self.hop)]
        if self.test_flag:
            file_name_parts.append("Test_mask")
        if self.cross_val:
            if self.test_flag:
                file_name_parts.append("Three_Cross_test_" + self.cross_val_location)
            else:
                file_name_parts.append("Three_Cross_train_" + self.cross_val_location)
        if self.balance:
            file_name_parts.append("balanced")
        if self.dataset_type == "singleCell":
            file_name_parts.append(self.gene_num + "_avg_" + str(self.avgDegree))
        if self.all_gene_pairs:
            file_name_parts.append("all_TF_gene_pairs")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        if self.use_embedding:
            file_name_parts.append("embedding_dim_" + str(self.embedding_dim))
        if self.directed2undirected:
            file_name_parts.append("directed2undirected")
        if self.use_gold_grn:
            file_name_parts.append("use_grn")
        file_name_parts.append("binary")
        file_name_parts.append(".pt")
        return ['_'.join(file_name_parts)]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        """主程序，对原始数据进行处理"""
        if self.dataset_type == "DREAM":
            trainNet_ori = np.load(os.path.join(
                self.filepath + "{}/goldGRN_{}.csc".format(self.dataset_name, self.dataset_name)),
                                   allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + "{}/goldExpression_{}.allx".format(self.dataset_name, self.dataset_name)),
                                 allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                        str(self.threshold))),
                                                                                        allow_pickle=True).tolist()
            Atrain_agent0 = trainNet_agent0.copy()  # the observed network

            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + "{}/MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                            str(self.threshold))),
                                                                                            allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_dream(allx)


        elif self.dataset_type == "SingleCell":
            singleCell_path2file = "{}/{}/{}/".format(self.scGRN_type, self.dataset_name, self.gene_num)
            trainNet_ori = np.load(os.path.join(
                self.filepath + singleCell_path2file +"goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()

            Atrain_agent0 = trainNet_agent0.copy()  # the observed network
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_singleCell(allx)

        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

        data = []

        if self.all_gene_pairs and self.cross_val:
            if self.test_flag:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                _, _, train_pos, train_neg = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)

            else:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)
        else:
            if self.directed2undirected:
                Atrain_agent0 = uf.directed2undirected(Atrain_agent0)
            if self.cross_val:
                self.test_ratio = 0.33  # 3 folder cross_val
                if self.test_flag:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    _, _, train_pos, train_neg = uf.Cross_3_V_sample_neg_binary(trainNet_ori,
                                                                                 max_train_num=1000000,
                                                                                 Partion=Partition)
                else:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary(trainNet_ori,
                                                                                max_train_num=1000000,
                                                                                Partion=Partition)
            else:
                train_pos, train_neg, _, _ = uf.sample_neg_binary(trainNet_ori,
                                                           test_ratio=0.0,
                                                           TF_num=self.TFs_num,
                                                           max_train_num=1000000)

        if self.use_embedding:
            train_embeddings = uf.generate_node2vec_embeddings(Atrain_agent0, self.embedding_dim, True, train_neg)
            trainAttributes = np.concatenate([train_embeddings, trainAttributes], axis=1)
        if self.use_gold_grn:
            Atrain_agent0 = trainNet_ori
        if self.test_flag:
            Atrain_agent0[train_pos[0], train_pos[1]] = 0  # mask test links
            Atrain_agent0[train_pos[1], train_pos[0]] = 0  # mask test links
        train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                    train_pos,
                                                                                    train_neg,
                                                                                    self.hop,
                                                                                    self.max_nodes_per_hop,
                                                                                    trainAttributes)

        for i in range(len(train_graphs_agent0)):
            feat_x0 = torch.from_numpy(train_graphs_agent0[i].node_features)
            adj_x0 = np.array(nx.adjacency_matrix(train_graphs_agent0[i].graph).todense())
            row_x0, col_x0, _ = ssp.find(adj_x0)
            edge_index0 = torch.tensor([row_x0, col_x0], dtype=torch.long)
            y_x0 = train_graphs_agent0[i].label
            y_x0 = torch.tensor([y_x0], dtype=torch.long)
            mydata = Data(x=feat_x0, edge_index=edge_index0, y=y_x0)
            data.append(mydata)
        self.data = data
        torch.save((self.data), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        return data


class DGraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def __init__(self, Data, features='default', normalize=False, assign_feat='default', max_num_nodes=0):
        self.adj_all_in = []
        self.adj_all_out = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.assign_feat_all = []


        if max_num_nodes == 0:
            self.max_num_nodes = max([data.num_nodes for data in Data])
        else:
            self.max_num_nodes = max_num_nodes

        # if features == 'default':
        self.feat_dim = Data[0].x.shape[1]

        for data in tqdm.tqdm(Data):
            row_idxes, col_vals = data.edge_index
            data_val = torch.ones(len(row_idxes))
            A = ssp.csc_matrix((data_val, (row_idxes, col_vals)), shape=(data.x.shape[0], data.x.shape[0]))
            adj = A.todense() + np.eye(data.x.shape[0])
            A_self = torch.from_numpy(adj)
            D_in = torch.sum(A_self, dim=0)
            D_out = torch.sum(A_self, dim=1)
            D_in = torch.pow(D_in, -0.5)
            D_out = torch.pow(D_out, -0.5)
            D_in = torch.diag(D_in)
            D_out = torch.diag(D_out)
            adj = torch.from_numpy(adj)
            adj = adj + adj.T
            adj_in = torch.matmul(D_in, torch.matmul(adj, D_in))
            adj_out = torch.matmul(D_out, torch.matmul(adj, D_out))
            self.adj_all_in.append(adj_in)
            self.adj_all_out.append(adj_out)
            self.len_all.append(data.num_nodes)
            self.label_all.append(data.y)
            # 将图归一化大小
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i in range(data.num_nodes):
                    f[i, :] = data.x[i, :]
                self.feature_all.append(f)
            if assign_feat == 'id':
                self.assign_feat_all.append(
                    np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])))
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all_in)

    def __getitem__(self, idx):
        adj_in = self.adj_all_in[idx]
        adj_out = self.adj_all_out[idx]
        num_nodes = adj_in.shape[0]
        adj_padded_in = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded_in[:num_nodes, :num_nodes] = adj_in
        adj_padded_out = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded_out[:num_nodes, :num_nodes] = adj_out

        # use all nodes for aggregation (baseline)
        return {'adj_in': adj_padded_in,
                'adj_out': adj_padded_out,
                'feats': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats': self.assign_feat_all[idx].copy()}

class savedGRDGrahp(Dataset):
    def __init__(self, root='/root/undergraduate/guozydata/expression2GRN', dataset_type='dream', network_type="Pearson", scGRN_type = "STRING",
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None, all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False, gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        self.filepath = root + os.sep + dataset_type + os.sep + "raw/"
        self.filenames = os.listdir(self.filepath)
        self.network_type = network_type
        self.scGRN_type = scGRN_type
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.threshold = threshold
        self.hop = hop
        self.test_ratio = test_ratio
        self.cross_val = cross_val
        self.max_nodes_per_hop = max_nodes_per_hop
        self.balance = balance
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.gene_num = gene_num
        self.avgDegree = avgDegree
        self.all_gene_pairs = all_TF_gene_pairs
        self.TFs_num = TFs_num
        self.test_flag = test_flag


        # if max_num_nodes == 0:
        #     self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        # else:
        #     self.max_num_nodes = max_num_nodes

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        # self.slices：一个切片字典，用于从该对象重构单个示例

    @property
    def raw_dir(self):
        """默认也是self.root/raw"""
        return self.filepath

    @property
    def processed_dir(self):
        """默认是self.root/processed"""
        if self.dataset_type == "dream":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.dataset_name)
        elif self.dataset_type == "singleCell":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.scGRN_type + "/" + self.dataset_name + "/" + self.gene_num )
        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")
    @property
    def raw_file_names(self):
        """"原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self):
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        file_name_parts = [self.dataset_name, self.network_type, self.scGRN_type, "Th" + str(self.threshold), "hop_" + str(self.hop)]
        if self.test_flag:
            file_name_parts.append("Test_mask")
        if self.cross_val:
            file_name_parts.append("Cross_test_" + str(self.test_ratio * 100))
        elif self.test_ratio > 0:
            file_name_parts.append("Cross_train_" + str((1 - self.test_ratio) * 100))
        if self.balance:
            file_name_parts.append("balanced")
        if self.dataset_type == "singleCell":
            file_name_parts.append(self.gene_num + "_avg_" + str(self.avgDegree))
        if self.all_gene_pairs:
            file_name_parts.append("all_TF_gene_pairs")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        file_name_parts.append(".pt")
        return ['_'.join(file_name_parts)]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        """主程序，对原始数据进行处理"""
        if self.dataset_type == "dream":
            trainNet_ori = np.load(os.path.join(
                self.filepath + "goldGRN/goldGRN_{}.csc".format(self.dataset_name)),
                                   allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + "goldGRN/goldExpression_{}.allx".format(self.dataset_name)),
                                 allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                        str(self.threshold))),
                                                                                        allow_pickle=True).tolist()
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + "{}/MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                            str(self.threshold))),
                                                                                            allow_pickle=True).tolist()
                Atrain_agent1 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_dream(allx)
            Atrain_agent0 = trainNet_agent0.copy()  # the observed network


        elif self.dataset_type == "singleCell":
            singleCell_path2file = "{}/{}/{}/".format(self.scGRN_type, self.dataset_name, self.gene_num)
            trainNet_ori = np.load(os.path.join(
                self.filepath + singleCell_path2file +"goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()
                Atrain_agent1 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_singleCell(allx)

            Atrain_agent0 = trainNet_agent0.copy()  # the observed network

        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

        data = []

        if self.all_gene_pairs:
            train_pos, train_neg,  tranin_lable1, tranin_lable2 = uf.sample_all_gene_paris(trainNet_ori, TF_num=self.TFs_num)
        else:
            if self.cross_val:
                _, _, train_pos, train_neg, _, _, tranin_lable1, tranin_lable2 = uf.sample_neg_balance(trainNet_ori,
                                                                                                       test_ratio=self.test_ratio,
                                                                                                       max_train_num=1000000,
                                                                                                       balance=self.balance)
            else:
                train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.sample_neg_balance(trainNet_ori,
                                                                                                       test_ratio=self.test_ratio,
                                                                                                       max_train_num=1000000,
                                                                                                       balance=self.balance)

        if self.network_type == "Pearson":

            if self.test_flag:
                Atrain_agent0[train_pos[0], train_pos[1]] = 0  # mask test links
                Atrain_agent0[train_pos[1], train_pos[0]] = 0  # mask test links
                Atrain_agent0[tranin_lable1[0], tranin_lable1[1]] = 0  # mask test links
                Atrain_agent0[tranin_lable1[1], tranin_lable1[0]] = 0  # mask test links
                Atrain_agent0[tranin_lable2[0], tranin_lable2[1]] = 0  # mask test links
                Atrain_agent0[tranin_lable2[1], tranin_lable2[0]] = 0  # mask test links
            train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_balance(Atrain_agent0,
                                                                                        train_pos,
                                                                                        train_neg,
                                                                                        tranin_lable1,
                                                                                        tranin_lable2,
                                                                                        self.hop,
                                                                                        self.max_nodes_per_hop,
                                                                                        trainAttributes)

            for i in range(len(train_graphs_agent0)):
                feat_x0 = torch.from_numpy(train_graphs_agent0[i].node_features)
                adj_x0 = np.array(nx.adjacency_matrix(train_graphs_agent0[i].graph).todense())
                row_x0, col_x0, _ = ssp.find(adj_x0)
                edge_index0 = torch.tensor([row_x0, col_x0], dtype=torch.long)
                y_x0 = train_graphs_agent0[i].label
                y_x0 = torch.tensor([y_x0], dtype=torch.long)
                mydata = Data(x=feat_x0, edge_index=edge_index0, y=y_x0)
                data.append(mydata)

        elif self.network_type == "MI":

            if self.test_flag:
                Atrain_agent1[train_pos[0], train_pos[1]] = 0  # mask test links
                Atrain_agent1[train_pos[1], train_pos[0]] = 0  # mask test links
                Atrain_agent1[tranin_lable1[0], tranin_lable1[1]] = 0  # mask test links
                Atrain_agent1[tranin_lable1[1], tranin_lable1[0]] = 0  # mask test links
                Atrain_agent1[tranin_lable2[0], tranin_lable2[1]] = 0  # mask test links
                Atrain_agent1[tranin_lable2[1], tranin_lable2[0]] = 0  # mask test links
            train_graphs_agent1, max_n_label_agent0 = uf.extractLinks2subgraphs_balance(Atrain_agent1,
                                                                                        train_pos,
                                                                                        train_neg,
                                                                                        tranin_lable1,
                                                                                        tranin_lable2,
                                                                                        self.hop,
                                                                                        self.max_nodes_per_hop,
                                                                                        trainAttributes)
            for i in range(len(train_graphs_agent1)):
                feat_x1 = torch.from_numpy(train_graphs_agent1[i].node_features)
                adj_x1 = np.array(nx.adjacency_matrix(train_graphs_agent1[i].graph).todense())
                row_x1, col_x1, _ = ssp.find(adj_x1)
                edge_index1 = torch.tensor([row_x1, col_x1], dtype=torch.long)
                y_x1 = train_graphs_agent1[i].label
                y_x1 = torch.tensor([y_x1], dtype=torch.long)
                mydata = Data(x=feat_x1, edge_index=edge_index1, y=y_x1)
                data.append(mydata)
        else:
            raise Exception("The network_type is wrong!!!!! Expected {Pearson or MI}......")
        # self.data, self.slices = self.collate(data)
        self.data = data
        torch.save((self.data), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        return data

class GRUDGrahp(Dataset):
    def __init__(self, root='/root/sdc/guozy/GRDGNN/data', dataset_type='DREAM',
                 network_type="Pearson", scGRN_type="STRING",
                 cross_val_location="last", use_embedding=False, embedding_dim=1, directed2undirected=False,
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None,
                 all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False,
                 gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        self.filepath = root + os.sep + dataset_type + os.sep + "processed/"
        self.filenames = os.listdir(self.filepath)
        self.network_type = network_type
        self.scGRN_type = scGRN_type
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.threshold = threshold
        self.hop = hop
        self.test_ratio = test_ratio
        self.cross_val = cross_val
        self.max_nodes_per_hop = max_nodes_per_hop
        self.balance = balance
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.gene_num = gene_num
        self.avgDegree = avgDegree
        self.all_gene_pairs = all_TF_gene_pairs
        self.TFs_num = TFs_num
        self.test_flag = test_flag
        self.cross_val_location = cross_val_location
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.directed2undirected = directed2undirected

        # if max_num_nodes == 0:
        #     self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        # else:
        #     self.max_num_nodes = max_num_nodes

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        # self.slices：一个切片字典，用于从该对象重构单个示例

    @property
    def raw_dir(self):
        """默认也是self.root/raw"""
        return self.filepath

    @property
    def processed_dir(self):
        """默认是self.root/processed"""
        if self.dataset_type == "DREAM":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.dataset_name)
        elif self.dataset_type == "SingleCell":
            return os.path.join(self.root,
                                self.dataset_type + "/processed/" + self.scGRN_type + "/" + self.dataset_name + "/" + self.gene_num)
        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

    @property
    def raw_file_names(self):
        """"原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self):
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        file_name_parts = [self.dataset_name, self.network_type, self.scGRN_type, "Th" + str(self.threshold),
                           "hop_" + str(self.hop)]
        if self.test_flag:
            file_name_parts.append("Test_mask")
        if self.cross_val:
            if self.test_flag:
                file_name_parts.append("Three_Cross_test_" + self.cross_val_location)
            else:
                file_name_parts.append("Three_Cross_train_" + self.cross_val_location)
        if self.balance:
            file_name_parts.append("balanced")
        if self.dataset_type == "SingleCell":
            file_name_parts.append(self.gene_num + "_avg_" + str(self.avgDegree))
        if self.all_gene_pairs:
            file_name_parts.append("all_TF_gene_pairs")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        if self.use_embedding:
            file_name_parts.append("embedding_dim_" + str(self.embedding_dim))
        if self.directed2undirected:
            file_name_parts.append("directed2undirected")
        file_name_parts.append("binary")
        file_name_parts.append("undirected.pt")
        return ['_'.join(file_name_parts)]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        """主程序，对原始数据进行处理"""
        if self.dataset_type == "DREAM":
            trainNet_ori = np.load(os.path.join(
                self.filepath + "{}/goldGRN_{}.csc".format(self.dataset_name, self.dataset_name)),
                allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + "{}/goldExpression_{}.allx".format(self.dataset_name, self.dataset_name)),
                allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + "{}/Pearson_Threshold_{}_Networks_Order3_TF_undirected.npy".format(self.dataset_name,
                                                                                            str(self.threshold))),
                allow_pickle=True).tolist()
            Atrain_agent0 = trainNet_agent0.copy()  # the observed network

            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + "{}/MutInfo_Threshold_{}_Networks_Order3_TF_undirected.npy".format(self.dataset_name,
                                                                                                str(self.threshold))),
                    allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_dream(allx)


        elif self.dataset_type == "SingleCell":
            singleCell_path2file = "{}/{}/{}/".format(self.scGRN_type, self.dataset_name, self.gene_num)
            trainNet_ori = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF_undirected.npy".format(
                        str(self.avgDegree))), allow_pickle=True).tolist()

            Atrain_agent0 = trainNet_agent0.copy()  # the observed network
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy_undirected.npy".format(
                            str(self.avgDegree))), allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_singleCell(allx)

        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

        data = []

        if self.all_gene_pairs:
            train_pos, train_neg, tranin_lable1, tranin_lable2 = uf.sample_all_gene_paris(trainNet_ori,
                                                                                          TF_num=self.TFs_num)
        else:

            if self.directed2undirected:
                Atrain_agent0 = uf.directed2undirected(Atrain_agent0)
            if self.cross_val:
                self.test_ratio = 0.33  # 3 folder cross_val
                if self.test_flag:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    _, _, train_pos, train_neg = uf.Cross_3_V_sample_neg_binary(trainNet_ori,
                                                                                max_train_num=1000000,
                                                                                Partion=Partition)
                else:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary(trainNet_ori,
                                                                                max_train_num=1000000,
                                                                                Partion=Partition)
            else:
                train_pos, train_neg, _, _ = uf.sample_neg_binary(trainNet_ori,
                                                                  test_ratio=0.0,
                                                                  TF_num=self.TFs_num,
                                                                  max_train_num=1000000)

        if self.use_embedding:
            train_embeddings = uf.generate_node2vec_embeddings(Atrain_agent0, self.embedding_dim, True, train_neg)
            trainAttributes = np.concatenate([train_embeddings, trainAttributes], axis=1)
        if self.test_flag:
            Atrain_agent0[train_pos[0], train_pos[1]] = 0  # mask test links
            Atrain_agent0[train_pos[1], train_pos[0]] = 0  # mask test links
        train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                   train_pos,
                                                                                   train_neg,
                                                                                   self.hop,
                                                                                   self.max_nodes_per_hop,
                                                                                   trainAttributes)

        for i in range(len(train_graphs_agent0)):
            feat_x0 = torch.from_numpy(train_graphs_agent0[i].node_features)
            adj_x0 = np.array(nx.adjacency_matrix(train_graphs_agent0[i].graph).todense())
            row_x0, col_x0, _ = ssp.find(adj_x0)
            edge_index0 = torch.tensor([row_x0, col_x0], dtype=torch.long)
            y_x0 = train_graphs_agent0[i].label
            y_x0 = torch.tensor([y_x0], dtype=torch.long)
            mydata = Data(x=feat_x0, edge_index=edge_index0, y=y_x0)
            data.append(mydata)
        self.data = data
        torch.save((self.data), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        return data

class GRDGrahpUseGeneLinkDataset(Dataset):
    def __init__(self, root='/root/undergraduate/guozydata/expression2GRN', dataset_type='dream', network_type="Pearson", scGRN_type = "STRING",
                 cross_val_location="last", use_embedding=False, embedding_dim=1, directed2undirected=False, useGoldasInput = False,
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None, all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False, gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        self.filepath = root + os.sep + dataset_type + os.sep + "raw/"
        self.filenames = os.listdir(self.filepath)
        self.network_type = network_type
        self.scGRN_type = scGRN_type
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.threshold = threshold
        self.hop = hop
        self.test_ratio = test_ratio
        self.cross_val = cross_val
        self.max_nodes_per_hop = max_nodes_per_hop
        self.balance = balance
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.gene_num = gene_num
        self.avgDegree = avgDegree
        self.all_gene_pairs = all_TF_gene_pairs
        self.TFs_num = TFs_num
        self.test_flag = test_flag
        self.cross_val_location = cross_val_location
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.directed2undirected = directed2undirected
        self.useGoldasInput = useGoldasInput


        # if max_num_nodes == 0:
        #     self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        # else:
        #     self.max_num_nodes = max_num_nodes

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        # self.slices：一个切片字典，用于从该对象重构单个示例

    @property
    def raw_dir(self):
        """默认也是self.root/raw"""
        return self.filepath

    @property
    def processed_dir(self):
        """默认是self.root/processed"""
        if self.dataset_type == "dream":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.dataset_name)
        elif self.dataset_type == "singleCell":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.scGRN_type + "/" + self.dataset_name + "/" + self.gene_num )
        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")
    @property
    def raw_file_names(self):
        """"原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self):
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        file_name_parts = [self.dataset_name, self.network_type, self.scGRN_type, "Th" + str(self.threshold), "hop_" + str(self.hop)]
        if self.test_flag:
            file_name_parts.append("Test_mask")
        if self.cross_val:
            if self.test_flag:
                file_name_parts.append("Three_Cross_test_" + self.cross_val_location)
            else:
                file_name_parts.append("Three_Cross_train_" + self.cross_val_location)
        if self.balance:
            file_name_parts.append("balanced")
        if self.dataset_type == "singleCell":
            file_name_parts.append(self.gene_num + "_avg_" + str(self.avgDegree))
        if self.all_gene_pairs:
            file_name_parts.append("all_TF_gene_pairs")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        if self.use_embedding:
            file_name_parts.append("embedding_dim_" + str(self.embedding_dim))
        if self.directed2undirected:
            file_name_parts.append("directed2undirected")
        file_name_parts.append("binary_useGeneLinkDataset_o")
        if self.useGoldasInput:
            file_name_parts.append("goldNetAsInput")
        file_name_parts.append(".pt")
        return ['_'.join(file_name_parts)]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        """主程序，对原始数据进行处理"""
        if self.dataset_type == "dream":
            trainNet_ori = np.load(os.path.join(
                self.filepath + "goldGRN/goldGRN_{}.csc".format(self.dataset_name)),
                                   allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + "goldGRN/goldExpression_{}.allx".format(self.dataset_name)),
                                 allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                        str(self.threshold))),
                                                                                        allow_pickle=True).tolist()
            Atrain_agent0 = trainNet_agent0.copy()  # the observed network

            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + "{}/MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                            str(self.threshold))),
                                                                                            allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_dream(allx)


        elif self.dataset_type == "singleCell":
            singleCell_path2file = "{}/{}/{}/".format(self.scGRN_type, self.dataset_name, self.gene_num)
            trainNet_ori = np.load(os.path.join(
                self.filepath + singleCell_path2file +"goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()

            Atrain_agent0 = trainNet_agent0.copy()  # the observed network
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_singleCell_all_feat(allx, all_dim=data_dim[self.dataset_name])

        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

        data = []

        if self.all_gene_pairs and self.cross_val:
            if self.test_flag:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                _, _, train_pos, train_neg = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)

            else:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)
        else:
            if self.directed2undirected:
                Atrain_agent0 = uf.directed2undirected(Atrain_agent0)
            if self.cross_val:
                self.test_ratio = 0.33  # 3 folder cross_val
                if self.test_flag:
                    train_pos, train_neg = ([], []), ([], [])
                    test_set_filepath = self.filepath + singleCell_path2file + self.cross_val_location + "_Test_set_o.csv"
                    test_set = pd.read_csv(test_set_filepath)
                    test_set_data = test_set.values
                    test_set_data = test_set_data[:, 1:]
                    for i in range(test_set_data.shape[0]):
                        if test_set_data[i, 2] == 1:
                            train_pos[0].append(test_set_data[i, 0])
                            train_pos[1].append(test_set_data[i, 1])
                        else:
                            train_neg[0].append(test_set_data[i, 0])
                            train_neg[1].append(test_set_data[i, 1])

                    print("The TF--Gene Pairs of test dataset was sampled....")


                else:
                    train_pos, train_neg = ([], []), ([], [])
                    test_set_filepath = self.filepath + singleCell_path2file + self.cross_val_location + "_Train_set_o.csv"
                    test_set = pd.read_csv(test_set_filepath)
                    test_set_data = test_set.values
                    test_set_data = test_set_data[:, 1:]
                    for i in range(test_set_data.shape[0]):
                        if test_set_data[i, 2] == 1:
                            train_pos[0].append(test_set_data[i, 0])
                            train_pos[1].append(test_set_data[i, 1])
                        else:
                            train_neg[0].append(test_set_data[i, 0])
                            train_neg[1].append(test_set_data[i, 1])

                    print("The TF--Gene Pairs of train dataset was sampled....")



            else:
                train_pos, train_neg, _, _ = uf.sample_neg_binary(trainNet_ori,
                                                           test_ratio=0.0,
                                                           TF_num=self.TFs_num,
                                                           max_train_num=1000000)

        if self.use_embedding:
            train_embeddings = uf.generate_node2vec_embeddings(Atrain_agent0, self.embedding_dim, True, train_neg)
            trainAttributes = np.concatenate([train_embeddings, trainAttributes], axis=1)
        if self.test_flag:
            if self.useGoldasInput:
                trainNet_ori[train_pos[0], train_pos[1]] = 0  # mask test links
                trainNet_ori[train_pos[1], train_pos[0]] = 0  # mask test links
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(trainNet_ori,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)

            else:
                Atrain_agent0[train_pos[0], train_pos[1]] = 0  # mask test links
                Atrain_agent0[train_pos[1], train_pos[0]] = 0  # mask test links
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)
        else:
            if self.useGoldasInput:
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(trainNet_ori,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)
            else:
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)

        for i in range(len(train_graphs_agent0)):
            feat_x0 = torch.from_numpy(train_graphs_agent0[i].node_features)
            adj_x0 = np.array(nx.adjacency_matrix(train_graphs_agent0[i].graph).todense())
            row_x0, col_x0, _ = ssp.find(adj_x0)
            edge_index0 = torch.tensor([row_x0, col_x0], dtype=torch.long)
            y_x0 = train_graphs_agent0[i].label
            y_x0 = torch.tensor([y_x0], dtype=torch.long)
            mydata = Data(x=feat_x0, edge_index=edge_index0, y=y_x0)
            data.append(mydata)
        self.data = data
        torch.save((self.data), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        return data

class GRDGrahpUseGeneLinkDataset_test(Dataset):
    def __init__(self, root='/root/undergraduate/guozydata/expression2GRN', dataset_type='dream', network_type="Pearson", scGRN_type = "STRING",
                 cross_val_location="last", use_embedding=False, embedding_dim=1, directed2undirected=False, useGoldasInput = False,
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None, all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False, gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        self.filepath = root + os.sep + dataset_type + os.sep + "raw/"
        self.filenames = os.listdir(self.filepath)
        self.network_type = network_type
        self.scGRN_type = scGRN_type
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.threshold = threshold
        self.hop = hop
        self.test_ratio = test_ratio
        self.cross_val = cross_val
        self.max_nodes_per_hop = max_nodes_per_hop
        self.balance = balance
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.gene_num = gene_num
        self.avgDegree = avgDegree
        self.all_gene_pairs = all_TF_gene_pairs
        self.TFs_num = TFs_num
        self.test_flag = test_flag
        self.cross_val_location = cross_val_location
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.directed2undirected = directed2undirected
        self.useGoldasInput = useGoldasInput


        # if max_num_nodes == 0:
        #     self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        # else:
        #     self.max_num_nodes = max_num_nodes

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        # self.slices：一个切片字典，用于从该对象重构单个示例

    @property
    def raw_dir(self):
        """默认也是self.root/raw"""
        return self.filepath

    @property
    def processed_dir(self):
        """默认是self.root/processed"""
        if self.dataset_type == "dream":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.dataset_name)
        elif self.dataset_type == "singleCell":
            return os.path.join(self.root, self.dataset_type + "/processed/" + self.scGRN_type + "/" + self.dataset_name + "/" + self.gene_num )
        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")
    @property
    def raw_file_names(self):
        """"原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self):
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        file_name_parts = [self.dataset_name, self.network_type, self.scGRN_type, "Th" + str(self.threshold), "hop_" + str(self.hop)]
        if self.test_flag:
            file_name_parts.append("Test_mask")
        if self.cross_val:
            if self.test_flag:
                file_name_parts.append("Three_Cross_test_" + self.cross_val_location)
            else:
                file_name_parts.append("Three_Cross_train_" + self.cross_val_location)
        if self.balance:
            file_name_parts.append("balanced")
        if self.dataset_type == "singleCell":
            file_name_parts.append(self.gene_num + "_avg_" + str(self.avgDegree))
        if self.all_gene_pairs:
            file_name_parts.append("all_TF_gene_pairs")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        if self.use_embedding:
            file_name_parts.append("embedding_dim_" + str(self.embedding_dim))
        if self.directed2undirected:
            file_name_parts.append("directed2undirected")
        file_name_parts.append("binary_useGeneLinkDataset_o")
        if self.useGoldasInput:
            file_name_parts.append("goldNetAsInput")
        file_name_parts.append("test.pt")
        return ['_'.join(file_name_parts)]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        """主程序，对原始数据进行处理"""
        if self.dataset_type == "dream":
            trainNet_ori = np.load(os.path.join(
                self.filepath + "goldGRN/goldGRN_{}.csc".format(self.dataset_name)),
                                   allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + "goldGRN/goldExpression_{}.allx".format(self.dataset_name)),
                                 allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                        str(self.threshold))),
                                                                                        allow_pickle=True).tolist()
            Atrain_agent0 = trainNet_agent0.copy()  # the observed network

            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + "{}/MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(self.dataset_name,
                                                                                            str(self.threshold))),
                                                                                            allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_dream(allx)


        elif self.dataset_type == "singleCell":
            singleCell_path2file = "{}/{}/{}/".format(self.scGRN_type, self.dataset_name, self.gene_num)
            trainNet_ori = np.load(os.path.join(
                self.filepath + singleCell_path2file +"goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                self.filepath + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet_agent0 = np.load(
                os.path.join(
                    self.filepath + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()

            Atrain_agent0 = trainNet_agent0.copy()  # the observed network
            if self.network_type == "MI":
                trainNet_agent1 = np.load(
                    os.path.join(
                        self.filepath + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy".format(str(self.avgDegree))), allow_pickle=True).tolist()
                Atrain_agent0 = trainNet_agent1.copy()
            allx = trainGroup.toarray().astype("float32")
            """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
            trainAttributes = uf.genenet_attribute_singleCell_all_feat_test(allx, all_dim=data_dim[self.dataset_name])

        else:
            raise Exception("The dataset_type is wrong!!!!! Expected {Dream or singleCell}......")

        data = []

        if self.all_gene_pairs and self.cross_val:
            if self.test_flag:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                _, _, train_pos, train_neg = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)

            else:
                Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary_allPairs(trainNet_ori,
                                                                                    max_train_num=1000000,
                                                                                    Partion=Partition,
                                                                                     TF_num=self.TFs_num)
        else:
            if self.directed2undirected:
                Atrain_agent0 = uf.directed2undirected(Atrain_agent0)
            if self.cross_val:
                self.test_ratio = 0.33  # 3 folder cross_val
                if self.test_flag:
                    train_pos, train_neg = ([], []), ([], [])
                    test_set_filepath = self.filepath + singleCell_path2file + self.cross_val_location + "_Test_set_o.csv"
                    test_set = pd.read_csv(test_set_filepath)
                    test_set_data = test_set.values
                    test_set_data = test_set_data[:, 1:]
                    for i in range(test_set_data.shape[0]):
                        if test_set_data[i, 2] == 1:
                            train_pos[0].append(test_set_data[i, 0])
                            train_pos[1].append(test_set_data[i, 1])
                        else:
                            train_neg[0].append(test_set_data[i, 0])
                            train_neg[1].append(test_set_data[i, 1])

                    print("The TF--Gene Pairs of test dataset was sampled....")


                else:
                    train_pos, train_neg = ([], []), ([], [])
                    test_set_filepath = self.filepath + singleCell_path2file + self.cross_val_location + "_Train_set_o.csv"
                    test_set = pd.read_csv(test_set_filepath)
                    test_set_data = test_set.values
                    test_set_data = test_set_data[:, 1:]
                    for i in range(test_set_data.shape[0]):
                        if test_set_data[i, 2] == 1:
                            train_pos[0].append(test_set_data[i, 0])
                            train_pos[1].append(test_set_data[i, 1])
                        else:
                            train_neg[0].append(test_set_data[i, 0])
                            train_neg[1].append(test_set_data[i, 1])

                    print("The TF--Gene Pairs of train dataset was sampled....")



            else:
                train_pos, train_neg, _, _ = uf.sample_neg_binary(trainNet_ori,
                                                           test_ratio=0.0,
                                                           TF_num=self.TFs_num,
                                                           max_train_num=1000000)

        if self.use_embedding:
            train_embeddings = uf.generate_node2vec_embeddings(Atrain_agent0, self.embedding_dim, True, train_neg)
            trainAttributes = np.concatenate([train_embeddings, trainAttributes], axis=1)
        if self.test_flag:
            if self.useGoldasInput:
                trainNet_ori[train_pos[0], train_pos[1]] = 0  # mask test links
                trainNet_ori[train_pos[1], train_pos[0]] = 0  # mask test links
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(trainNet_ori,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)

            else:
                Atrain_agent0[train_pos[0], train_pos[1]] = 0  # mask test links
                Atrain_agent0[train_pos[1], train_pos[0]] = 0  # mask test links
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)
        else:
            if self.useGoldasInput:
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(trainNet_ori,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)
            else:
                train_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs_binary(Atrain_agent0,
                                                                                           train_pos,
                                                                                           train_neg,
                                                                                           self.hop,
                                                                                           self.max_nodes_per_hop,
                                                                                           trainAttributes)

        for i in range(len(train_graphs_agent0)):
            feat_x0 = torch.from_numpy(train_graphs_agent0[i].node_features)
            adj_x0 = np.array(nx.adjacency_matrix(train_graphs_agent0[i].graph).todense())
            row_x0, col_x0, _ = ssp.find(adj_x0)
            edge_index0 = torch.tensor([row_x0, col_x0], dtype=torch.long)
            y_x0 = train_graphs_agent0[i].label
            y_x0 = torch.tensor([y_x0], dtype=torch.long)
            mydata = Data(x=feat_x0, edge_index=edge_index0, y=y_x0)
            data.append(mydata)
        self.data = data
        torch.save((self.data), self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        return data