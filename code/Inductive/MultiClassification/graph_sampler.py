import os
import scipy.sparse as ssp
import networkx as nx
import numpy as np
import torch
import torch.utils.data
from torch_geometric.data import Dataset, Data
from code.Inductive.MultiClassification.units import cmd_args
from code.Inductive.MultiClassification import units as uf
import tqdm
seed_value = 43
np.random.seed(seed_value)
torch.manual_seed(seed_value)
class GRDGrahp(Dataset):
    def __init__(self, root='/root/sdc/guozy/GRDGNN/data', dataset_type='DREAM', network_type="Pearson", scGRN_type = "STRING", cross_val_location="last",
                 use_embedding=False, embedding_dim=1, undirected=False,
                 dataset_name='net3', threshold=0.01, hop=0, max_nodes_per_hop=10, transform=None, all_TF_gene_pairs=False, test_flag=False,
                 balance=False, pre_transform=None, pre_filter=None, test_ratio=0.0, cross_val=False, gene_num="TFs+1000", avgDegree=3, TFs_num=334):

        self.root = root
        print(self.root)
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
        self.undirected = undirected


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
            if self.balance:
                file_name_parts.append("balanced")
        if cmd_args.feat_dim != 0:
            file_name_parts.append("featdim_" + str(cmd_args.feat_dim))
        if self.use_embedding:
            file_name_parts.append("embedding_dim_" + str(self.embedding_dim))
        if self.undirected:
            file_name_parts.append("input_undirected")
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

        Atrain_agent0 = uf.directed2undirected(Atrain_agent0)
        data = []

        if self.all_gene_pairs:
            train_pos, train_neg,  tranin_lable1, tranin_lable2 = uf.sample_all_gene_pairs(trainNet_ori,
                                                                                           TF_num=self.TFs_num,
                                                                                           balance=self.balance)
        else:
            if self.cross_val:
                self.test_ratio = 0.33  # 3 folder cross_val
                if self.test_flag:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    _, _, train_pos, train_neg, _, _, tranin_lable1, tranin_lable2 = uf.Cross_3_V_sample_neg_balance(
                        trainNet_ori,
                        test_ratio=self.test_ratio,
                        max_train_num=1000000,
                        balance=self.balance,
                        Partion=Partition)
                else:
                    Partition = {'pred': 0, 'mid': 1, 'last': 2}[self.cross_val_location]
                    train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.Cross_3_V_sample_neg_balance(
                        trainNet_ori,
                        test_ratio=self.test_ratio,
                        max_train_num=1000000,
                        balance=self.balance,
                        Partion=Partition)
            else:
                train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.sample_neg_balance(trainNet_ori,
                                                                                                       test_ratio=0.0,
                                                                                                       max_train_num=1000000,
                                                                                                       balance=self.balance)

        if self.use_embedding:
            train_embeddings = uf.generate_node2vec_embeddings(Atrain_agent0, self.embedding_dim, True, train_neg)
            trainAttributes = np.concatenate([train_embeddings, trainAttributes], axis=1)
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
        # self.data, self.slices = self.collate(data)
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
        # print("The max num nodes is:", self.max_num_nodes)
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

