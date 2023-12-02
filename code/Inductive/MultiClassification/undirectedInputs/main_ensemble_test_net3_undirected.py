import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os
import random
import torch.nn as nn
import torch
import numpy as np
import time
from torch.autograd import Variable
from code.Inductive.MultiClassification import units as uf
from code.Inductive.MultiClassification.graph_sampler import GRDGrahp, DGraphSampler
from torch.utils.data import DataLoader
from code.Inductive.MultiClassification.DGCNmodule import DGCNConv, SoftPoolingGcnEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置随机种子
seed_value = 43
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
"""parameters setting......."""

TFDict = {
    "net1": 195,
    "net2": 99,
    "net3": 334,
    "net4": 333,
    "hESC": 410,
    "hHEP": 448,
    "mDC": 321,
    "mESC": 620,
    "mHSC-GM": 132,
    "mHSC-L": 60,
    "mHSC-E": 204
}

train_dataset = "net4"
test_dataset = "net3"
dataset_type = "DREAM"
scGRN_type = "STRING"
gene_num = "TFs+1000"
location = "pred"
hop = 1
avgDegree = 19
threshold = 0.3
cross_val = False
train_balance = True
test_balance = False
undirected = True
if dataset_type == "SingleCell":
    scGRN_type = "Non-Specific Dataset"
    train_balance = False
    avgDegree = 19
if cross_val:
    test_dataset = train_dataset


def LoadData(args):
    root_path = uf.get_project_dir("GRDGNN") + "data"

    Dream_Pear_train_P = GRDGrahp(root=root_path, dataset_type=dataset_type, network_type="Pearson", scGRN_type=scGRN_type,
                                  cross_val_location=location, undirected=undirected,
                                  dataset_name=train_dataset, threshold=threshold, hop=hop, max_nodes_per_hop=10,
                                  all_TF_gene_pairs=False, test_flag=False,
                                  balance=train_balance, cross_val=cross_val, gene_num=gene_num, avgDegree=avgDegree,
                                  TFs_num=TFDict[train_dataset])

    Dream_Pear_test_P = GRDGrahp(root=root_path, dataset_type=dataset_type, network_type="Pearson", scGRN_type=scGRN_type,
                                 cross_val_location=location, undirected=undirected,
                                 dataset_name=test_dataset, threshold=threshold, hop=hop, max_nodes_per_hop=10,
                                 all_TF_gene_pairs=False, test_flag=True,
                                 balance=test_balance, cross_val=cross_val, gene_num=gene_num, avgDegree=avgDegree,
                                 TFs_num=TFDict[test_dataset])

    Dream_Pear_train_MI = GRDGrahp(root=root_path, dataset_type=dataset_type, network_type="MI", scGRN_type=scGRN_type,
                                   cross_val_location=location, undirected=undirected,
                                   dataset_name=train_dataset, threshold=threshold, hop=hop, max_nodes_per_hop=10,
                                   all_TF_gene_pairs=False, test_flag=False,
                                   balance=train_balance, cross_val=cross_val, gene_num=gene_num, avgDegree=avgDegree,
                                   TFs_num=TFDict[train_dataset])

    Dream_Pear_test_MI = GRDGrahp(root=root_path, dataset_type=dataset_type, network_type="MI", scGRN_type=scGRN_type,
                                  cross_val_location=location, undirected=undirected,
                                  dataset_name=test_dataset, threshold=threshold, hop=hop, max_nodes_per_hop=10,
                                  all_TF_gene_pairs=False, test_flag=True,
                                  balance=test_balance, cross_val=cross_val, gene_num=gene_num, avgDegree=avgDegree,
                                  TFs_num=TFDict[test_dataset])

    train_P = Dream_Pear_train_P
    test_P = Dream_Pear_test_P
    train_M = Dream_Pear_train_MI
    test_M = Dream_Pear_test_MI

    train_agent_sampler_P = DGraphSampler(train_P)
    test_agent_sampler_P = DGraphSampler(test_P)
    train_agent_sampler_M = DGraphSampler(train_M)
    test_agent_sampler_M = DGraphSampler(test_M)

    #   打乱pearson和Mutual information 数据集中的样本顺序打乱，并保持一致

    test_combined_dataset = list(zip(test_agent_sampler_P, test_agent_sampler_M))
    random.shuffle(test_combined_dataset)
    test_agent_sampler_P_Shuffle, test_agent_sampler_M_Shuffle = zip(*test_combined_dataset)

    train_agent_loader_P = DataLoader(train_agent_sampler_P, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_agent_loader_M = DataLoader(train_agent_sampler_M, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_agent_loader_P = DataLoader(test_agent_sampler_P_Shuffle, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0)
    test_agent_loader_M = DataLoader(test_agent_sampler_M_Shuffle, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0)

    return train_agent_loader_P, test_agent_loader_P, test_agent_sampler_P.max_num_nodes, test_agent_sampler_P.feat_dim, test_agent_sampler_P.assign_feat_dim, \
        train_agent_loader_M, test_agent_loader_M, test_agent_sampler_M.max_num_nodes, test_agent_sampler_M.feat_dim, test_agent_sampler_M.assign_feat_dim


def train(dataset, model, args, test_dataset=None, mask_nodes=True):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    for epoch in range(args.num_epochs):
        total_time = 0
        total_loss = []
        model.train()
        loop = tqdm.tqdm((enumerate(dataset)), total=len(dataset))
        for batch_idx, data in loop:
            begin_time = time.time()
            model.zero_grad()
            adj_in = Variable(data['adj_in'].float(), requires_grad=False).to(device)
            adj_out = Variable(data['adj_out'].float(), requires_grad=False).to(device)
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)

            logits, loss, acc, _ = model(h0, adj_in, adj_out, label, batch_num_nodes, assign_x=assign_input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            loss = loss.data.cpu().detach().numpy()
            elapsed = time.time() - begin_time
            total_time += elapsed
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_description('loss: %0.5f\tacc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(label))
        total_loss = np.array(total_loss)
        avg_loss = np.sum(total_loss, 0) / (batch_idx * args.batch_size + len(label))
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f \033[0m' % (
            epoch, avg_loss[0], avg_loss[1]))

        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            total_loss = []
            total_labels = []
            total_logits = []
            model.eval()
            loop = tqdm.tqdm((enumerate(test_dataset)), total=len(test_dataset))
            for batch_idx, data in loop:
                begin_time = time.time()
                adj_in = Variable(data['adj_in'].float(), requires_grad=False).to(device)
                adj_out = Variable(data['adj_out'].float(), requires_grad=False).to(device)
                h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
                assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)
                logits, loss, acc, _ = model(h0, adj_in, adj_out, label, batch_num_nodes, assign_x=assign_input)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                iter += 1
                loss = loss.data.cpu().detach().numpy()
                elapsed = time.time() - begin_time
                total_time += elapsed
                loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
                loop.set_description('loss: %0.5f\tacc: %0.5f' % (loss, acc))
                total_loss.append(np.array([loss, acc]) * len(label))
                total_labels.extend(label.cpu().detach())
                total_logits.append(logits.cpu().detach())
            total_loss = np.array(total_loss)
            avg_loss = np.sum(total_loss, 0) / (batch_idx * args.batch_size + len(label))
            print('\033[93maverage testing of epoch %d: loss %.5f acc %.5f \033[0m' % (
                epoch, avg_loss[0], avg_loss[1]))
    y_labels = torch.cat(total_labels).cpu().numpy()
    prob_scores = torch.cat(total_logits).cpu().numpy()
    pred = np.argmax(prob_scores, axis=1)

    return model, y_labels, pred, prob_scores


def main(iter):
    args = uf.arg_parse()

    """Load Pearson and Mutual information data"""
    train_Pearson, test_Pearson, max_num_nodes_Pearson, input_dim_Pearson, assign_input_dim_Pearson, \
        train_MI, test_MI, max_num_nodes_MI, input_dim_MI, assign_input_dim_MI = LoadData(args)

    """create pearson and mutual information model"""
    model_P = SoftPoolingGcnEncoder(
        max_num_nodes_Pearson,
        input_dim_Pearson, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim_Pearson).to(device)

    """MutualInformation"""

    model_M = SoftPoolingGcnEncoder(
        max_num_nodes_MI,
        input_dim_MI, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim_MI).to(device)

    """Training and testing"""

    trained_P, y_labels_P, pred_P, prob_scores_P = train(train_Pearson, model_P, args, test_dataset=test_Pearson,
                                                         mask_nodes=True)
    trained_M, y_labels_M, pred_M, prob_scores_M = train(train_MI, model_M, args, test_dataset=test_MI, mask_nodes=True)

    """Ensemble from prob_scores_P and prob_scores_M"""

    prob_scores_Ensem = 0.5 * prob_scores_P + 0.5 * prob_scores_M
    pred_Ensem = np.argmax(prob_scores_Ensem, axis=1)
    print(all(x == y for x, y in zip(y_labels_P, y_labels_M)))
    if all(x == y for x, y in zip(y_labels_P, y_labels_M)):
        y_labels_Ensem = y_labels_M

    """计算TP，FP，TN，FN"""
    print("==" * 20)
    print(" " * 5 + "*" * 20)
    print(" " * 8 + "【Ensemble Agent】")
    print(" " * 5 + "*" * 20)
    cnf_matrix_enesmb = confusion_matrix(y_labels_Ensem, pred_Ensem)
    print("==" * 40)
    uf.TP_FP_TN_FN(cnf_matrix_enesmb, len(y_labels_Ensem), "TP_FP_TN_FN")
    # 计算 acc, Precision_score, Recall_score, F1_score
    print("==" * 40)
    acc, Precision_score, Recall_score, F1_scor = uf.Calculate_ACC_P_R(y_labels_Ensem, pred_Ensem, "ACC_P_R")
    micro_auroc = roc_auc_score(y_labels_Ensem, prob_scores_Ensem, multi_class='ovr', average='micro')
    macro_auroc = roc_auc_score(y_labels_Ensem, prob_scores_Ensem, multi_class='ovr', average='macro')
    micro_aupr, macro_aupr = uf.PrPrint(y_labels_Ensem, prob_scores_Ensem)

    uf.writeReslut2File(acc, Precision_score, Recall_score, F1_scor,
                        [micro_auroc, macro_auroc],
                        [micro_aupr, macro_aupr], dataset_type=dataset_type,
                        network_type="Ensem_undirected", train_dataset=train_dataset,
                        scGRN_type=scGRN_type, gene_num=gene_num,
                        test_dataset=test_dataset, threshold=threshold,
                        hop=hop, location=location, cross_val=cross_val,
                        experiment_count=iter)


if __name__ == "__main__":
    for i in range(5):
        main(i)
        print("The {} time overed.....".format(str(i)))