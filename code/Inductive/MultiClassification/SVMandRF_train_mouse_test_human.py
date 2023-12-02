import torch
import numpy as np
import os.path
import random
import argparse
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from code.ComparedMethods.SVMandRF.SingleCell import units as uf


parser = argparse.ArgumentParser(description='baseline SVM to compare with GRDGNN')
# general settings
parser.add_argument('--traindata-name', default='hESC', help='train network name')
parser.add_argument('--testdata-name', default='hHEP', help='test network name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8,
                    help='pearson correlation as the network')
# parser.add_argument('--pearson_net', type=int, default=3,
#                     help='pearson correlation as the network')
# model settings
parser.add_argument('--hop', default=0, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=10,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
dataset_type = "SingleCell"
scGRN_type = "STRING"
gene_num = "TFs+1000"
location = "pred"
hop = 1
avgDegree = 19
threshold = 0.3
cross_val = False
if dataset_type == "SingleCell":
    scGRN_type = "Non-Specific Dataset"
    train_balance = False
    avgDegree = 19
networkDataset_name = {
    0: "hESC",
    1: "hHEP",
    2: "mDC",
    3: "mESC",
    4: "mHSC-E",
    5: "mHSC-GM",
    6: "mHSC-L"
}
dreamDataset_name = {
    0: "net3",
    1: "net4"
}
dreamTFdict={
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

def loadDreamTraindataset(root_path, traindata_name):

    trainNet_ori = np.load(os.path.join(root_path, 'goldGRN/goldGRN_{}.csc'.format(traindata_name)),
                           allow_pickle=True)
    trainGroup = np.load(os.path.join(root_path, 'goldGRN/goldExpression_{}.allx'.format(traindata_name)),
                         allow_pickle=True)
    trainNet = np.load(
        root_path + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(traindata_name, str(threshold)),
        allow_pickle=True).tolist()
    allx = trainGroup.toarray().astype('float32')
    # deal with the features:
    trainAttributes = uf.SVM_RF_genenet_attribute_Dream(allx)

    train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.sample_neg_balance(trainNet_ori,
                                                                                           test_ratio=0.0,
                                                                                           max_train_num=1000000,
                                                                                           balance=True)

    train_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        else:
            train_node_information = trainAttributes

    '''Train and apply classifier'''
    Atrain = trainNet.copy()  # the observed network

    train_graphs, train_labels = uf.extractLinks2subgraphs_balance_SVM(Atrain,
                                                                       train_pos,
                                                                       train_neg,
                                                                       tranin_lable1,
                                                                       tranin_lable2,
                                                                       h=1,
                                                                       max_nodes_per_hop=args.max_nodes_per_hop,
                                                                       train_node_information=train_node_information)
    return train_graphs, train_labels


def loadDreamTestdataset(root_path, traindata_name):
    trainNet_ori = np.load(os.path.join(root_path, 'goldGRN/goldGRN_{}.csc'.format(traindata_name)),
                           allow_pickle=True)
    trainGroup = np.load(os.path.join(root_path, 'goldGRN/goldExpression_{}.allx'.format(traindata_name)),
                         allow_pickle=True)
    trainNet = np.load(
        root_path + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(traindata_name, str(threshold)),
        allow_pickle=True).tolist()
    allx = trainGroup.toarray().astype('float32')
    # deal with the features:
    trainAttributes = uf.SVM_RF_genenet_attribute_Dream(allx)

    _, _, test_pos, test_neg, _, _, test_lable1, test_lable2 = uf.sample_neg_balance(trainNet_ori,
                                                                                     test_ratio=1.0,
                                                                                     max_train_num=1000000,
                                                                                     balance=False)

    train_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        else:
            train_node_information = trainAttributes

    '''Train and apply classifier'''
    Atest = trainNet.copy()  # the observed network
    Atest[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest[test_pos[1], test_pos[0]] = 0  # mask test links
    Atest[test_lable1[0], test_lable1[1]] = 0  # mask test links
    Atest[test_lable1[1], test_lable1[0]] = 0  # mask test links
    Atest[test_lable2[0], test_lable2[1]] = 0  # mask test links
    Atest[test_lable2[1], test_lable2[0]] = 0  # mask test links
    test_graphs, test_labels = uf.extractLinks2subgraphs_balance_SVM(Atest,
                                                                     test_pos,
                                                                     test_neg,
                                                                     test_lable1,
                                                                     test_lable2,
                                                                     h=1,
                                                                     max_nodes_per_hop=args.max_nodes_per_hop,
                                                                     train_node_information=train_node_information)
    return test_graphs, test_labels

def loadSinglCellTraindata(root_path, traindata_name):

    singleCell_path2file = "{}/{}/{}/".format(scGRN_type, traindata_name, gene_num)
    trainNet_ori = np.load(os.path.join(
        root_path + singleCell_path2file + "goldGRN.csc"), allow_pickle=True)
    trainGroup = np.load(os.path.join(
        root_path + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
    trainNet = np.load(
        os.path.join(
            root_path + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(
                str(avgDegree))), allow_pickle=True).tolist()


    allx = trainGroup.toarray().astype('float32')
    # deal with the features:
    trainAttributes = uf.SVM_RF_genenet_attribute_SingleCell(allx)
    # Prepare Testing




    train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.sample_neg_balance(trainNet_ori,
                                                                                           test_ratio=0.0,
                                                                                           max_train_num=1000000,
                                                                                           balance=False)
    train_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        else:
            train_node_information = trainAttributes

    '''Train and apply classifier'''
    Atrain = trainNet.copy()  # the observed network

    train_graphs, train_labels = uf.extractLinks2subgraphs_balance_SVM(Atrain,
                                                                   train_pos,
                                                                   train_neg,
                                                                   tranin_lable1,
                                                                   tranin_lable2,
                                                                   h=1,
                                                                   max_nodes_per_hop=args.max_nodes_per_hop,
                                                                   train_node_information=train_node_information)
    return train_graphs, train_labels


def loadSinglCellTestdata(root_path, traindata_name):
    singleCell_path2file = "{}/{}/{}/".format(scGRN_type, traindata_name, gene_num)
    trainNet_ori = np.load(os.path.join(
        root_path + singleCell_path2file + "goldGRN.csc"), allow_pickle=True)
    trainGroup = np.load(os.path.join(
        root_path + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
    trainNet = np.load(
        os.path.join(
            root_path + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(
                str(avgDegree))), allow_pickle=True).tolist()

    allx = trainGroup.toarray().astype('float32')
    # deal with the features:
    trainAttributes = uf.SVM_RF_genenet_attribute_SingleCell(allx)
    # Prepare Testing

    _, _, test_pos, test_neg, _, _, test_lable1, test_lable2 = uf.sample_neg_balance(trainNet_ori,
                                                                                     test_ratio=1.0,
                                                                                     max_train_num=1000000,
                                                                                     balance=False)

    train_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        else:
            train_node_information = trainAttributes

    '''Train and apply classifier'''
    Atest = trainNet.copy()  # the observed network
    Atest[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest[test_pos[1], test_pos[0]] = 0  # mask test links
    Atest[test_lable1[0], test_lable1[1]] = 0  # mask test links
    Atest[test_lable1[1], test_lable1[0]] = 0  # mask test links
    Atest[test_lable2[0], test_lable2[1]] = 0  # mask test links
    Atest[test_lable2[1], test_lable2[0]] = 0  # mask test links
    test_graphs, test_labels = uf.extractLinks2subgraphs_balance_SVM(Atest,
                                                                     test_pos,
                                                                     test_neg,
                                                                     test_lable1,
                                                                     test_lable2,
                                                                     h=1,
                                                                     max_nodes_per_hop=args.max_nodes_per_hop,
                                                                     train_node_information=train_node_information)
    return test_graphs, test_labels
# Inductive learning
# For 1vs 1
def main(iter):
    # load human dataset as training
    root_path = uf.get_project_dir("GRDGNN") + "data/SingleCell/processed/"
    # root_path = "/root/undergraduate/guozydata/expression2GRN/dream/raw/"
    for i in range(2, 7):
        temp_train_graphs, temp_train_labels = loadSinglCellTestdata(root_path, networkDataset_name[i])
        if i == 2:
            train_graphs, train_labels = temp_train_graphs, temp_train_labels
        else:
            train_graphs = train_graphs + temp_train_graphs
            train_labels = train_labels + temp_train_labels
    for j in range(2):
        temp_test_graphs, temp_test_labels = loadSinglCellTestdata(root_path, networkDataset_name[j])
        if j == 0:
            test_graphs, test_labels = temp_test_graphs, temp_test_labels
        else:
            test_graphs = test_graphs + temp_test_graphs
            test_labels = test_labels + temp_test_labels

    X = np.asarray(train_graphs)
    y = np.asarray(train_labels)
    testx = np.asarray(test_graphs)
    true_y = np.asarray(test_labels)

    # clf = LinearSVC()
    clf = svm.SVC(gamma='scale')
    # clf = svm.SVC()
    clf.fit(X, y)
    pred = clf.predict(testx)
    y_score = clf.decision_function(testx)
    y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
    print(classification_report(true_y, pred))
    cnf_matrix_enesmb = confusion_matrix(true_y, pred)
    """计算TP，FP，TN，FN"""
    print("==" * 20)
    print(" " * 5 + "*" * 20)
    print(" " * 8 + "【SVM Agent】")
    print(" " * 5 + "*" * 20)
    print("==" * 40)
    uf.TP_FP_TN_FN(cnf_matrix_enesmb, len(true_y), "TP_FP_TN_FN")
    # 计算 acc, Precision_score, Recall_score, F1_score
    print("==" * 40)
    acc, Precision_score, Recall_score, F1_scor = uf.Calculate_ACC_P_R(true_y, pred, "ACC_P_R")
    micro_auroc = roc_auc_score(true_y, y_score, multi_class='ovr', average='micro')
    macro_auroc = roc_auc_score(true_y, y_score, multi_class='ovr', average='macro')
    micro_aupr, macro_aupr = uf.PrPrint(true_y, y_score)
    # uf.writeReslut2File(acc, Precision_score, Recall_score, F1_scor,
    #                     [micro_auroc, macro_auroc],
    #                     [micro_aupr, macro_aupr], dataset_type=dataset_type,
    #                     network_type="SVM", train_dataset="Mouse",
    #                     scGRN_type=scGRN_type, gene_num=gene_num,
    #                     test_dataset="Human", threshold=threshold,
    #                     hop=hop, location=location, cross_val=cross_val,
    #                     experiment_count=iter)
    print("micro AUC: {:.4f}".format(micro_auroc))
    print("macro AUC: {:.4f}".format(macro_auroc))
    print("micro AUPR: {:.4f}".format(micro_aupr))
    print("macro AUPR: {:.4f}".format(macro_aupr))

    # precision, recall, _ = precision_recall_curve(true_y, y_score)
    # # plot no skill
    # plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # # plot the precision-recall curve for the model
    # plt.plot(recall, precision, marker='.')
    # # show the plot
    # #plt.show()
    # plt.savefig('SVM_34_e.png')

    # randomforest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    pred = rf.predict(testx)
    y_score = rf.predict_proba(testx)
    y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
    print(classification_report(true_y, pred))
    cnf_matrix_enesmb = confusion_matrix(true_y, pred)
    """计算TP，FP，TN，FN"""
    print("==" * 20)
    print(" " * 5 + "*" * 20)
    print(" " * 8 + "【RF Agent】")
    print(" " * 5 + "*" * 20)
    print("==" * 40)
    uf.TP_FP_TN_FN(cnf_matrix_enesmb, len(true_y), "TP_FP_TN_FN")
    # 计算 acc, Precision_score, Recall_score, F1_score
    print("==" * 40)
    acc, Precision_score, Recall_score, F1_scor = uf.Calculate_ACC_P_R(true_y, pred, "ACC_P_R")
    micro_auroc = roc_auc_score(true_y, y_score, multi_class='ovr', average='micro')
    macro_auroc = roc_auc_score(true_y, y_score, multi_class='ovr', average='macro')
    micro_aupr, macro_aupr = uf.PrPrint(true_y, y_score)
    uf.writeReslut2File(acc, Precision_score, Recall_score, F1_scor,
                        [micro_auroc, macro_auroc],
                        [micro_aupr, macro_aupr], dataset_type=dataset_type,
                        network_type="RF", train_dataset="Mouse",
                        scGRN_type=scGRN_type, gene_num=gene_num,
                        test_dataset="Human", threshold=threshold,
                        hop=hop, location=location, cross_val=cross_val,
                        experiment_count=iter)
    print("micro AUC: {:.4f}".format(micro_auroc))
    print("macro AUC: {:.4f}".format(macro_auroc))
    print("micro AUPR: {:.4f}".format(micro_aupr))
    print("macro AUPR: {:.4f}".format(macro_aupr))

if __name__ == "__main__":
    for i in range(5):
        main(i)
        print("This is the {} time...".format(str(i)))