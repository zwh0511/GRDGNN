import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
#import cPickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from code.ComparedMethods.SVMandRF.SingleCell import units as uf


parser = argparse.ArgumentParser(description='baseline SVM to compare with GRGNN')
# general settings
parser.add_argument('--traindata-name', default='net3', help='train network name')
parser.add_argument('--testdata-name', default='net3', help='test network name')
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


dreamTFdict = {
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

nums_gene = {
    0: "TFs+500",
    1: "TFs+1000"
}

goldNetwork_type = {
    0: "Lofgof Dataset",
    1: "Non-Specific Dataset",
    2: "Specific Dataset",
    3: "STRING Dataset"
}

networkDataset_name = {
    0: "net3",
    1: "net4",
}
location = {
    0: "pred",
    1: "mid",
    2: "last"
}

def main(iter=0, location = "pred", dataset_type="SingleCell", threshold=0.3, avgDegree=19, hop=1, scGRN_type = "Non-Specific Dataset",
         traindata_name="hESC",train_balance=False, test_balance=False, gene_num="TFs+1000", cross_val=True):

    if cross_val:
        test_dataset = traindata_name
    if dataset_type == "SingleCell":
        root_path = uf.get_project_dir("GRDGNN") + "data" + "/SingleCell/processed/"
        traindata_name = traindata_name
        testdata_name = traindata_name
    else:
        root_path = uf.get_project_dir("GRDGNN") + "data" + "/DREAM/processed/"
        traindata_name = traindata_name
        testdata_name = traindata_name

    if traindata_name is not None:
        # Select data name
        if dataset_type == "DREAM":
            trdata_name = traindata_name
            tedata_name = traindata_name

            # Prepare Training
            trainNet_ori = np.load(os.path.join(root_path, '{}/goldGRN_{}.csc'.format(traindata_name, traindata_name)),
                                   allow_pickle=True)
            trainGroup = np.load(os.path.join(root_path, '{}/goldExpression_{}.allx'.format(traindata_name, traindata_name)),
                                 allow_pickle=True)
            trainNet = np.load(
                root_path + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(traindata_name, str(threshold)),
                allow_pickle=True).tolist()
            # trainNet = np.load(root_path + '{}/' + "MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(str(threshold)),
            #                    allow_pickle=True).tolist()
            allx = trainGroup.toarray().astype('float32')
            # deal with the features:
            trainAttributes = uf.SVM_RF_genenet_attribute_Dream(allx)

            # Prepare Testing
            testNet_ori = np.load(os.path.join(root_path, '{}/goldGRN_{}.csc'.format(testdata_name, testdata_name)),
                                  allow_pickle=True)
            testGroup = np.load(os.path.join(root_path, '{}/goldExpression_{}.allx'.format(testdata_name, testdata_name)),
                                allow_pickle=True)
            testNet = np.load(
                root_path + "{}/Pearson_Threshold_{}_Networks_Order3_TF.npy".format(testdata_name, str(threshold)),
                allow_pickle=True).tolist()
            # testNet = np.load(root_path + '{}/' + "MutInfo_Threshold_{}_Networks_Order3_TF.npy".format(str(threshold)),
            #                    allow_pickle=True).tolist()

            allxt = testGroup.toarray().astype('float32')
            # deal with the features:
            testAttributes = uf.SVM_RF_genenet_attribute_Dream(allxt)
        else:
            # Prepare Training
            singleCell_path2file = "{}/{}/{}/".format(scGRN_type, traindata_name, gene_num)
            trainNet_ori = np.load(os.path.join(
                root_path + singleCell_path2file + "goldGRN.csc"), allow_pickle=True)
            trainGroup = np.load(os.path.join(
                root_path + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            trainNet = np.load(
                os.path.join(
                    root_path + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(
                        str(avgDegree))), allow_pickle=True).tolist()
            # trainNet = np.load(
            #     os.path.join(
            #         root_path + singleCell_path2file + "MutIfo_avgDegree_{}_Networks_Order3_TF.npy".format(
            #             str(avgDegree))), allow_pickle=True).tolist()

            allx = trainGroup.toarray().astype('float32')
            # deal with the features:
            trainAttributes = uf.SVM_RF_genenet_attribute(allx, dreamTFdict[traindata_name])
            # Prepare Testing

            singleCell_path2file = "{}/{}/{}/".format(scGRN_type, test_dataset, gene_num)
            testNet_ori = np.load(os.path.join(
                root_path + singleCell_path2file + "goldGRN.csc"), allow_pickle=True)
            testGroup = np.load(os.path.join(
                root_path + singleCell_path2file + "goldExpression.allx"), allow_pickle=True)
            testNet = np.load(
                os.path.join(
                    root_path + singleCell_path2file + "Pearson_avgDegree_{}_Networks_Order3_TF.npy".format(
                        str(avgDegree))), allow_pickle=True).tolist()

            allxt = testGroup.toarray().astype('float32')
            # deal with the features:
            testAttributes = uf.SVM_RF_genenet_attribute(allxt, dreamTFdict[test_dataset])

        if cross_val:
            Partition = {'pred': 0, 'mid': 1, 'last': 2}[location]
            train_pos, train_neg, _, _ = uf.Cross_3_V_sample_neg_binary(trainNet_ori,
                                                                         max_train_num=1000000,
                                                                         Partion=Partition)
            _, _, test_pos, test_neg = uf.Cross_3_V_sample_neg_binary(testNet_ori,
                                                                        max_train_num=1000000,
                                                                        Partion=Partition)
        else:

            train_pos, train_neg, _, _ = uf.sample_neg_binary(trainNet_ori,
                                                           test_ratio=0.0,
                                                           max_train_num=1000000)

            _, _, test_pos, test_neg = uf.sample_neg_binary(testNet_ori,
                                                             test_ratio=1.0,
                                                             max_train_num=1000000)
    train_node_information = None
    test_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
            test_node_information = np.concatenate([test_node_information, testAttributes], axis=1)
        else:
            train_node_information = trainAttributes
            test_node_information = testAttributes

    '''Train and apply classifier'''
    Atrain = trainNet.copy()  # the observed network
    Atest = testNet.copy()  # the observed network
    Atest[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest[test_pos[1], test_pos[0]] = 0  # mask test links

    train_pos, train_neg, _, _, tranin_lable1, tranin_lable2, _, _ = uf.Cross_3_V_sample_neg_balance(trainNet_ori,
                                                                                                     test_ratio=0.33,
                                                                                                     max_train_num=1000000,
                                                                                                     balance=True,
                                                                                                     Partion=2)

    _, _, test_pos, test_neg, _, _, test_lable1, test_lable2 = uf.Cross_3_V_sample_neg_balance(testNet_ori,
                                                                                               test_ratio=0.33,
                                                                                               max_train_num=1000000,
                                                                                               balance=False,
                                                                                               Partion=2)


    train_node_information = None
    test_node_information = None
    # if args.use_embedding:
    #     train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    #     train_node_information = train_embeddings
    #     test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    #     test_node_information = test_embeddings
    if args.use_attribute and trainAttributes is not None:
        if train_node_information is not None:
            train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
            test_node_information = np.concatenate([test_node_information, testAttributes], axis=1)
        else:
            train_node_information = trainAttributes
            test_node_information = testAttributes

    '''Train and apply classifier'''
    Atrain = trainNet.copy()  # the observed network
    Atest = testNet.copy()  # the observed network
    Atest[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest[test_pos[1], test_pos[0]] = 0  # mask test links
    Atest[test_lable1[0], test_lable1[1]] = 0  # mask test links
    Atest[test_lable1[1], test_lable1[0]] = 0  # mask test links
    Atest[test_lable2[0], test_lable2[1]] = 0  # mask test links
    Atest[test_lable2[1], test_lable2[0]] = 0  # mask test links

    train_graphs, train_labels = uf.extractLinks2subgraphs_balance_SVM(Atrain,
                                                                       train_pos,
                                                                       train_neg,
                                                                       tranin_lable1,
                                                                       tranin_lable2,
                                                                       h=1,
                                                                       max_nodes_per_hop=args.max_nodes_per_hop,
                                                                       train_node_information=train_node_information)
    test_graphs, test_labels = uf.extractLinks2subgraphs_balance_SVM(Atest,
                                                                     test_pos,
                                                                     test_neg,
                                                                     test_lable1,
                                                                     test_lable2,
                                                                     h=1,
                                                                     max_nodes_per_hop=args.max_nodes_per_hop,
                                                                     train_node_information=test_node_information)

    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

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
    # uf.writeReslut2File_dream(acc, Precision_score, Recall_score, F1_scor, [micro_auroc, macro_auroc],
    #                           [micro_aupr, macro_aupr], dataset_type=dataset_type, network_type="MultiClassSVM",
    #                           test_dataset=args.traindata_name, threshold=0.3, hop=1, location=location, cross_val=True,
    #                           experiment_count=iter)
    # print("micro AUC: {:.4f}".format(micro_auroc))
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
    # uf.writeReslut2File_dream(acc, Precision_score, Recall_score, F1_scor, [micro_auroc, macro_auroc],
    #                           [micro_aupr, macro_aupr], dataset_type=dataset_type, network_type="MultiClassRF",
    #                           test_dataset=args.traindata_name, threshold=0.3, hop=1, location=location, cross_val=True,
    #                           experiment_count=iter)
    print("micro AUC: {:.4f}".format(micro_auroc))
    print("macro AUC: {:.4f}".format(macro_auroc))
    print("micro AUPR: {:.4f}".format(micro_aupr))
    print("macro AUPR: {:.4f}".format(macro_aupr))

if __name__ == "__main__":
    for i in range(0, 2):
        for k in range(1, 2):
            for m in range(3):
                for n in range(5):
                    main(iter=n, location=location[m],
                         dataset_type="DREAM",
                         threshold=0.3, avgDegree=19,
                         hop=1, traindata_name=networkDataset_name[i],
                         train_balance=False, test_balance=False,
                         gene_num=nums_gene[k], cross_val=True)
                print("This is the {} time...".format(str(i)))