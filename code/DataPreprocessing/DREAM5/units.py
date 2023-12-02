import pandas as pd
import numpy as np
import os
import pickle
import tqdm
import torch
import scipy.sparse as ssp
import argparse
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize, scale
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def calc_MI(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
def SSE_Caculate_Order3_MutIfo(data, save_path):
    print("caculate third order SSEMatrix....\n")
    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data)
    data = data.to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]))
    edata = edata.to(device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]))
    SSE_ij = SSE_ij.to(device)

    for i in tqdm.tqdm(range(data.shape[1])):
        for j in tqdm.tqdm(range(data.shape[1])):
            if i != j:
                A = torch.ones((data.shape[0], 4))
                T = torch.ones((data.shape[0], 1))
                T[:, 0] = data[:, i]
                T = T.to(device)
                for k in range(4):
                    A[:, k] = torch.pow(data[:, j], 3-k)
                A = A.to(device)
                x = torch.mm(A.t(), A)
                x = torch.inverse(x)
                x = torch.mm(x, A.t())
                x = torch.mm(x, T)
                temptesnor = x[0] + x[1] * data[:, j] + x[2] * torch.pow(data[:, j], 2) + x[3] * torch.pow(data[:, j], 3)
                SSE_ij[i, j] = torch.sum(torch.pow(data[:, i] - temptesnor, 2))
                mi = calc_MI(data.cpu().numpy()[:, i], data.cpu().numpy()[:, j])
                mi = torch.tensor(mi)
                mi = mi.to(device)
                edata[i, j] = torch.pow(torch.abs(mi), 4) * torch.pow(torch.exp(-SSE_ij[i, j]), 0.25)
        if i % np.floor(data.shape[1] / 100) == 0:
            print("The completed rate is :  " + str(i / data.shape[1] * 100) + "%")
    # 归一化：
    np.save(save_path  + 'MutInfo_SSE_Order3.npy', SSE_ij.cpu().numpy)
    np.savetxt(save_path + "MutInfo_SSE_Order3.csv", SSE_ij.cpu().numpy(), delimiter=",")
    Wik = torch.zeros((edata.shape[1], edata.shape[1]))
    Wik = Wik.to(device)
    for i in range(edata.shape[1]):
        temp = torch.pow(torch.sum(torch.multiply(edata[i, :], edata[i, :]), axis=0), 0.5)
        for j in range(edata.shape[1]):
            if temp != 0:
                Wik[i, j] = edata[i, j] / temp
    np.save(save_path +  'MutInfo_Wik_Order3.npy', Wik.cpu().numpy())
    np.savetxt(save_path + "MutInfo_Wik_Order3.csv", Wik.cpu().numpy(), delimiter=",")
    print("The SSE processing is over!!!!")
    return SSE_ij.cpu().numpy(), Wik.cpu().numpy()
def Normalization_MutIfo(edata, save_path):
    Wik = np.zeros((edata.shape[1], edata.shape[1]))
    for i in range(edata.shape[1]):
        temp = np.power(np.sum(np.multiply(edata[i, :], edata[i, :]), axis=0), 0.5)
        for j in range(edata.shape[1]):
            if temp != 0:
                Wik[i, j] = edata[i, j] / temp
        if i % int(edata.shape[0] / 100) == 0:
            print("The Completed rate is： " + str(i / (edata.shape[0] / 100)) + '%')
    np.save(save_path + 'MutIfo_Wik_Order3.npy', Wik)
    np.savetxt(save_path + "MutIfo_Wik_Order3.csv", Wik, delimiter=",")
    print("Normaliation over!!!!!!!!!!!!!")
    return Wik

def SSE_Caculate_Order3_Pearson(data, save_path):
    print("caculate third order SSEMatrix....\n")
    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data)
    data = data.to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]))
    edata = edata.to(device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]))
    SSE_ij = SSE_ij.to(device)

    for i in tqdm.tqdm(range(data.shape[1])):
        for j in tqdm.tqdm(range(data.shape[1])):
            if i != j:
                A = torch.ones((data.shape[0], 4))
                T = torch.ones((data.shape[0], 1))
                T[:, 0] = data[:, i]
                T = T.to(device)
                for k in range(4):
                    A[:, k] = torch.pow(data[:, j], 3-k)
                A = A.to(device)
                x = torch.mm(A.t(), A)
                x = torch.inverse(x)
                x = torch.mm(x, A.t())
                x = torch.mm(x, T)
                temptesnor = x[0] + x[1] * data[:, j] + x[2] * torch.pow(data[:, j], 2) + x[3] * torch.pow(data[:, j], 3)
                SSE_ij[i, j] = torch.sum(torch.pow(data[:, i] - temptesnor, 2))
                corr, _ = pearsonr(data.cpu()[:, i], data.cpu()[:, j])
                corr = torch.tensor(corr)
                corr = corr.to(device)
                edata[i, j] = torch.pow(torch.abs(corr), 4) * torch.pow(torch.exp(-SSE_ij[i, j]), 0.25)
        if i % np.floor(data.shape[1] / 100) == 0:
            print("The completed rate is :  " + str(i / data.shape[1] * 100) + "%")
    # 归一化：
    np.save(save_path  + 'Pearson_SSE_Order3.npy', SSE_ij.cpu().numpy)
    np.savetxt(save_path + "Pearson_SSE_Order3.csv", SSE_ij.cpu().numpy(), delimiter=",")
    Wik = torch.zeros((edata.shape[1], edata.shape[1]))
    Wik = Wik.to(device)
    for i in range(edata.shape[1]):
        temp = torch.pow(torch.sum(torch.multiply(edata[i, :], edata[i, :]), axis=0), 0.5)
        for j in range(edata.shape[1]):
            if temp != 0:
                Wik[i, j] = edata[i, j] / temp
    np.save(save_path +  'Pearson_Wik_Order3.npy', Wik.cpu().numpy())
    np.savetxt(save_path + "Pearson_Wik_Order3.csv", Wik.cpu().numpy(), delimiter=",")
    print("The SSE processing is over!!!!")
    return SSE_ij.cpu().numpy(), Wik.cpu().numpy()

def Normalization_Pearson(edata, save_path):
    Wik = np.zeros((edata.shape[1], edata.shape[1]))
    for i in range(edata.shape[1]):
        temp = np.power(np.sum(np.multiply(edata[i, :], edata[i, :]), axis=0), 0.5)
        for j in range(edata.shape[1]):
            if temp != 0:
                Wik[i, j] = edata[i, j] / temp
        if i % int(edata.shape[0] / 100) == 0:
            print("The Completed rate is： " + str(i / (edata.shape[0] / 100)) + '%')
    np.save(save_path + 'Pearson_Wik_Order3.npy', Wik)
    np.savetxt(save_path + "Pearson_Wik_Order3.csv", Wik, delimiter=",")
    print("Normaliation over!!!!!!!!!!!!!")
    return Wik
def BuildNetworksByThreshould(wik, Threshold = 0.8):
    networks = np.zeros((wik.shape[0], wik.shape[1]))
    for i in range(wik.shape[0]):
        for j in range(wik.shape[1]):
            if i != j and abs(wik[i, j]) > Threshold:
                networks[i, j] = 1
    networks = ssp.csc_matrix(networks)
    return networks

def BuildNetworksByThreshould_undirected(wik, Threshold = 0.8):
    networks = np.zeros((wik.shape[0], wik.shape[1]))
    for i in range(wik.shape[0]):
        for j in range(wik.shape[1]):
            if i != j and abs(wik[i, j]) > Threshold:
                networks[i, j] = 1
                networks[j, i] = 1
    networks = ssp.csc_matrix(networks)
    return networks


def load_eprData(filename, rownum, colnum, outdim=256):
    """
    :param filename: the file name of Expression data
    :param rownum: the number of the row
    :param colnum: the number of the col
    :outdim: the dim of processed Expression data, outdim < rownum
    :return: simply process to the Expression data
    """
    if outdim > rownum:
        raise Exception("The outdim > rownum, {} > {} , cannot get the {} dim expression data.....".format(str(outdim),
                                                                                                           str(rownum),
                                                                                                           str(outdim)))
    data = np.zeros((rownum, colnum))
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                words = line.split()
                ncount = 0
                for word in words:
                    data[count, ncount] = float(word)
                    ncount = ncount + 1
            count = count + 1
        f.close()
    data = data.T
    data = scale(data)
    pca = PCA(n_components=outdim)
    data = pca.fit_transform(data)
    return data

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

def detect_TF_Gene_Pairs(gold_net, noisy_net):
    undirected_daj = gold_net.todense()
    row, col, _ = ssp.find(gold_net)
    for i in range(len(row)):
        undirected_daj[row[i], col[i]] = 1
        undirected_daj[col[i], row[i]] = 1
    net_triu = ssp.csc_matrix(undirected_daj)
    net_triu = ssp.triu(net_triu, k=1)
    # sample positive links for train/test
    gold_row, gold_col, _ = ssp.find(net_triu)

    undirected_daj_noisy = gold_net.todense()
    row, col, _ = ssp.find(noisy_net)
    for i in range(len(row)):
        undirected_daj_noisy[row[i], col[i]] = 1
        undirected_daj_noisy[col[i], row[i]] = 1
    net_triu_noisy = ssp.csc_matrix(undirected_daj_noisy)
    net_triu_noisy = ssp.triu(net_triu_noisy, k=1)
    # sample positive links for train/test
    noisy_row, noisy_col, _  = ssp.find(net_triu_noisy)
    count = 0
    for i in range(len(noisy_row)):
        for j in range(len(gold_row)):
            if gold_row[j] == noisy_row[i] and gold_col[j] == noisy_col[i]:
                count += 1
    print("The gold TF-Gene Pairs ars: ", len(gold_row))
    print("The noisy TF-Gene Pairs Pairs are: ", len(noisy_row))
    print("The number of gold TF-Gene Pairs in noisy are:  ", count)
    if count >= len(gold_row):
        print("The noisy network includes all the pairs......")
    else:
        print("There are {} pairs not in noisy network....".format(str(len(gold_row - count))))