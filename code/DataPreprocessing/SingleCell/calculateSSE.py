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


def SSE_Caculate_Order3_Pearson(data, save_path, num_TF=334):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Calculating third-order SSEMatrix....\n")

    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data).to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]), device=device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]), device=device)

    for i in tqdm.tqdm(range(num_TF)):
        for j in range(data.shape[1]):
            if i != j:
                xs = data[:, i]
                ys = data[:, j]
                # 检查xs和ys是否包含无效值
                if torch.any(torch.isinf(xs)) or torch.any(torch.isnan(xs)) or torch.any(torch.isinf(ys)) or torch.any(
                        torch.isnan(ys)):
                    edata[i, j] = torch.tensor(0)
                else:
                    sum_xs = torch.sum(xs)
                    sum_xs_sq = torch.sum(xs ** 2)
                    sum_xs_cube = torch.sum(xs ** 3)
                    X = torch.tensor([
                        [data.shape[0], sum_xs, sum_xs_sq, sum_xs_cube],
                        [sum_xs, sum_xs_sq, sum_xs_cube, torch.sum(xs ** 4)],
                        [sum_xs_sq, sum_xs_cube, torch.sum(xs ** 4), torch.sum(xs ** 5)],
                        [sum_xs_cube, torch.sum(xs ** 4), torch.sum(xs ** 5), torch.sum(xs ** 6)]
                    ], device=device)
                    Y = torch.tensor(
                        [torch.sum(ys), torch.sum(xs * ys), torch.sum(xs ** 2 * ys), torch.sum(xs ** 3 * ys)],
                        device=device)
                    w_hat = torch.linalg.solve(X, Y)
                    y_hat = w_hat[0] + w_hat[1] * xs + w_hat[2] * xs ** 2 + w_hat[3] * xs ** 3
                    SSE_ij[i, j] = torch.sum((ys - y_hat) ** 2)
                    corr, _ = pearsonr(xs.cpu(), ys.cpu())
                    corr = torch.tensor(corr, device=device)
                    edata[i, j] = torch.pow(torch.abs(corr), 4) * torch.pow(torch.exp(-SSE_ij[i, j]), 0.25)
    # 归一化：
    SSE_ij_np = SSE_ij.cpu().numpy()
    np.save(save_path + 'Pearson_SSE_Order3_TF.npy', SSE_ij_np)
    np.savetxt(save_path + "Pearson_SSE_Order3_TF.csv", SSE_ij_np, delimiter=",")

    Wik = torch.zeros((data.shape[1], edata.shape[1]), device=device)
    for i in range(edata.shape[1]):
        temp = torch.pow(torch.sum(edata[:, i] ** 2), 0.5)
        for j in range(num_TF):
            if temp > 0:
                Wik[j, i] = edata[j, i] / temp
            else:
                break

    Wik_np = Wik.cpu().numpy()
    np.save(save_path + 'Pearson_Wik_Order3_TF.npy', Wik_np)
    np.savetxt(save_path + "Pearson_Wik_Order3_TF.csv", Wik_np, delimiter=",")

    print("The SSE processing is over!!!!")
    return SSE_ij_np, Wik_np

def SSE_Caculate_Order3_Pearson_undirected(data, save_path, num_TF=334):
    device = torch.device("cpu")
    print("Calculating third-order SSEMatrix....\n")

    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data).to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]), device=device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]), device=device)

    for i in tqdm.tqdm(range(num_TF)):
        for j in range(data.shape[1]):
            if i != j:
                xs = data[:, i]
                ys = data[:, j]
                corr, _ = pearsonr(xs.cpu(), ys.cpu())
                corr = torch.tensor(corr, device=device)
                edata[i, j] = corr
                SSE_ij[i, j] = corr
    # 归一化：
    SSE_ij_np = SSE_ij.cpu().numpy()
    np.save(save_path + 'Pearson_SSE_Order3_TF_undirected.npy', SSE_ij_np)
    np.savetxt(save_path + "Pearson_SSE_Order3_TF_undirected.csv", SSE_ij_np, delimiter=",")
    Wik = edata
    Wik_np = Wik.cpu().numpy()
    np.save(save_path + 'Pearson_Wik_Order3_TF_undirected.npy', Wik_np)
    np.savetxt(save_path + "Pearson_Wik_Order3_TF_undirected.csv", Wik_np, delimiter=",")

    print("The SSE processing is over!!!!")
    return SSE_ij_np, Wik_np

def calc_MI(x, y, bins=100):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def SSE_Caculate_Order3_MutIfo(data, save_path, num_TF=334):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Calculating third-order SSEMatrix....\n")

    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data).to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]), device=device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]), device=device)

    for i in tqdm.tqdm(range(num_TF)):
        for j in range(data.shape[1]):
            if i != j:
                xs = data[:, i]
                ys = data[:, j]
                if torch.any(torch.isinf(xs)) or torch.any(torch.isnan(xs)) or torch.any(torch.isinf(ys)) or torch.any(
                        torch.isnan(ys)):
                    edata[i, j] = torch.tensor(0)
                else:
                    sum_xs = torch.sum(xs)
                    sum_xs_sq = torch.sum(xs ** 2)
                    sum_xs_cube = torch.sum(xs ** 3)
                    X = torch.tensor([
                        [data.shape[0], sum_xs, sum_xs_sq, sum_xs_cube],
                        [sum_xs, sum_xs_sq, sum_xs_cube, torch.sum(xs ** 4)],
                        [sum_xs_sq, sum_xs_cube, torch.sum(xs ** 4), torch.sum(xs ** 5)],
                        [sum_xs_cube, torch.sum(xs ** 4), torch.sum(xs ** 5), torch.sum(xs ** 6)]
                    ], device=device)
                    Y = torch.tensor(
                        [torch.sum(ys), torch.sum(xs * ys), torch.sum(xs ** 2 * ys), torch.sum(xs ** 3 * ys)],
                        device=device)
                    w_hat = torch.linalg.solve(X, Y)
                    y_hat = w_hat[0] + w_hat[1] * xs + w_hat[2] * xs ** 2 + w_hat[3] * xs ** 3
                    SSE_ij[i, j] = torch.sum((ys - y_hat) ** 2)
                    mi = calc_MI(xs.cpu().numpy(), ys.cpu().numpy())
                    mi = torch.tensor(mi)
                    mi = mi.to(device)
                    edata[i, j] = torch.pow(mi, 4) * torch.pow(torch.exp(-SSE_ij[i, j]), 0.25)
    # 归一化：
    SSE_ij_np = SSE_ij.cpu().numpy()
    np.save(save_path + 'MutIfo_SSE_Order3_TF.npy', SSE_ij_np)
    np.savetxt(save_path + "MutIfo_SSE_Order3_TF.csv", SSE_ij_np, delimiter=",")

    Wik = torch.zeros((data.shape[1], edata.shape[1]), device=device)
    for i in range(edata.shape[1]):
        temp = torch.pow(torch.sum(edata[:, i] ** 2), 0.5)
        for j in range(num_TF):
            if temp > 0:
                Wik[j, i] = edata[j, i] / temp
            else:
                break

    Wik_np = Wik.cpu().numpy()
    np.save(save_path + 'MutIfo_Wik_Order3_TF.npy', Wik_np)
    np.savetxt(save_path + "MutIfo_Wik_Order3_TF.csv", Wik_np, delimiter=",")

    print("The SSE processing is over!!!!")
    return SSE_ij_np, Wik_np

def SSE_Caculate_Order3_MutIfo_undirected(data, save_path, num_TF=334):
    device = torch.device("cpu")
    print("Calculating third-order SSEMatrix....\n")

    """求取残差平方和"""
    data = data.T
    data = torch.from_numpy(data).to(device)
    edata = torch.zeros((data.shape[1], data.shape[1]), device=device)
    SSE_ij = torch.zeros((data.shape[1], data.shape[1]), device=device)

    for i in tqdm.tqdm(range(num_TF)):
        for j in range(data.shape[1]):
            if i != j:
                xs = data[:, i]
                ys = data[:, j]

                mi = calc_MI(xs.cpu().numpy(), ys.cpu().numpy())
                mi = torch.tensor(mi)
                mi = mi.to(device)
                edata[i, j] = mi
                SSE_ij[i, j] = mi
    # 归一化：
    SSE_ij_np = SSE_ij.cpu().numpy()
    np.save(save_path + 'MutIfo_SSE_Order3_TF_undirected.npy', SSE_ij_np)
    np.savetxt(save_path + "MutIfo_SSE_Order3_TF_undirected.csv", SSE_ij_np, delimiter=",")
    Wik = torch.zeros((data.shape[1], edata.shape[1]), device=device)
    for i in range(edata.shape[1]):
        temp = torch.pow(torch.sum(edata[:, i] ** 2), 0.5)
        for j in range(num_TF):
            if temp > 0:
                Wik[j, i] = edata[j, i] / temp
            else:
                break
    Wik_np = Wik.cpu().numpy()
    np.save(save_path + 'MutIfo_Wik_Order3_TF_undirected.npy', Wik_np)
    np.savetxt(save_path + "MutIfo_Wik_Order3_TF_undirected.csv", Wik_np, delimiter=",")

    print("The SSE processing is over!!!!")
    return SSE_ij_np, Wik_np