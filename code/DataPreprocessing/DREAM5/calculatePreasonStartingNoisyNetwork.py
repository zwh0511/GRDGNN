'''
通过残差平方和获取具有噪声的初始GRN
'''
import pandas as pd
import numpy as np
import os
import pickle
import torch
import scipy.sparse as ssp
import argparse
from calculateSSE import SSE_Caculate_Order3_Pearson
import units as uf


TFDict = {
    'net1': 195,
    'net2': 99,
    'net3': 334,
    'net4': 333
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', type=bool, default=False, help='Whether to map different datasets to the same data...')
    parser.add_argument('--dataset', type=str, default='net3', help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
    return parser.parse_args()
# 获取根目录

def main(args):
    data_root_path = uf.get_project_dir("GRDGNN")
    data_path = os.path.join(data_root_path, "data/DREAM/processed/{}/".format(args.dataset))
    SSE_path = os.path.join(data_path, "Pearson_SSE_Order3_TF.npy")
    wik_path = os.path.join(data_path, "Pearson_Wik_Order3_TF.npy")
    if args.mapping:
        expre_data_path = os.path.join(data_path, "goldExpression_PCA_Mapped.csv")
    else:
        expre_data_path = os.path.join(data_path, "expression.csv")


    data = np.loadtxt(expre_data_path, delimiter=",")
    if os.path.exists(wik_path):
        wik = np.load(os.path.join(wik_path), allow_pickle=True)
    elif os.path.exists(SSE_path):
        SSE = np.load(os.path.join(SSE_path), allow_pickle=True)
        wik = uf.Normalization_Pearson(SSE, data_path)
    else:
        _, wik = SSE_Caculate_Order3_Pearson(data, data_path, num_TF=TFDict[args.dataset])
    temp = 5
    for i in range(20):
        temp = temp + 5
        Threshould = (temp / 100)
        if Threshould == 0.3 or Threshould == 0.8:
            networks = uf.BuildNetworksByThreshould(wik, Threshold=Threshould)
            np.save(data_path + 'Pearson_Threshold_' + str(Threshould) + '_Networks_Order3_TF.npy', networks)
            networks = networks.todense()
            np.savetxt(data_path + "Pearson_Threshold_" + str(Threshould) + "_Networks_Order3_TF.csv", networks, delimiter=",")


            print("==" * 20)
            print("  " * 10 + "{} time is processed....".format(str(i)))
            print("==" * 20)
        else:
            continue

if __name__ == "__main__":
    args = parse_args()
    main(args)