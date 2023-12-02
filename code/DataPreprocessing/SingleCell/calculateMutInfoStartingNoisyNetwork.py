'''
通过残差平方和获取具有噪声的初始GRN
'''

import numpy as np
import os
import tqdm
from calculateSSE import SSE_Caculate_Order3_MutIfo
import units as uf


TFDict = {
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
    0: "hESC",
    1: "hHEP",
    2: "mDC",
    3: "mESC",
    4: "mHSC-E",
    5: "mHSC-GM",
    6: "mHSC-L"
}

cell_dict = {
    "hESC": 758,
    "hHEP": 425,
    "mDC": 383,
    "mESC": 421,
    "mHSC-GM": 889,
    "mHSC-L": 847,
    "mHSC-E": 1071
}

gene_dict = {
    "TFs+500": {
        "hESC": 910,
        "hHEP": 948,
        "mDC": 821,
        "mESC": 1120,
        "mHSC-GM": 631,
        "mHSC-L": 560,
        "mHSC-E": 703
    },
    "TFs+1000": {
        "hESC": 1410,
        "hHEP": 1448,
        "mDC": 1321,
        "mESC": 1620,
        "mHSC-GM": 1131,
        "mHSC-L": 692,
        "mHSC-E": 1203
    }
}

def main():

    for i in range(0, 4):
        for j in tqdm.tqdm(range(0, 7)):
            if i == 0 and j != 3:
                continue
            else:
                for k in range(0, 2):
                    data_root_path = uf.get_project_dir("GRDGNN")
                    data_path = os.path.join(data_root_path, "data/SingleCell/processed/{}/{}/{}/".format(goldNetwork_type[i], networkDataset_name[j], nums_gene[k]))
                    SSE_path = os.path.join(data_path, "MutIfo_SSE_Order3_TF.npy")
                    wik_path = os.path.join(data_path, "MutIfo_Wik_Order3_TF.npy")
                    expre_data_path = os.path.join(data_path, "expression_processed.csv")
                    goldGRN = np.load(os.path.join(data_path, 'goldGRN.csc'), allow_pickle=True)

                    data = np.loadtxt(expre_data_path, delimiter=",")
                    if os.path.exists(wik_path):
                        wik = np.load(os.path.join(wik_path), allow_pickle=True)
                    elif os.path.exists(SSE_path):
                        SSE = np.load(os.path.join(SSE_path), allow_pickle=True)
                        wik = uf.Normalization_Pearson(SSE, data_path)
                    else:
                        _, wik = SSE_Caculate_Order3_MutIfo(data, data_path, num_TF=TFDict[networkDataset_name[j]])
                    for ii in range(19, 20):
                        networks = uf.BuildNetworksBySettingNumberOfEdges(wik=wik, nodes=goldGRN.shape[0], average=ii)
                        np.save(data_path + 'MutIfo_avgDegree_' + str(ii) + '_Networks_Order3_TF.npy', networks)
                        networks = networks.todense()
                        np.savetxt(data_path + "MutIfo_avgDegree_" + str(ii) + "_Networks_Order3_TF.csv", networks, delimiter=",")
                        print("==" * 20)
                        print("  " * 10 + "{} time is processed....".format(str(ii)))
                        print("==" * 20)

if __name__ == "__main__":
    main()