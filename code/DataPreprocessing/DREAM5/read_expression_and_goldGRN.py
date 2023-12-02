import pandas as pd
import numpy as np
import os
import pickle
import scipy.sparse as ssp
import argparse
from scipy.stats import pearsonr
import units as uf
# Setting of Dream dataset size
rowDict = {}
colDict = {}
filename = {}
Goldfilename = {}
rowDict['net3'] = 805
rowDict['net4'] = 536
colDict['net3'] = 4511
colDict['net4'] = 5950
dreamTFdict = {}
dreamTFdict['net3'] = 334
dreamTFdict['net4'] = 333

dataName=["net3", "net4"]



root_path = uf.get_project_dir("GRDGNN")
print(root_path)
# 基因表达数据路径
filename['net3'] = root_path + "data/DREAM/raw/net3/net3_expression_data.tsv"
filename['net4'] = root_path + "data/DREAM/raw/net4/net4_expression_data.tsv"

# 基因金标准数据路径
Goldfilename['net3'] = root_path + "data/DREAM/raw/net3/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv"
Goldfilename['net4'] = root_path + "data/DREAM/raw/net4/DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv"

# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, sample_size, feature_size):
    print("starting read feature file csc\n")
    data = np.zeros((feature_size, sample_size))
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                words = line.split()
                ncount = 0
                for word in words:
                    data[count, ncount] = word
                    ncount = ncount + 1
            count = count + 1
        f.close()
    data = data.T
    feature = ssp.csc_matrix(data)
    print(" read feature file csc is overed.....\n")
    return feature


# Load gold standard edges into sparse matrix
def read_edge_file_csc(filename, sample_size):
    print("starting read edge file csc\n")
    row = []
    col = []
    data = np.zeros((sample_size, sample_size))
    record_ij = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = int(words[0][1:]) - 1
            end2 = int(words[1][1:]) - 1
            if words[2][:] == "1":
                data[end1, end2] = 1
                if str(end1) + "_" + str(end2) in record_ij:
                    print(str(end1) + "_" + str(end2))
                record_ij[str(end1) + "_" + str(end2)] = ''
    f.close()
    mtx = ssp.csc_matrix(data)
    print(" read edge file csc is overed.....\n")
    return mtx


for datasetname in dataName:
    feature_filename = filename[datasetname]
    edge_filename = Goldfilename[datasetname]
    # output
    sample_size = colDict[datasetname]
    feature_size = rowDict[datasetname]
    graphcsc = read_edge_file_csc(edge_filename, sample_size=sample_size)
    allx = read_feature_file_sparse(feature_filename, sample_size=sample_size, feature_size=feature_size)
    pickle.dump(graphcsc, open(root_path + "data/DREAM/processed/{}/goldGRN_{}.csc".format(datasetname, datasetname), "wb"))
    pickle.dump(allx, open(root_path + "data/DREAM/processed/{}/goldExpression_{}.allx".format(datasetname, datasetname), "wb"))

    print('preprocessing is over.......................')