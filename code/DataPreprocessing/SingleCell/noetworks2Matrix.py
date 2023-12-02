import pandas as pd
import numpy as np
import os
import pickle
import scipy.sparse as ssp
import os
import numpy as np
import units as uf
import tqdm

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
        "mHSC-GM": 632,
        "mHSC-L": 560,
        "mHSC-E": 704
    },
    "TFs+1000": {
        "hESC": 1410,
        "hHEP": 1448,
        "mDC": 1321,
        "mESC": 1620,
        "mHSC-GM": 1132,
        "mHSC-L": 692,
        "mHSC-E": 1204
    }
}
# Load gold standard edges into sparse matrix
def read_edge_file_csc(filename, sample_size):
    data = np.zeros((sample_size, sample_size))
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = int(words[0][1:])
            end2 = int(words[1][1:])

            if end1 >= sample_size or end2 >= sample_size:
                raise Exception("The error in {}".format(filename))
            data[end1, end2] = 1
    f.close()
    mtx = ssp.csc_matrix(data)
    return mtx


def main():
    for i in range(0, 4):
        for j in range(0, 7):
            if i == 0 and j != 3:
                continue
            else:
                for k in range(0, 2):
                    data_root_path = uf.get_project_dir("GRDGNN")
                    net_data_path = os.path.join(data_root_path,
                                                 "data/SingleCell/processed/{}/{}/{}/goldStandardNet.tsv".format(
                                                     goldNetwork_type[i], networkDataset_name[j], nums_gene[k]))
                    data_save_path = os.path.join(data_root_path,
                                                  "data/SingleCell/processed/{}/{}/{}/".format(
                                                      goldNetwork_type[i], networkDataset_name[j], nums_gene[k]))
                    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)

                    graphcsc = read_edge_file_csc(net_data_path, sample_size=gene_dict[nums_gene[k]][networkDataset_name[j]])
                    print("{}-{}:{}".format(goldNetwork_type[i], networkDataset_name[j], str(graphcsc.nnz)))
                    pickle.dump(graphcsc, open(data_save_path + "goldGRN.csc", "wb"))

            print("="*40)
            print(" "*10 + "{} is overed...".format(goldNetwork_type[i]))
            print("="*40)


if __name__ == '__main__':
    main()
