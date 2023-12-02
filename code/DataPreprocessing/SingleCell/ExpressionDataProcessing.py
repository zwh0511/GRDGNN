import os
import numpy as np
import units as uf
import tqdm
import pickle
import scipy.sparse as ssp

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

def main():
    for i in range(0, 4):
        for j in tqdm.tqdm(range(0, 7)):
            if i == 0 and j != 3:
                continue
            else:
                for k in range(0, 2):
                    data_root_path = uf.get_project_dir("GRDGNN")
                    expr_data_path = os.path.join(data_root_path, "data/SingleCell/processed/{}/{}/{}/expression.tsv".format(goldNetwork_type[i], networkDataset_name[j], nums_gene[k]))
                    data_save_path = os.path.join(data_root_path, "data/SingleCell/processed/{}/{}/{}/".format(goldNetwork_type[i], networkDataset_name[j], nums_gene[k]))
                    expr_data = uf.load_eprData(expr_data_path, rownum=gene_dict[nums_gene[k]][networkDataset_name[j]], colnum=cell_dict[networkDataset_name[j]], outdim=cell_dict[networkDataset_name[j]])
                    np.savetxt(data_save_path + "expression_processed.csv", expr_data, delimiter=",")
                    pickle.dump(ssp.csc_matrix(expr_data), open(data_save_path + "goldExpression.allx", "wb"))
            print("="*40)
            print(" "*10 + "{} is overed...".format(goldNetwork_type[i]))
            print("="*40)


if __name__ == '__main__':
    main()
