import pandas as pd
import tqdm
import os
import units as uf
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

def process_data(gold_type, dataset_name, num_gene):
    root_path = uf.get_project_dir("GRDGNN")
    data_root_path = os.path.join(root_path, "data/SingleCell/raw/{}/{}/{}/".format(goldNetwork_type[gold_type],
                                                                                    networkDataset_name[dataset_name],
                                                                                    nums_gene[num_gene]))
    data_root_savePath = os.path.join(root_path,
                                      "data/SingleCell/processed/{}/{}/{}/".format(goldNetwork_type[gold_type],
                                                                                   networkDataset_name[dataset_name],
                                                                                   nums_gene[num_gene]))

    ExpressionData_path = os.path.join(data_root_path, "BL--ExpressionData.csv")
    network_path = os.path.join(data_root_path, "BL--network.csv")
    Target_path = os.path.join(data_root_path, "Target.csv")
    TF_path = os.path.join(data_root_path, "TF.csv")

    # 加载数据
    expreData = pd.read_csv(ExpressionData_path, lineterminator='\n', encoding='utf-8')
    netwoData = pd.read_csv(network_path)
    targeData = pd.read_csv(Target_path)
    tfs_Data = pd.read_csv(TF_path)

    tfs = tfs_Data.iloc[:, 1].tolist()
    nums_TF = len(tfs)
    targets = targeData.iloc[:, 1].tolist()

    # 从 target 中去除 tf
    gene = [x for x in targets if x not in set(tfs)]
    # 将 TF 放在前面，gene 放在后面
    newIndex = tfs + gene

    # 保存 target_newId.tsv
    target_newId_path = os.path.join(data_root_savePath, "Target2genelink.csv")
    with open(target_newId_path, "w") as targets_f:
        targets_f.write("#ID,Gene,index\n")
        for i, index in enumerate(newIndex):
            targets_f.write(f"{i},{index},{i}\n")
    targets_f.close()
    # 保存 TF_newId.tsv
    TF_newId_path = os.path.join(data_root_savePath, "TF2genelink.csv")
    with open(TF_newId_path, "w") as tfs_f:
        tfs_f.write("#ID,TF,index\n")
        for i, index in enumerate(newIndex[:nums_TF]):
            tfs_f.write(f"{i},{index},{i}\n")
    tfs_f.close()
    GeneDict = {index: f"{i}" for i, index in enumerate(newIndex)}
    GeneDict_1 = {index: i for i, index in enumerate(newIndex)}
    GeneDict_2 = {i: f"{i}" for i in range(len(newIndex))}

    # 保存 goldStandardNet.tsv
    goldStandardNet_path = os.path.join(data_root_savePath, "Label2genelink.csv")
    with open(goldStandardNet_path, "w") as goldNet_f:
        goldNet_f.write("#ID,TF,index\n")
        idx = 0
        for row in netwoData.values:
            str3 = f"{idx},{GeneDict[row[0]]},{GeneDict[row[1]]}"
            idx += 1
            goldNet_f.write(f"{str3}\n")
    goldNet_f.close()
    # 对 expreData 进行排序和转换
    expreData[expreData.columns[0]] = expreData[expreData.columns[0]].map(GeneDict_1)
    expreData = expreData.sort_values(by=expreData.columns[0]).reset_index(drop=True)
    expreData[expreData.columns[0]] = expreData[expreData.columns[0]].map(GeneDict_2)

    # 保存 expression.tsv
    expression_path = os.path.join(data_root_savePath, "ExpressionData2Genelink.csv")
    expreData.to_csv(expression_path, sep=",", index=False)

for gold_type in range(0, 4):
    for dataset_name in tqdm.tqdm(range(0, 7)):
        if gold_type == 0 and dataset_name != 3:
            continue
        else:
            for num_gene in range(0, 2):
                process_data(gold_type, dataset_name, num_gene)

    print("【{}】 processing is finished.".format(goldNetwork_type[gold_type]))
print("overed......")
