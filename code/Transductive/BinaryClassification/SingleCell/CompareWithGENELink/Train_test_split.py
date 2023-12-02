import pandas as pd
import numpy as np
import tqdm
import os
import argparse
import units as uf

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
args = parser.parse_args()

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

def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,density,p_val=args.p_val, Location="pred"):

    print("Starting split the dataset....")
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)


    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:

            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]

        else:
            np.random.shuffle(pos_dict[k])
            if Location == "pred":

                train_pos[k] = pos_dict[k][len(pos_dict[k]) * 1 // 3:]
                test_pos[k] = pos_dict[k][:len(pos_dict[k]) * 1 // 3]
                val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
                train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]

            elif Location == "mid":
                train_pos[k] = np.concatenate(
                    (pos_dict[k][:len(pos_dict[k]) * 1 // 3], pos_dict[k][len(pos_dict[k]) * 2 // 3:]))
                test_pos[k] = pos_dict[k][len(pos_dict[k]) * 1 // 3:len(pos_dict[k]) * 2 // 3]
                val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
                train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]

            else:
                train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
                test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]
                val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
                train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]


    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(len(train_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
            train_neg[k].append(neg)


    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]



    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)


    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
            val_neg[k].append(neg)


    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k, j])


    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label


    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:,0]
    val_sample['Target'] = val_set_a[:,1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)



    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density-count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_neg_set = []
    for i in range(test_neg_num):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1,t2] in test_neg_set:
            t2 = np.random.choice(gene_set)

        test_neg_set.append([t1,t2])

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)

def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file,val_set_file,test_set_file,
                                          p_val=args.p_val, Location="pred"):
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)

        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) ==1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) ==2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            if Location == "pred":

                train_pos[k] = pos_dict[k][len(pos_dict[k]) * 1 // 3:]
                test_pos[k] = pos_dict[k][:len(pos_dict[k]) * 1 // 3]
                val_pos[k] = test_pos[k][:len(train_pos[k]) // 5]
                test_pos[k] = test_pos[k][:]

            elif Location == "mid":
                train_pos[k] = np.concatenate(
                    (pos_dict[k][:len(pos_dict[k]) * 1 // 3], pos_dict[k][len(pos_dict[k]) * 2 // 3:]))
                test_pos[k] = pos_dict[k][len(pos_dict[k]) * 1 // 3:len(pos_dict[k]) * 2 // 3]
                val_pos[k] = test_pos[k][:len(train_pos[k]) // 5]
                test_pos[k] = test_pos[k][:]

            else:
                train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
                test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]
                val_pos[k] = test_pos[k][:len(train_pos[k]) // 5]
                test_pos[k] = test_pos[k][:]

    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        if len(neg_dict[k]) == 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_neg[k] = neg_dict[k]
            else:
                test_neg[k] = neg_dict[k]

        elif len(pos_dict[k]) == 2:
            np.random.shuffle(neg_dict[k])
            train_neg[k] = [neg_dict[k][0]]
            test_neg[k] = [neg_dict[k][1]]
        else:
            neg_num = len(neg_dict[k])
            np.random.shuffle(neg_dict[k])
            if Location == "pred":
                train_neg[k] = neg_dict[k][neg_num * 1 // 3:]
                val_neg[k] = neg_dict[k][:neg_num * 1 // 3 // 5]
                test_neg[k] = neg_dict[k][:neg_num * 1 // 3]

            elif Location == "mid":
                train_neg[k] = neg_dict[k][:neg_num * 1 // 3] + neg_dict[k][neg_num * 2 // 3:]
                val_neg[k] = neg_dict[k][neg_num * 1 // 3: neg_num * 1 // 3 + neg_num * 1 // 3 // 5]
                test_neg[k] = neg_dict[k][neg_num * 1 // 3: neg_num * 2 // 3]

            else:
                train_neg[k] = neg_dict[k][:neg_num * 2 // 3]
                val_neg[k] = neg_dict[k][neg_num * 2 // 3: neg_num * 2 // 3 + neg_num * 1 // 3 // 5]
                test_neg[k] = neg_dict[k][neg_num * 2 // 3:]
    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k,val])

    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k,val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]



    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k,val])

    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k,val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)




    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k,j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k,j])


    test_set = test_pos_set +test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:,0]
    test['Target'] = test_sample[:,1]
    test['Label'] = test_label
    test.to_csv(test_set_file)

if __name__ == '__main__':
    Location = {
        0: "pred",
        1: "mid",
        2: "last"
    }

    root_path = uf.get_project_dir("GRDGNN")
    data_loader_path = os.path.join(root_path, "data/SingleCell/processed")

    data_save_path = data_loader_path

    for gold_type in range(0, 4):
        for dataset_name in tqdm.tqdm(range(0, 7)):
            if gold_type == 0 and dataset_name != 3:
                continue
            else:
                for num_gene in range(0, 2):
                    net_type = goldNetwork_type[gold_type]
                    data_type = networkDataset_name[dataset_name]
                    gene_num = nums_gene[num_gene]
                    if num_gene == 0:
                        numName = "500"
                    else:
                        numName = "1000"
                    density = uf.Network_Statistic(data_type=data_type, net_scale=numName, net_type=net_type)
                    TF2file = data_loader_path + '/' + net_type + '/' + data_type + '/' + gene_num + '/TF2genelink.csv'
                    Gene2file = data_loader_path + '/' + net_type + '/' + data_type + '/' + gene_num + '/Target2genelink.csv'
                    label_file = data_loader_path + '/' + net_type + '/' + data_type + '/' + gene_num + '/Label2genelink.csv'
                    for i in range(3):
                        train_set_file = data_save_path + '/' + net_type + '/' + data_type + '/' + gene_num + "/" + Location[i] + '_Train_set_o.csv'
                        test_set_file = data_save_path + '/' + net_type + '/' + data_type + '/' + gene_num + "/" + Location[i] + '_Test_set_o.csv'
                        val_set_file = data_save_path + '/' + net_type + '/' + data_type + '/' + gene_num + "/" + Location[i] + '_Validation_set_o.csv'

                        path = data_save_path + '/' + net_type + '/' + data_type + '/' + gene_num + "/"
                        if not os.path.exists(path):
                            os.makedirs(path)


                        if net_type == 'Specific Dataset':
                            Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
                                                                  test_set_file, Location=Location[i])
                        else:
                            train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file, density, Location=Location[i])

        print("【{}】 processing is finished.".format(goldNetwork_type[gold_type]))