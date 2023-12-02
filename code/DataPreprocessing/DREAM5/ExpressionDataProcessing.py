import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import units as uf

rowDict = {
    'net1': 805,
    'net2': 160,
    'net3': 805,
    'net4': 536
}

colDict = {
    'net1': 1643,
    'net2': 2810,
    'net3': 4511,
    'net4': 5950
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dim', type=int, default=160, help='the dim of output after PCA...')
    parser.add_argument('--mapping', type=bool, default=False, help='Whether to map different datasets to the same data...')
    parser.add_argument('--map_dataset', type=str, default="net3", help='Setting the basic dataset...')
    parser.add_argument('--dataset', type=str, default='net3', help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
    return parser.parse_args()

class CrossSpeciesTranscriptomics(nn.Module):
    def __init__(self, input_dim_A, input_dim_B, hidden_dim):
        super(CrossSpeciesTranscriptomics, self).__init__()
        self.encoder_A = nn.Sequential(
            nn.Linear(input_dim_A, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
        )
        self.encoder_B = nn.Sequential(
            nn.Linear(input_dim_B, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 16, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim_B),
        )

    def forward(self, input_A, input_B):
        encoded_A = self.encoder_A(input_A)
        encoded_B = self.encoder_B(input_B)
        decoded = self.decoder(encoded_A - encoded_B)
        return decoded

def main(args):

    for i in range(3, 5):
        args.dataset = "net" + str(i)
        if args.dataset != "net":
            data_root_path = uf.get_project_dir("pycharm_project_377")
            expr_data_path = os.path.join(data_root_path,
                                          "data/DREAM/raw/{}/{}_expression_data.tsv".format(
                                              args.dataset, args.dataset))
            data_save_path = os.path.join(data_root_path,
                                          "data/DREAM/processed/{}/".format(args.dataset))

            expr_data = uf.load_eprData(expr_data_path, rownum=rowDict[args.dataset], colnum=colDict[args.dataset],
                                        outdim=args.feat_dim)
            os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
            np.savetxt(data_save_path + "expression.csv", expr_data, delimiter=",")

            if args.mapping:
                mapping_data_path = os.path.join(data_root_path,
                                                 "data/DREAM/processed/{}/goldExpression_PCA.csv".format(
                                                     args.map_dataset))
                A_data = np.loadtxt(mapping_data_path, delimiter=",")
                A_data = np.array(A_data)
                B_data = expr_data.copy()
                input_dim_A = A_data.shape[1]
                input_dim_B = B_data.shape[1]
                # Move tensors to the GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                data_A_tensor = torch.tensor(A_data, dtype=torch.float, device=device)
                data_B_tensor = torch.tensor(B_data, dtype=torch.float, device=device)

                # Initialize model
                model = CrossSpeciesTranscriptomics(input_dim_A, input_dim_B, args.feat_dim).to(device)

                # Train model
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                num_epochs = 100
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    outputs = model(data_A_tensor, data_B_tensor)
                    loss = criterion(outputs, data_B_tensor)
                    loss.backward()
                    optimizer.step()

                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
                print("Mapping is overed !!!!")
                # Save mapped data to file
                with torch.no_grad():
                    mapped_data = model(data_A_tensor, data_B_tensor).detach().cpu().numpy()
                np.savetxt(data_save_path + "goldExpression_PCA_Mapped.csv", mapped_data, delimiter=",")

            print("=="*20)
            print(" "*10 + "{} is overed...".format(args.dataset))
            print("=="*20)


if __name__ == '__main__':
    args = parse_args()
    main(args)