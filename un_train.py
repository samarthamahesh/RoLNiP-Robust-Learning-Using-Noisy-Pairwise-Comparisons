import os, random, argparse
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


parser = argparse.ArgumentParser()
parser.add_argument('--seed',
                    type=int,
                    required=True)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

parser.add_argument('--datapath',
                    type=str,
                    required=True)

parser.add_argument('--numepochs',
                    type=int,
                    default=30)

parser.add_argument('--batchsize',
                    type=int,
                    default=1024)

parser.add_argument('--runs',
                    type=int,
                    required=True)
                    
parser.add_argument('--lname',
                    type=str,
                    default='sigmoid')
                    
parser.add_argument('--noise',
                    type=float,
                    required=True)


class CustomDataset(Dataset):

    def __init__(self, data_dir):
        self.data_x = np.load(os.path.join(data_dir, 'x.npy'))
        self.data_y = np.load(os.path.join(data_dir, 'y.npy'))
        self.data_true_y = np.load(os.path.join(data_dir, 'y_true.npy'))

    def find_params(self):
        num_features = self.data_x.shape[2]
        DATASIZE = len(self.data_y)
        pi_s_eta = len(np.where(self.data_y == 1)[0]) / DATASIZE
        pi_d_eta = 1 - pi_s_eta

        return num_features, pi_s_eta, pi_d_eta

    def __getitem__(self, ind):
        return torch.from_numpy(self.data_x[ind]), torch.from_numpy(self.data_true_y[ind]), torch.from_numpy(self.data_y[ind])
                                                                    
    def __len__(self):
        return len(self.data_y)


class CustomModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(CustomModel, self).__init__()
        
        activation = nn.ReLU()

        modules = []
        
        modules.append(nn.Linear(in_dim, 128))
        modules.append(activation)
        
        modules.append(nn.Linear(128, 32))
        modules.append(activation)

        modules.append(nn.Linear(32, 8))
        modules.append(activation)

        modules.append(nn.Linear(8, out_dim))
        modules.append(nn.Sigmoid())

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x).squeeze()

    def prob(self, x):
        return self.forward(x)

    def predict(self, x):
        out = self.prob(x)
        return torch.where(out < 0.5, 0, 1)


class CustomLoss(nn.Module):

    def __init__(self, pi_p_eta, pi_s_eta, loss_name):
        super(CustomLoss, self).__init__()
        self.pi_p_eta = pi_p_eta
        self.pi_s_eta = pi_s_eta
        self.lname = loss_name

    def loss_l(self, z, t):
        if self.lname == 'sigmoid':
            return 1/(1 + torch.exp(z * t))
        elif self.lname == 'ramp':
            tmp1 = 1 - z * t
            tmp2 = - 1 - z * t
            return ((tmp1 + torch.abs(tmp1)) - (tmp2 + torch.abs(tmp2))) / 2
        elif self.lname == 'mse':
            return (z - t)**2
        elif self.lname == 'mae':
            return torch.abs(z - t)

    def loss_L(self, z, t):
        return (self.pi_p_eta*self.loss_l(z, t) - (1-self.pi_p_eta)*self.loss_l(z, -t)) / (2*self.pi_p_eta-1)

    def forward(self, output, target):
        L1 = self.loss_L(output, 1)
        L0 = self.loss_L(output, -1)

        L1 = torch.where(target == 1, 1.0, 0.0) * L1
        L0 = torch.where(target == 1, 0.0, 1.0) * L0
        
        l1 = torch.sum(L1) / (2 * torch.sum(target))
        l0 = torch.sum(L0) / (2 * len(target) - 2 * torch.sum(target))

        return self.pi_s_eta * l1 + (1 - self.pi_s_eta) * l0
        

def training(ds, dir_path, num_epochs, runs, batch_size, lr, k, pi_p_eta, pi_s_eta, num_features, lname, noise):    

    splits=KFold(n_splits=k,shuffle=True,random_state=42)

    f_path = os.path.join(dir_path, 'output.log')
    r_path = os.path.join(dir_path, 'report.log')

    print("Noise: {}".format(noise), flush=True)
    print("Noisy Prior (PI_P_ETA, PI_S_ETA): {}, {}".format(pi_p_eta, pi_s_eta), flush=True)
    print("\nLoss function: {}".format(lname), flush=True)

    with open(f_path, 'a') as f:
        f.write("Noise: {}\n".format(noise))
        f.write("Noisy Prior (PI_P_ETA): {}\n".format(pi_p_eta))
        f.write("Noisy Prior (PI_S_ETA): {}\n".format(pi_s_eta))
        f.write("\nLoss function: {}\n".format(lname))
        f.close()

    history = {
        'tr_loss': [],
        'val_loss': [],
        'tr_acc': [],
        'val_acc': []
    }

    for run in range(runs):

        print("\n\nRun {}".format(run+1), flush=True)
        with open(f_path, 'a') as f:
            f.write('\n\nRun {}\n'.format(run+1))
            f.close()

    
        model = CustomModel(num_features, 1)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = CustomLoss(pi_p_eta, pi_s_eta, lname)

        r_tr_loss, r_val_loss, r_tr_acc, r_val_acc = 0, 0, 0, 0
        for fold, (tr_id, val_id) in enumerate(splits.split(np.arange(len(ds)))):

            print('\nFold {}\n'.format(fold+1), flush=True)
            with open(f_path, 'a') as f:
                f.write('\nFold {}\n\n'.format(fold+1))
                f.close()

            tr_sampler = SubsetRandomSampler(tr_id)
            val_sampler = SubsetRandomSampler(val_id)
            tr_loader = DataLoader(ds, batch_size=batch_size, sampler=tr_sampler)
            val_loader = DataLoader(ds, batch_size=batch_size, sampler=val_sampler)

            fold_sum = 0
            fold_total = 0


            for epoch in range(num_epochs):
            
                tr_loss, tr_acc = 0, 0
                model.train()
                for _, data in enumerate(tr_loader):
                    data_x, data_true_y, data_y = data

                    fold_sum += torch.sum(data_y)
                    fold_total += len(data_y)

                    data_x = data_x.float()

                    data_x = data_x.to(device)
                    data_y = data_y.to(device)

                    opt.zero_grad()
                    out = model.forward(data_x)
                    loss = criterion(out, data_y)
                    
                    pred = model.predict(data_x)
                    pred = pred.to('cpu')

                    acc = accuracy_score(pred.reshape(-1), data_true_y.reshape(-1))

                    loss.backward()
                    opt.step()

                    tr_loss += loss.item() * len(data_x)
                    tr_acc += acc * len(data_x)


                val_loss, val_acc = 0, 0
                model.eval()
                for _, data in enumerate(val_loader):
                    data_x, data_true_y, data_y = data
                    data_x = data_x.float()
                    
                    data_x = data_x.to(device)
                    data_y = data_y.to(device)

                    out = model.forward(data_x)
                    loss = criterion(out, data_y)

                    pred = model.predict(data_x)
                    pred = pred.to('cpu')
                    acc = accuracy_score(pred.reshape(-1), data_true_y.reshape(-1))

                    val_loss += loss.item() * len(data_x)
                    val_acc += acc * len(data_x)

                tr_loss = tr_loss / len(tr_loader.sampler)
                tr_acc = tr_acc / len(tr_loader.sampler) * 100
                val_loss = val_loss / len(val_loader.sampler)
                val_acc = val_acc / len(val_loader.sampler) * 100

                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %".format(epoch + 1,num_epochs, tr_loss, val_loss, tr_acc, val_acc), flush=True)
                with open(f_path, 'a') as f:
                    f.write("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %\n".format(epoch + 1,num_epochs, tr_loss, val_loss, tr_acc, val_acc))
                    f.close()

            r_tr_loss += tr_loss
            r_val_loss += val_loss
            r_tr_acc += tr_acc
            r_val_acc += val_acc


            print(fold_sum/fold_total)


        r_tr_loss /= k
        r_val_loss /= k
        r_tr_acc /= k
        r_val_acc /= k

        print("\n\nRUN:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %".format(run + 1, runs, r_tr_loss, r_val_loss, r_tr_acc, r_val_acc), flush=True)
        with open(f_path, 'a') as f:
            f.write("\n\nRUN:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %\n".format(run + 1, runs, r_tr_loss, r_val_loss, r_tr_acc, r_val_acc))
            f.close()

        history['tr_loss'].append(r_tr_loss)
        history['val_loss'].append(r_val_loss)
        history['tr_acc'].append(r_tr_acc)
        history['val_acc'].append(r_val_acc)


    tr_loss_mean, tr_loss_std = np.mean(np.array(history['tr_loss'])), np.std(np.array(history['tr_loss']))
    val_loss_mean, val_loss_std = np.mean(np.array(history['val_loss'])), np.std(np.array(history['val_loss']))
    tr_acc_mean, tr_acc_std = np.mean(np.array(history['tr_acc'])), np.std(np.array(history['tr_acc']))
    val_acc_mean, val_acc_std = np.mean(np.array(history['val_acc'])), np.std(np.array(history['val_acc']))


    with open(r_path, 'a') as f:
        f.write("Train loss -> Mean: {}\tStd: {}\n".format(tr_loss_mean, tr_loss_std))
        f.write("Val loss -> Mean: {}\tStd: {}\n".format(val_loss_mean, val_loss_std))
        
        f.write("Train acc -> Mean: {}\tStd: {}\n".format(tr_acc_mean, tr_acc_std))
        f.write("Val acc -> Mean: {}\tStd: {}\n".format(val_acc_mean, val_acc_std))

        f.close()


if __name__ == '__main__':
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = args.datapath
    save_dir = args.outpath
    BATCHSIZE = args.batchsize
    EPOCHS = args.numepochs
    RUNS = args.runs
    LNAME = args.lname
    NOISE = args.noise

    LR = 0.001
    K = 10

    ds = CustomDataset(data_dir)
    NUM_FEATURES, PI_S_ETA, PI_D_ETA = ds.find_params()

    PI_P_ETA = (1 + (2 * PI_S_ETA - 1)**0.5) / 2
    PI_N_ETA = 1 - PI_P_ETA

    dir_path = os.path.join(save_dir, str(NOISE))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    training(ds, dir_path, EPOCHS, RUNS, BATCHSIZE, LR, K, PI_P_ETA, PI_S_ETA, NUM_FEATURES, LNAME, NOISE)
