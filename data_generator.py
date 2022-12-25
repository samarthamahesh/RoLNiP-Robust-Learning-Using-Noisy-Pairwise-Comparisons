import os, shutil, random, argparse
import numpy as np
from dataclass import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    required=False)

parser.add_argument('--dataset',
                    type=str,
                    required=True)

parser.add_argument('--datainpath',
                    type=str,
                    required=True)

parser.add_argument('--dataoutpath',
                    type=str,
                    required=True)

parser.add_argument('--noisesd',
                    type=float,
                    required=True)

parser.add_argument('--noiseds',
                    type=float,
                    required=True)

parser.add_argument('--datasize',
                    type=int,
                    required=True)
                    
parser.add_argument('--prior',
                    type=float,
                    required=True)



if __name__ == '__main__':

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    DATASET_NAME = args.dataset

    data_in_path = args.datainpath
    data_out_dir = args.dataoutpath
    NOISESD = args.noisesd
    NOISEDS = args.noiseds
    PRIOR = args.prior
    DATASIZE = args.datasize

    PI_P = PRIOR
    PI_N = 1 - PI_P
    PI_S = PI_P ** 2 + PI_N ** 2
    PI_D = 1 - PI_S

    if DATASET_NAME == 'adult':
        dataclass = AdultDataset(data_in_path)
    elif DATASET_NAME == 'breast_cancer':
        dataclass = BreastCancerDataset(data_in_path)
    elif DATASET_NAME == 'codrna':
        dataclass = CodRNADataset(data_in_path)
    elif DATASET_NAME == 'ionosphere':
        dataclass = IonosphereDataset(data_in_path)
    elif DATASET_NAME == 'phishing':
        dataclass = PhishingDataset(data_in_path)
    elif DATASET_NAME == 'w8a':
        dataclass = W8aDataset(data_in_path)

    pair_data = CustomDataset(dataclass, PRIOR, PI_S, DATASIZE, NOISESD, NOISEDS)

    X_data, y_data, true_y_data = pair_data.get_data()

    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)
    else:
        shutil.rmtree(data_out_dir)
        os.makedirs(data_out_dir)

    np.save(os.path.join(data_out_dir, "x.npy"), X_data)
    np.save(os.path.join(data_out_dir, "y.npy"), y_data)
    np.save(os.path.join(data_out_dir, "y_true.npy"), true_y_data)

    with open(os.path.join(data_out_dir, "noise.t"), 'a') as f:
        f.write(str(NOISEDS) + "\n" + str(NOISESD))
        f.close()