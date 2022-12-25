import numpy as np

# Parent class with distribution generator functions
class DatasetModel():

    def __init__(self):
        pass

    def get_gen0(self, n):
        X0 = self.X_data[self.y_data == 0]
        n0 = len(X0)
        ind = np.random.choice(range(n0), n, replace=True)
        return X0[ind]

    def get_gen1(self, n):
        X1 = self.X_data[self.y_data == 1]
        n1 = len(X1)
        ind = np.random.choice(range(n1), n, replace=True)
        return X1[ind]


class AdultDataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(123)
            
                for pair in tmp[1:]:
                    ind, val = map(int, pair.split(":"))
                    fv[ind-1] = val

                X_data.append(fv)
                y_data.append((int(tmp[0])+1)//2)

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

class BreastCancerDataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(10)
            
                for pair in tmp[1:]:
                    ind, val = map(float, pair.split(":"))
                    fv[int(ind)-1] = val

                X_data.append(fv)
                if tmp[0] == '2':
                    y_data.append(0)
                else:
                    y_data.append(1)

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

class CodRNADataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(8)
            
                for pair in tmp[1:]:
                    ind, val = map(float, pair.split(":"))
                    fv[int(ind)-1] = val

                X_data.append(fv)
                y_data.append((int(tmp[0])+1)//2)

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

class IonosphereDataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(34)
            
                for pair in tmp[1:]:
                    ind, val = map(float, pair.split(":"))
                    fv[int(ind)-1] = val

                X_data.append(fv)
                y_data.append((int(tmp[0])+1)//2)

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

class PhishingDataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(68)
            
                for pair in tmp[1:]:
                    ind, val = map(float, pair.split(":"))
                    fv[int(ind)-1] = val

                X_data.append(fv)
                y_data.append(int(tmp[0]))

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

class W8aDataset(DatasetModel):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            X_data, y_data = [], []

            for line in lines:
                if line == "":
                    continue

                tmp = line.split()
                fv = np.zeros(300)

                if len(tmp) < 10:
                    continue

                for pair in tmp[1:]:
                    ind, val = map(int, pair.split(":"))
                    fv[ind-1] = val

                X_data.append(fv)
                y_data.append((int(tmp[0])+1)//2)

            f.close()

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)


# Pairwise data generation class
class CustomDataset():

    def __init__(self, ds, prior, PI_S, datasize, noisesd, noiseds):
        ns = int(PI_S * datasize)
        nd = datasize - ns

        nspp = np.random.binomial(ns, prior**2 / (prior**2 + (1-prior)**2))
        nsnn = ns - nspp

        xs = np.concatenate((
            np.dstack((ds.get_gen1(nspp), ds.get_gen1(nspp))),
            np.dstack((ds.get_gen0(nsnn), ds.get_gen0(nsnn)))
        ))
        ys = np.concatenate((
            np.hstack((np.ones((nspp, 1)), np.ones((nspp, 1)))),
            np.hstack((np.zeros((nsnn, 1)), np.zeros((nsnn, 1))))
        ))

        ndpn = np.random.binomial(nd, 0.5)
        ndnp = nd - ndpn

        xd = np.concatenate((
            np.dstack((ds.get_gen1(ndpn), ds.get_gen0(ndpn))),
            np.dstack((ds.get_gen0(ndnp), ds.get_gen1(ndnp)))
        ))

        yd = np.concatenate((
            np.hstack((np.ones((ndpn, 1)), np.zeros((ndpn, 1)))),
            np.hstack((np.zeros((ndnp, 1)), np.ones((ndnp, 1))))
        ))

        self.data_x = np.transpose(np.vstack((xs, xd)), (0, 2, 1))
        self.data_true_y = np.vstack((ys, yd))
        self.data_y = np.hstack((np.ones(ns), np.zeros(nd))).reshape(-1, 1)

        y0_ind = np.where(self.data_y == 0)[0]
        y1_ind = np.where(self.data_y == 1)[0]
        
        y0_ind_rep = np.random.choice(y0_ind, size=int(noisesd * len(y0_ind)), replace=False)
        self.data_y[y0_ind_rep] = 1 - self.data_y[y0_ind_rep]

        y1_ind_rep = np.random.choice(y1_ind, size=int(noiseds * len(y1_ind)), replace=False)
        self.data_y[y1_ind_rep] = 1 - self.data_y[y1_ind_rep]

    def get_data(self):
        return self.data_x, self.data_y, self.data_true_y
