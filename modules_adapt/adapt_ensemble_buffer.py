import numpy as np


class BufferData:
    def __init__(self, x_data, act_data, y_data):
        self.x_data = x_data
        self.act_data = act_data
        self.y_data = y_data

    def extend(self, x_data, act_data, y):
        self.x_data = np.vstack((self.x_data, x_data))
        self.act_data = np.vstack((self.act_data, act_data))
        self.y_data = np.hstack((self.y_data, y))

    def get_buffer_sample(self, n_samples):
        idxs = np.random.choice(np.arange(self.x_data.shape[0]), size=n_samples, replace=True)
        # set x data
        x_data = self.x_data[idxs]

        # set y data
        y_data = np.take(self.y_data, idxs)

        # set act data
        act_data = self.act_data[idxs]

        return x_data, act_data, y_data


class AdaptationEnsemble:
    def __init__(self, act_ensemble_dict):
        self.act_ensemble_dict = act_ensemble_dict
        self.data_buffers = {}
        self.last_fit_x_data = None
        self.last_fit_act_data = None
        self.last_fit_y_data = None
        self.weights = []

    def fit(self, X_data, act_data, y):
        self.partial_fit(X_data, act_data, y)

    def partial_fit(self, X_data, act_data, y):
        # stores x data and activation data in class buffers and fits to clusterer
        act_data = np.array(act_data)
        for i in range(len(act_data[0])):
            self.act_ensemble_dict[i].partial_fit(act_data[:,i], y)
        self.__partial_fit_buffers(X_data, act_data, y)
        self.last_fit_x_data = X_data
        self.last_fit_act_data = act_data
        self.last_fit_y_data = y



    def __partial_fit_buffers(self, X_data, act_data, y):
        # add known instances
        y_key = np.array(y).astype(int)
        for k, v in self.data_buffers.items():
            idxs = np.where(y_key == k)[0]
            if len(idxs) > 0:
                v.extend(X_data[idxs], act_data[idxs], np.take(y, idxs))

        # add new classes
        for n in np.unique(y_key):
            if n not in self.data_buffers.keys():
                idxs = np.where(y_key == n)[0]
                self.data_buffers[n] = BufferData(X_data[idxs], act_data[idxs], np.take(y, idxs))
                # maybe augment new data and add it here


    def get_buffer_data(self, y, n_samples):
        # returns a selection of original x data and the corresponding activation data based on the y values provided
        x_data = []
        act_data = []
        y_data = []
        y_key = np.array(y).astype(int)
        true_y_values = np.unique(y_key)
        print('true y values: %s'%(true_y_values))
        for c, v in self.data_buffers.items():
            if c not in y_key:
                n_samples = 20
            else:
                x, a, y = self.data_buffers[c].get_buffer_sample(n_samples)
                x_data.extend(x)
                act_data.extend(a)
                y_data.extend(y)

        # append the last fitted data to ensure it is included in the returned sample
        if self.last_fit_x_data is not None:
            if len(x_data) == 0:
                x_data = self.last_fit_x_data
            else:
                x_data = np.vstack((x_data, self.last_fit_x_data))
        if self.last_fit_y_data is not None:
            if len(y_data) == 0:
                y_data = self.last_fit_y_data
            else:
                y_data = np.hstack((y_data, self.last_fit_y_data))
        if self.last_fit_act_data is not None:
            if len(act_data) == 0:
                act_data = self.last_fit_act_data
            else:
                act_data = np.vstack((act_data, self.last_fit_act_data))

        return x_data, act_data, y_data



    def predict_act(self, X):
        # get predicts from each classifier
        predicts = []
        for k in range(len(X[0])):
            predicts.append(self.act_ensemble_dict[k].predict(X[:, k]))
        predicts = np.array(predicts)

        weighted_predicts = []
        # append predicts based on weights
        for i, p in enumerate(predicts):
            for _ in range(self.weights[i]):
                weighted_predicts.append(p)
        weighted_predicts = np.array(weighted_predicts)

        weighted_predicts = [np.bincount(weighted_predicts[:, p]).argmax() for p in
                    range(len(weighted_predicts[0]))]

        return weighted_predicts


