import numpy as np
import os.path


class WeightedMF:
    """
    Do weighted matrix factorization by gradient descent
    Arguments:
        - P, C: P, C matrices
        - dict_user, dict_item: mapping between name and index of users and items
        - depth: number of latent features. Default: 5
        - n_epochs: number of epochs to train. Default: 10
        - lr: learning rate. Default: 1e-6
        - rgl: regularization value (lambda). Default: 0.02
        - early_stopping: if True, stop early in gradient descent when loss may not improve; otherwise, train to the last epoch. Default: True
        - verbose: if True, print out the information of process. Default: False
        - U, I: hidden representation for each user, item
    Methods:
        - fit_derivative(): training model with derivative
        - fit_formula(): training model with formula
        - get_recommendations(): return list of recommendations for a user given the user_id and number of items to recommend
        - save(), load(): save and load U, I if they have been calculated already

    """
    def __init__(self, P, C, n_epochs=10, depth=5, lr=1e-6, rgl=0.02, early_stopping=True, verbose=False):
        self.P = P
        self.C = C
        self.n_users = P.shape[0]
        self.n_items = P.shape[1]
        self.depth = depth
        self.U = np.random.normal(0, 0.5, size=(self.n_users, self.depth))
        self.I = np.random.normal(0, 0.5, size=(self.depth, self.n_items))
        self.predict = np.zeros_like(P)
        self.is_loaded = False
        self.rgl = rgl
        self.lr = lr
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.n_epochs = n_epochs

    def fit(self):
        if not self.is_loaded:
            loss = 0
            for current_epoch in range(self.n_epochs):
                prev_loss = loss
                for i in range(self.n_users):
                    C_u = np.zeros((self.n_items, self.n_items))
                    E = np.zeros((self.depth, self.depth))
                    for j in range(self.n_items):
                        C_u[j, j] = self.C[i, j]
                    self.U[i] = np.dot(
                        np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.I, C_u), self.I.T) + self.rgl * E), self.I), C_u),
                        self.P[i].T).T
                for i in range(self.n_users):
                    C_i = np.zeros((self.n_users, self.n_users))
                    E = np.zeros((self.depth, self.depth))
                    for j in range(self.n_users):
                        C_i[j, j] = self.C[i, j]
                    self.I[:, i] = np.dot(
                        np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.U.T, C_i), self.U) + self.rgl * E), self.U.T), C_i),
                        self.P[:, i])
                loss = np.sqrt(np.sum((np.dot(self.U, self.I) - self.P) ** 2) / (self.n_users * self.n_items))
                # Run at least 5 epochs
                if self.early_stopping and current_epoch > 5:
                    if prev_loss - loss < 0.0001:
                        break
                if self.verbose:
                    print("Training at {}th epoch".format(current_epoch + 1))
                    print("Avg. loss: {:.5f}".format(loss))
        self.predict = np.dot(self.U, self.I)

    def get_recommendations(self, user_index, n_rec_items):
        recommendations = np.argsort(self.predict[user_index])
        recommendations = recommendations[-n_rec_items:]
        # name_of_songs_rec = []
        # for i in range(n_rec_items):
        #     name_of_songs_rec.append(self.dict_item[recommendations[i]])
        # return name_of_songs_rec
        return recommendations

    def save(self):
        np.savetxt('./data/U.txt', self.U, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/I.txt', self.I, delimiter=' ', fmt='%.5f')
        np.savetxt('./data/predict.txt', self.predict, delimiter=' ', fmt='%.5f')

    def load(self):
        if os.path.isfile('./data/U.txt') and os.path.isfile('./data/I.txt'):
            with open('./data/U.txt', 'r') as f:
                self.U = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.U = np.array(self.U)
            with open('./data/I.txt', 'r') as f:
                self.I = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.I = np.array(self.I)
            self.is_loaded = True
        if os.path.isfile('./data/predict.txt'):
            with open('./data/predict.txt', 'r') as f:
                self.I = [[float(num) for num in line[:-1].split(' ')] for line in f]
                self.I = np.array(self.I)

