# app/algorithms/svdpp.py
import numpy as np
from .base import Recommender

class SVDppRecommender(Recommender):
    def __init__(self, k, iterations=20, learning_rate=0.005, lambda_reg=0.02):
        super().__init__(k)
        self.name = "SVD++"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = R[R > 0].mean()

        for i in range(self.iterations):
            for u, item_idx in np.argwhere(R > 0):
                pred = self.global_mean + self.user_bias[u] + self.item_bias[item_idx] + self.P[u, :] @ self.Q[item_idx, :].T
                error = R[u, item_idx] - pred

                bu = self.user_bias[u]
                bi = self.item_bias[item_idx]
                pu = self.P[u, :]
                qi = self.Q[item_idx, :]

                self.user_bias[u] += self.learning_rate * (error - self.lambda_reg * bu)
                self.item_bias[item_idx] += self.learning_rate * (error - self.lambda_reg * bi)
                self.P[u, :] += self.learning_rate * (error * qi - self.lambda_reg * pu)
                self.Q[item_idx, :] += self.learning_rate * (error * pu - self.lambda_reg * qi)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self

    def predict(self):
        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], self.Q.shape[0], axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], self.P.shape[0], axis=0)

        self.R_predicted = self.global_mean + user_bias_matrix + item_bias_matrix + (self.P @ self.Q.T)
        return self.R_predicted