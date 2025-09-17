# app/algorithms/funksvd.py
import numpy as np
from .base import Recommender

class FunkSVDRecommender(Recommender):
    def __init__(self, k, iterations=100, learning_rate=0.005, lambda_reg=0.02):
        super().__init__(k)
        self.name = "FunkSVD"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        for i in range(self.iterations):
            for u, item_idx in np.argwhere(R > 0):
                error = R[u, item_idx] - self.P[u, :] @ self.Q[item_idx, :].T
                p_u = self.P[u, :]
                q_i = self.Q[item_idx, :]

                self.P[u, :] += self.learning_rate * (error * q_i - self.lambda_reg * p_u)
                self.Q[item_idx, :] += self.learning_rate * (error * p_u - self.lambda_reg * q_i)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self