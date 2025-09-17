# algorithms/als.py
import numpy as np
from .base import Recommender

class ALSRecommender(Recommender):
    def __init__(self, k, iterations=10, lambda_reg=0.1):
        super().__init__(k)
        self.name = "ALS"
        self.iterations = iterations
        self.lambda_reg = lambda_reg

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        rated_mask = R > 0

        for i in range(self.iterations):
            for u in range(num_users):
                rated_indices = np.where(rated_mask[u, :])[0]
                if len(rated_indices) > 0:
                    Qu = self.Q[rated_indices, :]
                    Ru = R[u, rated_indices]
                    A = Qu.T @ Qu + self.lambda_reg * np.eye(self.k)
                    b = Qu.T @ Ru
                    self.P[u, :] = np.linalg.solve(A, b)

            for item_idx in range(num_items):
                rated_indices = np.where(rated_mask[:, item_idx])[0]
                if len(rated_indices) > 0:
                    Pi = self.P[rated_indices, :]
                    Ri = R[rated_indices, item_idx]
                    A = Pi.T @ Pi + self.lambda_reg * np.eye(self.k)
                    b = Pi.T @ Ri
                    self.Q[item_idx, :] = np.linalg.solve(A, b)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self