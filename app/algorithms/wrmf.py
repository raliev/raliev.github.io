# app/algorithms/wrmf.py
import numpy as np
from .base import Recommender

class WRMFRecommender(Recommender):
    def __init__(self, k, iterations=10, lambda_reg=0.1, alpha=40):
        super().__init__(k)
        self.name = "WRMF"
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.alpha = alpha

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        # Create confidence matrix
        C = 1 + self.alpha * R

        for i in range(self.iterations):
            # Update user factors
            for u in range(num_users):
                Cu = np.diag(C[u, :])
                A = self.Q.T @ Cu @ self.Q + self.lambda_reg * np.eye(self.k)
                b = self.Q.T @ Cu @ (R[u, :] > 0).astype(int)
                self.P[u, :] = np.linalg.solve(A, b)

            # Update item factors
            for item_idx in range(num_items):
                Ci = np.diag(C[:, item_idx])
                A = self.P.T @ Ci @ self.P + self.lambda_reg * np.eye(self.k)
                b = self.P.T @ Ci @ (R[:, item_idx] > 0).astype(int)
                self.Q[item_idx, :] = np.linalg.solve(A, b)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self