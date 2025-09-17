# algorithms/als_improved.py
import numpy as np
from .base import Recommender

class ALSImprovedRecommender(Recommender):
    # --- CHANGE: Added a separate regularization parameter for biases ---
    def __init__(self, k, iterations=10, lambda_reg=0.1, lambda_biases=10.0):
        super().__init__(k)
        self.name = "ALS (Improved)"
        self.iterations = iterations
        # lambda_reg will now be treated as lambda for factors
        self.lambda_factors = lambda_reg
        self.lambda_biases = lambda_biases
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape

        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)

        self.global_mean = R[R > 0].mean() if np.any(R > 0) else 0

        rated_mask = R > 0

        for i in range(self.iterations):
            # --- Update User Factors and Biases ---
            for u in range(num_users):
                rated_indices = np.where(rated_mask[u, :])[0]
                if len(rated_indices) > 0:
                    Qu = self.Q[rated_indices, :]
                    Ru = R[u, rated_indices] - self.global_mean - self.item_bias[rated_indices]

                    # --- CHANGE: Use lambda_factors for factor regularization ---
                    A = Qu.T @ Qu + self.lambda_factors * np.eye(self.k)
                    b = Qu.T @ Ru
                    self.P[u, :] = np.linalg.solve(A, b)

                    # --- CHANGE: Use lambda_biases for bias regularization ---
                    self.user_bias[u] = np.sum(Ru - self.P[u, :] @ Qu.T) / (len(rated_indices) + self.lambda_biases)

            # --- Update Item Factors and Biases ---
            for item_idx in range(num_items):
                rated_indices = np.where(rated_mask[:, item_idx])[0]
                if len(rated_indices) > 0:
                    Pi = self.P[rated_indices, :]
                    Ri = R[rated_indices, item_idx] - self.global_mean - self.user_bias[rated_indices]

                    # --- CHANGE: Use lambda_factors for factor regularization ---
                    A = Pi.T @ Pi + self.lambda_factors * np.eye(self.k)
                    b = Pi.T @ Ri
                    self.Q[item_idx, :] = np.linalg.solve(A, b)

                    # --- CHANGE: Use lambda_biases for bias regularization ---
                    self.item_bias[item_idx] = np.sum(Ri - self.Q[item_idx, :] @ Pi.T) / (len(rated_indices) + self.lambda_biases)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self

    def predict(self):
        """
        Generates the matrix of predicted ratings, including bias terms.
        """
        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], self.Q.shape[0], axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], self.P.shape[0], axis=0)

        self.R_predicted = self.global_mean + user_bias_matrix + item_bias_matrix + (self.P @ self.Q.T)
        return self.R_predicted