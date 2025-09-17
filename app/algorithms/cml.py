# app/algorithms/cml.py
import numpy as np
from .base import Recommender

class CMLRecommender(Recommender):
    def __init__(self, k, iterations=100, learning_rate=0.01, lambda_reg=0.01, margin=0.5):
        super().__init__(k)
        self.name = "CML"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.margin = margin

    def fit(self, R, progress_callback=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        for i in range(self.iterations):
            for u, item_i in np.argwhere(R > 0):
                # Negative sampling
                item_j = np.random.randint(num_items)
                while R[u, item_j] > 0:
                    item_j = np.random.randint(num_items)

                p_u = self.P[u, :]
                q_i = self.Q[item_i, :]
                q_j = self.Q[item_j, :]

                dist_ui = np.linalg.norm(p_u - q_i)
                dist_uj = np.linalg.norm(p_u - q_j)

                if (dist_ui - dist_uj + self.margin) > 0:
                    # Update vectors
                    self.P[u, :] -= self.learning_rate * ((p_u - q_i) - (p_u - q_j))
                    self.Q[item_i, :] -= self.learning_rate * (-(p_u - q_i))
                    self.Q[item_j, :] -= self.learning_rate * (p_u - q_j)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self

    def predict(self):
        # For CML, prediction is based on distance, so we'll convert it back to a score
        # A smaller distance means a higher score
        self.R_predicted = -np.linalg.norm(self.P[:, np.newaxis, :] - self.Q, axis=2)
        return self.R_predicted