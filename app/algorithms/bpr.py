# algorithms/bpr.py
import numpy as np
from .base import Recommender

class BPRRecommender(Recommender):
    def __init__(self, k, iterations=1000, learning_rate=0.01, lambda_reg=0.01):
        super().__init__(k)
        self.name = "BPR"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(self, R, progress_callback=None):
        R_binary = (R > 0).astype(int)
        num_users, num_items = R_binary.shape
        self.P = np.random.normal(scale=0.1, size=(num_users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, self.k))
        positive_pairs = np.argwhere(R_binary > 0)

        for i in range(self.iterations):
            np.random.shuffle(positive_pairs)
            for u, item_i in positive_pairs:
                j = np.random.randint(num_items)
                while R_binary[u, j] > 0:
                    j = np.random.randint(num_items)

                p_u, q_i, q_j = self.P[u, :], self.Q[item_i, :], self.Q[j, :]
                x_uij = p_u @ q_i.T - p_u @ q_j.T
                sigmoid_x = 1 / (1 + np.exp(x_uij))

                self.P[u, :] += self.learning_rate * (sigmoid_x * (q_i - q_j) - self.lambda_reg * p_u)
                self.Q[item_i, :] += self.learning_rate * (sigmoid_x * p_u - self.lambda_reg * q_i)
                self.Q[j, :] += self.learning_rate * (sigmoid_x * -p_u - self.lambda_reg * q_j)

            if progress_callback:
                progress_callback((i + 1) / self.iterations)
        return self