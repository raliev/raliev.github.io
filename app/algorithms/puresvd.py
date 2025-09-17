# app/algorithms/puresvd.py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .base import Recommender

class PureSVDRecommender(Recommender):
    def __init__(self, k, **kwargs):
        super().__init__(k)
        self.name = "PureSVD"
        self.svd = TruncatedSVD(n_components=self.k, random_state=42)
        self.sigma = None

    def fit(self, R, progress_callback=None):
        self.svd.fit(R)
        self.P = self.svd.transform(R) / self.svd.singular_values_
        self.sigma = np.diag(self.svd.singular_values_)
        self.Q = self.svd.components_.T

        if progress_callback:
            progress_callback(1.0)
        return self