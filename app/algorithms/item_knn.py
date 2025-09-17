# app/algorithms/item_knn.py
import numpy as np
from .base import Recommender

class ItemKNNRecommender(Recommender):
    def __init__(self, k=20, similarity_metric='cosine', min_support=2, shrinkage=0.0, **kwargs):
        super().__init__(k)
        self.name = "ItemKNN"
        self.k = k # Number of neighbors
        self.similarity_matrix = None
        self.train_data = None
        self.similarity_metric = similarity_metric
        self.min_support = min_support
        self.shrinkage = shrinkage

    def fit(self, R, progress_callback=None):
        self.train_data = R
        num_items = R.shape[1]
        self.similarity_matrix = np.zeros((num_items, num_items))

        if self.similarity_metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            self.similarity_matrix = cosine_similarity(R.T)
        elif self.similarity_metric == 'adjusted_cosine':
            user_means = R.mean(axis=1)
            # Center the ratings by subtracting the user's mean rating.
            # Users who haven't rated anything will have a mean of 0, so their ratings will be unchanged.
            # Handle the case of users with no ratings to avoid division by zero if we were to normalize.
            R_centered = R - np.where(user_means[:, np.newaxis] > 0, user_means[:, np.newaxis], 0)
            from sklearn.metrics.pairwise import cosine_similarity
            self.similarity_matrix = cosine_similarity(R_centered.T)
        elif self.similarity_metric == 'pearson':
            # np.corrcoef calculates the Pearson correlation coefficient.
            # Row-wise variables, so we need to transpose R.
            self.similarity_matrix = np.corrcoef(R.T)
            # corrcoef can return NaNs if a column has zero variance (e.g., all ratings for an item are the same)
            self.similarity_matrix = np.nan_to_num(self.similarity_matrix)


        # Co-rated counts, for min_support and shrinkage
        co_rated_counts = (self.train_data > 0).astype(float).T @ (self.train_data > 0).astype(float)

        if self.min_support > 0:
            self.similarity_matrix[co_rated_counts < self.min_support] = 0

        if self.shrinkage > 0:
            self.similarity_matrix = (co_rated_counts / (co_rated_counts + self.shrinkage)) * self.similarity_matrix


        # We don't want an item to be its own neighbor, so we set diagonal to 0
        np.fill_diagonal(self.similarity_matrix, 0)

        if progress_callback:
            progress_callback(1.0)
        return self

    def predict_for_user(self, user_ratings):
        # user_ratings is a vector of shape (num_items,)
        predictions = np.zeros_like(user_ratings, dtype=float)

        # Iterate through all items to make predictions
        for item_to_predict in range(len(user_ratings)):
            # Only predict for items the user hasn't rated
            if user_ratings[item_to_predict] == 0:

                rated_indices = np.where(user_ratings > 0)[0]

                # The similarity scores of the item to predict to the items the user has rated
                sims_to_rated_items = self.similarity_matrix[item_to_predict, rated_indices]

                # The ratings the user gave to those items
                ratings_of_rated_items = user_ratings[rated_indices]

                # Get top K neighbors
                if self.k > 0 and len(sims_to_rated_items) > self.k:
                    top_neighbors_indices = np.argsort(-np.abs(sims_to_rated_items))[:self.k]
                    sims_to_rated_items = sims_to_rated_items[top_neighbors_indices]
                    ratings_of_rated_items = ratings_of_rated_items[top_neighbors_indices]

                # Weighted sum of similarities
                numerator = sims_to_rated_items @ ratings_of_rated_items
                denominator = np.abs(sims_to_rated_items).sum() + 1e-8

                if denominator > 0:
                    predictions[item_to_predict] = numerator / denominator
        return predictions

    def predict(self):
        num_users, _ = self.train_data.shape
        predictions = np.zeros_like(self.train_data, dtype=float)

        for u in range(num_users):
            predictions[u, :] = self.predict_for_user(self.train_data[u, :])

        self.R_predicted = np.where(self.train_data > 0, self.train_data, predictions)
        return self.R_predicted