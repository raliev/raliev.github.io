# app/algorithms/ncf.py
import numpy as np
from .base import Recommender

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, k, model_type='NeuMF'):
        super(NCFModel, self).__init__()
        self.model_type = model_type
        self.use_gmf = model_type in ['GMF', 'NeuMF']
        self.use_mlp = model_type in ['NCF', 'NeuMF']

        if self.use_gmf:
            self.gmf_user_embedding = nn.Embedding(num_users, k)
            self.gmf_item_embedding = nn.Embedding(num_items, k)

        if self.use_mlp:
            self.mlp_user_embedding = nn.Embedding(num_users, k)
            self.mlp_item_embedding = nn.Embedding(num_items, k)
            self.mlp_layers = nn.Sequential(
                nn.Linear(2 * k, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU()
            )

        if model_type == 'NeuMF':
            self.final_layer = nn.Linear(16 + k, 1)
        elif model_type == 'GMF':
            self.final_layer = nn.Linear(k, 1)
        else: # NCF (MLP only)
            self.final_layer = nn.Linear(16, 1)

    def forward(self, user_indices, item_indices):
        outputs = []
        if self.use_gmf:
            gmf_user_embed = self.gmf_user_embedding(user_indices)
            gmf_item_embed = self.gmf_item_embedding(item_indices)
            gmf_output = gmf_user_embed * gmf_item_embed
            outputs.append(gmf_output)

        if self.use_mlp:
            mlp_user_embed = self.mlp_user_embedding(user_indices)
            mlp_item_embed = self.mlp_item_embedding(item_indices)
            mlp_input = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
            mlp_output = self.mlp_layers(mlp_input)
            outputs.append(mlp_output)

        concat = torch.cat(outputs, dim=-1)
        logits = self.final_layer(concat)
        return torch.sigmoid(logits)

class NCFRecommender(Recommender):
    def __init__(self, k, model_type='NeuMF', epochs=10, batch_size=64, learning_rate=0.001, **kwargs):
        super().__init__(k)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch' to use this recommender.")

        self.name = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.num_users = None
        self.num_items = None

    def fit(self, R, progress_callback=None):
        self.num_users, self.num_items = R.shape

        # Prepare training data with negative sampling
        user_ids, item_ids = R.nonzero()
        labels = np.ones(len(user_ids), dtype=np.float32)

        num_neg_samples = 4
        neg_user_ids, neg_item_ids = [], []
        rated_items_map = {u: set(R[u, :].nonzero()[0]) for u in range(self.num_users)}

        for u, items in rated_items_map.items():
            for _ in range(len(items) * num_neg_samples):
                j = np.random.randint(self.num_items)
                while j in items:
                    j = np.random.randint(self.num_items)
                neg_user_ids.append(u)
                neg_item_ids.append(j)

        all_user_ids = torch.LongTensor(np.concatenate([user_ids, np.array(neg_user_ids)]))
        all_item_ids = torch.LongTensor(np.concatenate([item_ids, np.array(neg_item_ids)]))
        all_labels = torch.FloatTensor(np.concatenate([labels, np.zeros(len(neg_user_ids), dtype=np.float32)]))

        dataset = TensorDataset(all_user_ids, all_item_ids, all_labels.view(-1, 1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NCFModel(self.num_users, self.num_items, self.k, self.name)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for u, i, l in dataloader:
                optimizer.zero_grad()
                predictions = self.model(u, i)
                loss = criterion(predictions, l)
                loss.backward()
                optimizer.step()
            if progress_callback:
                progress_callback((epoch + 1) / self.epochs)
        return self

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            all_user_ids = torch.arange(self.num_users).long()
            all_item_ids = torch.arange(self.num_items).long()

            # Create all (user, item) pairs
            users_grid, items_grid = torch.meshgrid(all_user_ids, all_item_ids, indexing='ij')
            user_input = users_grid.flatten()
            item_input = items_grid.flatten()

            predictions = self.model(user_input, item_input)
            self.R_predicted = predictions.view(self.num_users, self.num_items).numpy()

        return self.R_predicted