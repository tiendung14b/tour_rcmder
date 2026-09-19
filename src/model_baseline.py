import numpy as np
import pickle

class TourismRecommenderModel:
    def __init__(self, no_components=15, learning_rate=0.01, reg=0.1, **kwargs):
        self.no_components = no_components
        self.lr = learning_rate
        self.reg = reg
        
        self.global_bias = 0.0
        self.user_bias = None
        self.item_bias = None
        self.user_emb = None
        self.item_emb = None
        
    def fit(self, interactions, item_features=None, epochs=20, num_threads=2):
        """Train Matrix Factorization with Biases using Stochastic Gradient Descent (SGD)."""
        print(f"Training NumPy MF with Bias (epochs={epochs}, lr={self.lr}, reg={self.reg})...")
        # interactions is a sparse coo_matrix of shape (num_users, num_items)
        num_users, num_items = interactions.shape
        
        users = interactions.row
        items = interactions.col
        ratings = interactions.data
        
        # Initialize biases and embeddings
        self.global_bias = np.mean(ratings) if len(ratings) > 0 else 0.0
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        
        # Random normal initialization with small standard deviation
        np.random.seed(42)
        self.user_emb = np.random.normal(scale=1./self.no_components, size=(num_users, self.no_components))
        self.item_emb = np.random.normal(scale=1./self.no_components, size=(num_items, self.no_components))
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data for SGD
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)
            
            total_loss = 0.0
            
            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]
                
                # Predict
                dot_product = np.dot(self.user_emb[u], self.item_emb[i])
                pred = self.global_bias + self.user_bias[u] + self.item_bias[i] + dot_product
                
                # Error
                err = r - pred
                total_loss += err ** 2
                
                # Update Biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                
                # Update Embeddings
                u_emb_temp = self.user_emb[u].copy()
                i_emb_temp = self.item_emb[i].copy()
                
                self.user_emb[u] += self.lr * (err * i_emb_temp - self.reg * self.user_emb[u])
                self.item_emb[i] += self.lr * (err * u_emb_temp - self.reg * self.item_emb[i])
                
            rmse = np.sqrt(total_loss / len(ratings))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | RMSE: {rmse:.4f}")
                
        print("Training completed.")
        
    def predict_for_user(self, user_ids, item_ids, item_features=None, num_threads=2):
        """Predict scores for given users and items."""
        # Note: user_ids and item_ids are arrays of the same length
        scores = np.zeros(len(user_ids))
        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            dot_product = np.dot(self.user_emb[u], self.item_emb[i])
            scores[idx] = self.global_bias + self.user_bias[u] + self.item_bias[i] + dot_product
        return np.clip(scores, 1.0, 5.0)
    
    def save(self, filepath):
        """Save the model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'gb': self.global_bias,
                'ub': self.user_bias,
                'ib': self.item_bias,
                'ue': self.user_emb,
                'ie': self.item_emb
            }, f)
            
    def load(self, filepath):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.global_bias = data['gb']
            self.user_bias = data['ub']
            self.item_bias = data['ib']
            self.user_emb = data['ue']
            self.item_emb = data['ie']
