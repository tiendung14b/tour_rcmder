import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

class TourismDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transactions = None
        self.users = None
        self.items = None
        self.cities = None
        
        # Mappings for LightFM (Internal ID to Original ID and vice versa)
        self.user_le = LabelEncoder()
        self.item_le = LabelEncoder()
        
    def load_data(self):
        """Load all necessary Excel files."""
        try:
            self.transactions = pd.read_excel(os.path.join(self.data_dir, "Transaction.xlsx"))
            self.users = pd.read_excel(os.path.join(self.data_dir, "User.xlsx"))
            self.items = pd.read_excel(os.path.join(self.data_dir, "Item.xlsx"))
            self.cities = pd.read_excel(os.path.join(self.data_dir, "City.xlsx"))
            print(f"Data loaded successfully:")
            print(f"- Transactions: {len(self.transactions)} rows")
            print(f"- Users: {len(self.users)} rows")
            print(f"- Items: {len(self.items)} rows")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

    def preprocess_data(self):
        """Clean missing values and encode IDs for LightFM."""
        # Clean missing
        self.transactions = self.transactions.dropna(subset=['UserId', 'AttractionId', 'Rating'])
        
        # We only keep items and users that exist in the transactions and the metadata tables
        valid_users = set(self.users['UserId'].unique())
        valid_items = set(self.items['AttractionId'].unique())
        
        self.transactions = self.transactions[
            (self.transactions['UserId'].isin(valid_users)) &
            (self.transactions['AttractionId'].isin(valid_items))
        ].copy()
        
        # Remove duplicate interactions by taking the average rating or max rating
        self.transactions = self.transactions.groupby(['UserId', 'AttractionId']).agg({
            'Rating': 'max',
            'VisitMode': 'first'
        }).reset_index()

        # Label encoding for contiguous IDs (0 to N-1)
        self.transactions['user_idx'] = self.user_le.fit_transform(self.transactions['UserId'])
        self.transactions['item_idx'] = self.item_le.fit_transform(self.transactions['AttractionId'])
        
        # Also encode the metadata DataFrames
        self.users = self.users[self.users['UserId'].isin(self.user_le.classes_)].copy()
        self.items = self.items[self.items['AttractionId'].isin(self.item_le.classes_)].copy()
        
        self.users['user_idx'] = self.user_le.transform(self.users['UserId'])
        self.items['item_idx'] = self.item_le.transform(self.items['AttractionId'])

    def get_interaction_matrix(self):
        """Build the interaction COO matrix (user x item)."""
        num_users = len(self.user_le.classes_)
        num_items = len(self.item_le.classes_)
        
        interactions = coo_matrix((
            self.transactions['Rating'].values,
            (self.transactions['user_idx'].values, self.transactions['item_idx'].values)
        ), shape=(num_users, num_items))
        
        return interactions

    def get_item_features(self):
        """
        Build item features matrix (item x features).
        We will use AttractionTypeId as a feature.
        """
        num_items = len(self.item_le.classes_)
        
        # Ensure items dataframe is sorted by item_idx
        sorted_items = self.items.sort_values('item_idx')
        
        # Create a simple feature: AttractionTypeId
        # In LightFM, we can provide an identity matrix + feature matrix
        # For simplicity, we just use one-hot encoding of AttractionTypeId
        type_le = LabelEncoder()
        type_idx = type_le.fit_transform(sorted_items['AttractionTypeId'])
        num_features = len(type_le.classes_)
        
        # The feature matrix format: [item_index, feature_index, weight]
        item_features = coo_matrix((
            np.ones(len(sorted_items)),
            (sorted_items['item_idx'].values, type_idx)
        ), shape=(num_items, num_features))
        
        return item_features
    
    def get_item_info(self, item_ids):
        """Return item details given original Item IDs"""
        # Fix dirty data mapping: 1->Bali, 2->Malang, 3->Yogyakarta
        city_mapping = {1: 3013, 2: 3156, 3: 3296}
        items_fixed = self.items.copy()
        items_fixed['AttractionCityId'] = items_fixed['AttractionCityId'].replace(city_mapping)
        
        merged = pd.merge(items_fixed, self.cities, left_on='AttractionCityId', right_on='CityId', how='left')
        return merged[merged['AttractionId'].isin(item_ids)]

    def get_tabular_data(self, df):
        """Prepare tabular data for LightGBM."""
        # Merge df with item features
        merged = df.merge(self.items[['item_idx', 'AttractionTypeId']], on='item_idx', how='left')
        
        # Merge with user features
        merged = merged.merge(self.users[['user_idx', 'ContenentId', 'RegionId', 'CountryId', 'CityId']], on='user_idx', how='left')
        
        # Define features for tabular model
        features = ['user_idx', 'item_idx', 'VisitMode', 'AttractionTypeId', 'ContenentId', 'RegionId', 'CountryId', 'CityId']
        
        # Handle VisitMode NaN if any
        merged['VisitMode'] = merged['VisitMode'].fillna(-1)
        
        X = merged[features].copy()
        y = merged['Rating'].values
        return X, y
