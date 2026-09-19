try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False

import numpy as np

class AdvancedClassifier:
    def __init__(self, **kwargs):
        if HAS_LGBM:
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=150,
                learning_rate=0.03,
                max_depth=6,
                min_child_samples=20,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
            self.model_name = "LightGBM Classifier"
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=100, 
                learning_rate=0.05, 
                class_weight='balanced',
                random_state=42
            )
            self.model_name = "HistGBM Classifier (Fallback)"
        
    def fit(self, X_train, y_train):
        """y_train should be categorical: 0, 1, 2"""
        print(f"Training Advanced Classifier: {self.model_name}...")
        self.model.fit(X_train, y_train)
        print("Training completed.")
        
    def print_feature_importance(self, features_names, top_n=10):
        if HAS_LGBM and isinstance(self.model, lgb.LGBMClassifier):
            importance = self.model.feature_importances_
            # Sort descending
            indices = np.argsort(importance)[::-1]
            print(f"\n--- Top {top_n} Feature Importance ---")
            for i in range(min(top_n, len(features_names))):
                idx = indices[i]
                print(f"{i+1}. {features_names[idx]:<25} : {importance[idx]}")
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
