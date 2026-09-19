import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

from src.data_loader import TourismDataLoader
from src.model_baseline import TourismRecommenderModel
from src.model_advanced import AdvancedClassifier
from src.metrics import evaluate_classifier_buckets, print_classifier_samples, bucketize_to_labels

def main():
    # 1. Khởi tạo DataLoader và load dữ liệu
    data_dir = r"D:\Workspace\projects\komoberi\Tourism Dataset"
    loader = TourismDataLoader(data_dir=data_dir)
    
    print("Loading and preprocessing data...")
    loader.load_data()
    loader.preprocess_data()
    
    # 2. Chuẩn bị dữ liệu Train/Test
    print("Splitting train/test...")
    train_df, test_df = train_test_split(loader.transactions, test_size=0.2, random_state=42)
    
    # Chuẩn bị dữ liệu cho MF Baseline (Sparse Matrix)
    num_users = len(loader.user_le.classes_)
    num_items = len(loader.item_le.classes_)
    
    train_interactions = coo_matrix((
        train_df['Rating'].values,
        (train_df['user_idx'].values, train_df['item_idx'].values)
    ), shape=(num_users, num_items))
    
    # Chuẩn bị dữ liệu cho LightGBM Advanced (Tabular)
    X_train, y_train = loader.get_tabular_data(train_df)
    X_test, y_test = loader.get_tabular_data(test_df)
    
    # Target Encoding cho Classifier (0=Low, 1=Medium, 2=High)
    y_train_labels = bucketize_to_labels(y_train)
    y_test_labels = bucketize_to_labels(y_test)
    
    # ==================================================
    # 3. Huấn luyện và Đánh giá: BASELINE (NumPy MF Bias)
    # ==================================================
    print("\n" + "*"*50)
    print("1. Training Baseline Model (Matrix Factorization)")
    print("*"*50)
    mf_model = TourismRecommenderModel(no_components=15, learning_rate=0.05, reg=0.1)
    mf_model.fit(train_interactions, epochs=20)
    
    # Dự đoán (Regression) -> Convert sang Labels
    y_pred_mf_reg = mf_model.predict_for_user(test_df['user_idx'].values, test_df['item_idx'].values)
    y_pred_mf_labels = bucketize_to_labels(y_pred_mf_reg)
    acc_mf, b_acc_mf, f1_mf = evaluate_classifier_buckets(y_test_labels, y_pred_mf_labels, model_name="MF Baseline")
    
    # ==================================================
    # 4. Huấn luyện và Đánh giá: ADVANCED CLASSIFIER (LightGBM)
    # ==================================================
    print("\n" + "*"*50)
    print("2. Training Advanced Model (Tabular Classifier)")
    print("*"*50)
    lgb_model = AdvancedClassifier()
    lgb_model.fit(X_train, y_train_labels)
    
    # Dự đoán Labels và Xác suất (Probabilities)
    y_pred_lgb_labels = lgb_model.predict(X_test)
    y_pred_lgb_probs = lgb_model.predict_proba(X_test)
    
    acc_lgb, b_acc_lgb, f1_lgb = evaluate_classifier_buckets(y_test_labels, y_pred_lgb_labels, model_name=f"Advanced ({lgb_model.model_name})")
    
    # ==================================================
    # 5. Bảng So sánh và In Mẫu
    # ==================================================
    print("\n" + "*"*50)
    print("3. SUMMARY AND COMPARISON")
    print("*"*50)
    print(f"Metrics             | MF Baseline | Advanced Classifier")
    print(f"--------------------|-------------|--------------------")
    print(f"Bucket Accuracy     | {acc_mf*100:9.2f}% | {acc_lgb*100:17.2f}%")
    print(f"Balanced Accuracy   | {b_acc_mf*100:9.2f}% | {b_acc_lgb*100:17.2f}%")
    print(f"Macro F1-Score      | {f1_mf:9.4f}  | {f1_lgb:17.4f}")
    
    # Ghép thông tin ID gốc và xác suất để in mẫu
    test_samples_df = pd.DataFrame({
        'UserId': loader.user_le.inverse_transform(test_df['user_idx']),
        'AttractionId': loader.item_le.inverse_transform(test_df['item_idx']),
        'VisitMode': test_df['VisitMode'],
        'TrueRating': y_test,
        'PredRating': y_pred_lgb_probs[:, 0] * 2.0 + y_pred_lgb_probs[:, 1] * 3.5 + y_pred_lgb_probs[:, 2] * 4.5,
        'PredLabel': y_pred_lgb_labels,
        'Prob_0': y_pred_lgb_probs[:, 0],
        'Prob_1': y_pred_lgb_probs[:, 1],
        'Prob_2': y_pred_lgb_probs[:, 2]
    })
    
    print_classifier_samples(test_samples_df, num_samples=10, model_name=lgb_model.model_name)

if __name__ == "__main__":
    main()
