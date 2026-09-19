import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

def bucketize_to_labels(ratings):
    """
    Convert continuous ratings into integer labels:
    - 0 (Low): [1.0, 3.0]
    - 1 (Medium): (3.0, 4.0]
    - 2 (High): (4.0, 5.0]
    """
    labels = []
    for r in ratings:
        if r <= 3.0:
            labels.append(0)
        elif r <= 4.0:
            labels.append(1)
        else:
            labels.append(2)
    return np.array(labels)

def labels_to_names(labels):
    mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    return np.array([mapping[l] for l in labels])

def evaluate_classifier_buckets(y_true_labels, y_pred_labels, model_name="Classifier"):
    """
    Evaluate predicted labels against true labels.
    """
    true_names = labels_to_names(y_true_labels)
    pred_names = labels_to_names(y_pred_labels)
    
    # 1. Bucket Accuracy & Balanced Accuracy
    acc = accuracy_score(true_names, pred_names)
    b_acc = balanced_accuracy_score(true_names, pred_names)
    macro_f1 = f1_score(true_names, pred_names, average='macro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"Bucket Evaluation Report - {model_name}")
    print(f"{'='*50}")
    print(f"1. Bucket Accuracy   : {acc * 100:.2f}%")
    print(f"   Balanced Accuracy : {b_acc * 100:.2f}%")
    print(f"   Macro F1-Score    : {macro_f1:.4f}")
    
    # 2. Classification Report
    print("\n2. Classification Report:")
    classes = ['Low', 'Medium', 'High']
    print(classification_report(true_names, pred_names, labels=classes, zero_division=0))
    
    # 3. Confusion Matrix
    print("3. Confusion Matrix (Row: True, Col: Pred):")
    cm = confusion_matrix(true_names, pred_names, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)
    
    return acc, b_acc, macro_f1

def print_classifier_samples(df, num_samples=10, model_name="Classifier"):
    """
    Print a few random samples of the predictions with probabilities.
    df must contain: 'UserId', 'AttractionId', 'VisitMode', 'TrueRating', 'PredLabel', 'Prob_0', 'Prob_1', 'Prob_2'
    """
    print(f"\n--- Top {num_samples} Random Samples ({model_name}) ---")
    samples = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    true_labels = bucketize_to_labels(samples['TrueRating'])
    true_names = labels_to_names(true_labels)
    pred_names = labels_to_names(samples['PredLabel'])
    
    results = []
    for i, (_, row) in enumerate(samples.iterrows()):
        true_b = true_names[i]
        pred_b = pred_names[i]
        correct = "Correct" if true_b == pred_b else "Incorrect"
        results.append({
            'User': int(row['UserId']),
            'Attraction': int(row['AttractionId']),
            'VisitMode': int(row['VisitMode']),
            'True_Rating': f"{row['TrueRating']:.2f}",
            'Pred_Rating': f"{row['PredRating']:.2f}",
            'True_Bucket': true_b,
            'Pred_Bucket': pred_b,
            'Prob_Low': f"{row['Prob_0']:.2f}",
            'Prob_Medium': f"{row['Prob_1']:.2f}",
            'Prob_High': f"{row['Prob_2']:.2f}",
            'Result': correct
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
