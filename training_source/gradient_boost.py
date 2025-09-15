import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

# Gradient boost main body 
def gradient_boost(input_training_file, input_validation_file, out_put_txt="path\to\your\out_put_file_name.txt"): # change your output path here
    df_train = pd.read_csv(input_training_file)
    df_validate = pd.read_csv(input_validation_file)
    print("---------------✔️ Loaded CSV files.------------------")
    features = [col for col in df_train.columns if col not in ["password", "label"]]
    X_train = df_train[features]        

    # select features 
    features = [col for col in df_train.columns if col not in ["password", "label"]]
    X_train = df_train[features].copy()
    X_val   = df_validate[features].copy() 
    
    # keep only numeric feautues 
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 3) Encode label
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["label"].astype(str))
    y_val   = label_encoder.transform(df_validate["label"].astype(str))
    classes = label_encoder.classes_

    # 4) Model: HistGradientBoosting 
    model = HistGradientBoostingClassifier(
        learning_rate=0.03,          # để nhó giúp tổng quát tốt hơn
        max_iter=1200,               # điều chỉnh số vòng boost
        max_leaf_nodes=63,           # độ sâu của cây 
        min_samples_leaf=25,         # 20–30 giúp tránh overfit
        l2_regularization=0.2,      
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,         
        scoring="f1_macro",          
        random_state=42,
        verbose=1
    )
    
    print("--------------------🚀 Training model...-------------------")
    model.fit(X_train, y_train)
    print("--------------------✔️ Model trained.---------------------")

    # 5) Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    cm       = confusion_matrix(y_val, y_pred)

    TP_total      = np.trace(cm)
    total_samples = np.sum(cm)
    FP_array      = np.sum(cm, axis=0) - np.diag(cm)
    FN_array      = np.sum(cm, axis=1) - np.diag(cm)
    TN_array      = total_samples - (FP_array + FN_array + np.diag(cm))

    TP = TP_total
    FP = FP_array.sum()
    FN = FN_array.sum()
    TN = TN_array.sum()

    TPR = TP / (TP + FN + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    TNR = TN / (TN + FP + 1e-9)
    FNR = FN / (FN + TP + 1e-9)

    report = classification_report(y_val, y_pred, target_names=classes)

    # 6) Save out put results
    with open(out_put_txt, 'w', encoding='utf-8') as f:
        f.write("=== Features Used (ordered) ===\n")
        f.write(", ".join(features) + "\n\n")
        f.write("=== Classes ===\n")
        f.write(", ".join(classes) + "\n\n")
        f.write("=== Confusion Matrix ===\n")
        f.write("fair - medium - strong - weak\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")


    print("\n=== Confusion Matrix ===\n", cm)
    print("fair - medium - strong - weak\n")
    print("\n=== Classification Report ===\n", report)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")

        


# Main 
input_training_file = r"Path\to\your\training.csv" #change  to your path
input_validation_file = r"Path\to\your\validation.csv" # change to your path
gradient_boost(input_training_file, input_validation_file)

