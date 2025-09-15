import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def svm_classification(input_training_file, input_validation_file):
    df_train = pd.read_csv(input_training_file)
    df_validate = pd.read_csv(input_validation_file)
    print("---------------✔️ Loaded CSV files.------------------")

    features = [col for col in df_train.columns if col not in ["password", "label"]]
    X_train = df_train[features]
    X_val   = df_validate[features]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["label"])
    y_val   = label_encoder.transform(df_validate["label"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    model = LinearSVC(C=1.0, max_iter=1000, verbose=1, random_state=42) # change model parameters here
    print("--------------------🚀 Training model...-------------------")
    model.fit(X_train_scaled, y_train)
    print("--------------------✔️ Model trained.---------------------")

    
    y_pred = model.predict(X_val_scaled)

    accuracy = accuracy_score(y_val, y_pred)
    cm       = confusion_matrix(y_val, y_pred)

    TP_total     = np.trace(cm)
    total_samples= np.sum(cm)
    FP_array     = np.sum(cm, axis=0) - np.diag(cm)
    FN_array     = np.sum(cm, axis=1) - np.diag(cm)
    TN_array     = total_samples - (FP_array + FN_array + np.diag(cm))

    TP = TP_total
    FP = FP_array.sum()
    FN = FN_array.sum()
    TN = TN_array.sum()

    TPR = TP / (TP + FN + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    TNR = TN / (TN + FP + 1e-9)
    FNR = FN / (FN + TP + 1e-9)

    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)

    with open('change\to\your\output_result_path.txt', 'w', encoding='utf-8') as f: # change to your output path here
        f.write("=== Confusion Matrix ===\n")
        f.write("fair - medium - strong - weak\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n") 
        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")

    
    print("\n=== Confusion Matrix ===\n", cm)
    print("\n=== Classification Report ===\n", report)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")


# Run
input_training_file = r"change\to\your\path\training.csv" #change  to your path
input_validation_file = r"change\to\your\path\validation.csv" # change to your path
svm_classification(input_training_file, input_validation_file)
