#clayzzz

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from scipy.stats import mode

# random forest process
def random_forest(input_training_file, input_validation_file):
    # Read data
    df_train = pd.read_csv(input_training_file)
    df_validate = pd.read_csv(input_validation_file)

    # Extract features
    features = [col for col in df_train.columns if col not in ["password", "label"]] # remove password and label column
    #metrix features
    X_train = df_train[features]
    X_val = df_validate[features]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["label"])
    y_val = label_encoder.transform(df_validate["label"])

    # Build Random Forest manually
    base_tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42) # depth = how depth the model will learn, min sample leaf to reduce overfiting
    forest = []
    n_trees = 300 # forest size 

    for _ in tqdm(range(n_trees), desc="Training Random Forest", total=n_trees, dynamic_ncols=True):
        indices = np.random.choice(len(X_train), len(X_train), replace=True) #Bootstrap sample (1 dòng dữ liệu có thể lấy nhiều lần)
        n_features = int(len(features) * 0.6) # take randomly 60% features
        selected_features = np.random.choice(features, n_features, replace=False)
        X_bootstrap = X_train[selected_features].iloc[indices] # take features from bootstrap as the slected_features above 
        y_bootstrap = y_train[indices] # take features from bootstrap as the slected_features above 
        tree = clone(base_tree)
        tree.fit(X_bootstrap, y_bootstrap)
        forest.append((tree, selected_features))  # Save along with selected features

    # Prediction by majority voting
    def forest_predict(X_input): 
        preds = []
        for tree, selected in forest:  #loop through the forest
            preds.append(tree.predict(X_input[selected])) # dự đoán dùng đúng các cột đặc trưng mà cây đã dùng khi huấn luyện
        preds = np.array(preds)
        final_preds, _ = mode(preds, axis=0, keepdims=False) #voting (mỗi cột (tương ứng với 1 mẫu đầu vào), lấy giá trị xuất hiện nhiều nhất trong 300 dự đoán từ các cây.)
        return final_preds.flatten()

    # Predict on validation set
    y_pred = forest_predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # Predict new password sample
    new_pwd_feature = pd.DataFrame([{
        "length": 10,
        "count_lower": 6,
        "ratio_lower": 0.6,
        "count_upper": 2,
        "ratio_upper": 0.2,
        "count_digits": 1,
        "ratio_digits": 0.1,
        "count_special_chars": 1,
        "ratio_special_chars": 0.1,
        "char_diversity": 4,
        "common_phrase": 0
    }])
    predicted = forest_predict(new_pwd_feature)

    # Write results to file
    with open('...', 'w', encoding='utf-8') as f: # change result output in txt
        f.write("=== Confusion Matrix ===\n")
        cm = confusion_matrix(y_val, y_pred)
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_val, y_pred, target_names=label_encoder.classes_) + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"=> Prediction for new password strength: {label_encoder.inverse_transform(predicted)[0]}\n")
        
        # === Calculate overall TPR, FPR, TNR, FNR ===
        TP_total = np.trace(cm)
        total_samples = np.sum(cm)
        FP_total = np.sum(cm, axis=0) - np.diag(cm)
        FN_total = np.sum(cm, axis=1) - np.diag(cm)
        TN_total = total_samples - (FP_total + FN_total + np.diag(cm))

        TP = TP_total
        FP = FP_total.sum()
        FN = FN_total.sum()
        TN = TN_total.sum()

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")

    # Print to console
    print("=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    print(f"Accuracy: {accuracy:.4f}")
    print("\n=> Prediction for new password strength:", label_encoder.inverse_transform(predicted)[0])
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")


# Run
input_training_file = r"..." #change to your training input
input_validation_file = r"..." # change to your validate input
random_forest(input_training_file, input_validation_file)
