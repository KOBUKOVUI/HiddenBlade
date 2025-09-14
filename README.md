## 📦 Phần I: Thu thập dữ liệu

Dữ liệu mật khẩu được tổng hợp từ nhiều nguồn lớn:

- **Rockyou.txt**: 14 triệu mật khẩu

- **LinkedIn leak**: 20 triệu mật khẩu

- **Hotmail**: 1 triệu mật khẩu

- **Thực tế** (Việt hóa, sưu tầm): 1 triệu mật khẩu

## 🧹 Phần II: Tiền xử lý dữ liệu

- **Lọc theo độ dài**: Chỉ giữ lại mật khẩu từ 4 đến 72 ký tự (zxcvbn chỉ phân tích tối đa 72 ký tự).

- **Lọc trùng lặp**: Loại bỏ các mật khẩu bị lặp.

- **Loại bỏ mật khẩu không chuẩn**:
  
  - Chứa ký tự lạ như `\n`, hex, hoặc mã hóa URL.
  
  - Có cấu trúc email: `@gmail`, `@hotmail`,…
  
  - Chứa tên miền `.com`

- **Lọc nâng cao với Rockyou**:
  
  - Giữ ~1.5 triệu mật khẩu yếu và ~600k mạnh.
  
  - Tập còn lại chỉ giữ mật khẩu mạnh và thực tế hơn.
  
  - Lấy ngẫu nhiên 25% từ Rockyou, tập khác giữ nguyên để cân bằng nhãn.

## 🏷️ Phần III: Gán nhãn và đánh giá

- Sử dụng **zxcvbn** để gán điểm strength (`0–4`), tính **entropy**.

- Mật khẩu **fair** và **strong** được xác định kỹ hơn qua entropy.

- Sau oversampling, giữ tập dữ liệu cân bằng với 4 lớp: `weak`, `fair`, `medium`, `strong`.
  
  ==================================================
  
  weak: 1,734,734 -> 27.37%
  
  medium: 1,717,970 -> 27.14%
  
  fair: 1,346,700 -> 21.25%
  
  strong 1,535,966 -> 24.24%
  
  🔢 ---> TOTAL: 6,337,370

==================================================

## 🧪 Phần IV: Trích xuất đặc trưng

Các đặc trưng chính được sử dụng trong mô hình:

| Đặc trưng                                                           | Mô tả                                             |
| ------------------------------------------------------------------- | ------------------------------------------------- |
| `length`                                                            | Độ dài mật khẩu                                   |
| `entropy_length`                                                    | Điểm entropy                                      |
| `count_lower`, `count_upper`, `count_digits`, `count_special_chars` | Số lượng ký tự thường, in hoa, số, đặc biệt       |
| `ratio_*`                                                           | Tỉ lệ ký tự thường/in hoa/số/đặc biệt trên độ dài |
| `char_diversity`                                                    | Đa dạng loại ký tự sử dụng                        |
| `common_phrase`                                                     | Có thuộc các cụm phổ biến không                   |

## 🧠 Phần V: Huấn luyện mô hình

### 1. Random Forest

Mô hình **Random Forest** được triển khai với các tham số sau:

- **Số cây (trees)**: 300
- **Tỉ lệ đặc trưng được trích xuất**: 60%
- **Độ sâu tối đa (max depth)**: 12
- **Kích thước tập huấn luyện**: 5.000.000 mẫu
- **Kích thước tập kiểm thử (validation)**: 1.280.000 mẫu

Kết quả đánh giá:

- **Accuracy**: `0.6961`
- **Precision / Recall / F1-score theo từng lớp**:
  - `Fair`: Precision `0.67`, Recall `0.55`, F1 `0.61`
  - `Medium`: Precision `0.69`, Recall `0.89`, F1 `0.78`
  - `Strong`: Precision `0.62`, Recall `0.75`, F1 `0.68`
  - `Weak`: Precision `0.87`, Recall `0.59`, F1 `0.70`
- **Chỉ số tổng thể**:
  - **TPR**: `0.6961`
  - **FPR**: `0.1013`
  - **TNR**: `0.8987`
  - **FNR**: `0.3039`

🧾 **Nhận xét**: Mô hình thể hiện khả năng phân loại tốt, đặc biệt ở các lớp “medium” và “strong”. Tuy nhiên, nhầm lẫn ở lớp “fair” và “weak” vẫn cần được cải thiện. Có thể áp dụng thêm kỹ thuật như boosting hoặc tuning thêm các đặc trưng để cải thiện độ chính xác.

### 2.1 Logistic regression: OvR

**Nhận xét tổng quan**  
Mô hình Logistic Regression OvR cho kết quả độ chính xác 60.65% trên 1.28 triệu mẫu, với hiệu năng tốt ở hai thái cực “weak” và “strong” nhưng hạn chế rõ rệt ở hai lớp trung gian “fair” và “medium”. Điều này cho thấy ranh giới giữa các mức “trung bình” chưa được đặc trưng hoá đầy đủ, cần cải thiện feature hoặc chuyển chiến lược softmax để nâng cao chất lượng phân loại.

---

**Nhận xét ngắn gọn**  
– Mô hình phân biệt tốt hai lớp “strong” (recall = 0.85) và “weak” (recall = 0.77), F1‐score đều 0.68.  
– Khó khăn ở hai lớp trung gian “fair” (recall = 0.40, F1 = 0.50) và “medium” (recall = 0.41, F1 = 0.51).  
– Xác suất “fair” dễ bị nhầm thành “strong” và “medium” dễ bị nhầm thành “weak”.

---

**Các thông số chính**

- **Accuracy:** 0.6065

- **Fair:** precision = 0.65 | recall = 0.40 | f1-score = 0.50

- **Medium:** precision = 0.67 | recall = 0.41 | f1-score = 0.51

- **Strong:** precision = 0.56 | recall = 0.85 | f1-score = 0.68

- **Weak:** precision = 0.60 | recall = 0.77 | f1-score = 0.68

- **Macro avg:** precision = 0.62 | recall = 0.61 | f1-score = 0.59

- **Weighted avg:** precision = 0.62 | recall = 0.61 | f1-score = 0.59

### 2.2 Logistic Regression: Softmax

Mô hình Softmax đã cải thiện rõ rệt so với OvR, đạt **67.23%** accuracy (tăng gần 6.6 điểm phần trăm). Mọi lớp đều có mức precision, recall và F1‐score đồng đều hơn, cho thấy giải pháp đa lớp (“multinomial”) tối ưu hoá tốt hơn ranh giới giữa các mức độ.

---

**Chi tiết theo lớp**

- **fair**:
  
  - Precision = 0.66 (tăng nhẹ)
  
  - Recall = 0.55 (tăng từ 0.40 → 0.55)
  
  - F1 = 0.60 (tăng từ 0.50 → 0.60)  
    → Ranh “fair”/“strong” đã rõ hơn, giảm nhầm lẫn sang “strong”.

- **medium**:
  
  - Precision = 0.74 (tăng từ 0.67)
  
  - Recall = 0.65 (tăng từ 0.41)
  
  - F1 = 0.69 (tăng từ 0.51)  
    → Lớp trung cấp tiếp tục dễ nhận diện hơn, cả sensitivity và specificity đều tăng.

- **strong**:
  
  - Precision = 0.61 (tăng từ 0.56)
  
  - Recall = 0.74 (giảm từ 0.85)
  
  - F1 = 0.67 (giảm nhẹ từ 0.68)  
    → Ít “false alarms” hơn (precision cao hơn), chấp nhận giảm một chút recall để cân bằng với lớp khác.

- **weak**:
  
  - Precision = 0.70 (tăng từ 0.60)
  
  - Recall = 0.75 (giảm nhẹ từ 0.77)
  
  - F1 = 0.72 (tăng từ 0.68)  
    → Dự đoán “weak” chính xác và bao phủ tốt, F1 cải thiện.

---

**Các chỉ số tổng thể**

- **Accuracy:** 0.6723

- **Macro avg:** precision = 0.68 | recall = 0.67 | F1 = 0.67

- **Weighted avg:** precision = 0.68 | recall = 0.67 | F1 = 0.67

--- 

### 3 SVM 
Model Evaluation Report (Linear SVM)
Dataset & Setup

Tổng số mẫu: 1,280,000

Các lớp (classes): fair, medium, strong, weak

Thuật toán: Linear Support Vector Machine (LinearSVC, C=1.0, max_iter=1000)

✅ Kết quả tổng quan

Accuracy (Độ chính xác): 60.26%

Macro Average (trung bình trên tất cả lớp):

Precision: 0.64

Recall: 0.60

F1-score: 0.57

🔎 Classification Report chi tiết
Class	Precision	Recall	F1-score	Support
fair	0.65	0.35	0.45	320,000
medium	0.76	0.32	0.45	320,000
strong	0.55	0.88	0.67	320,000
weak	0.60	0.86	0.71	320,000

👉 Điểm nổi bật:

Medium có precision khá cao (0.76), nhưng recall thấp (0.32) → model thường đoán "medium" đúng, nhưng bỏ sót nhiều trường hợp thực sự là "medium".

Strong và Weak có recall rất cao (0.88 và 0.86), nghĩa là model nhận diện tốt các trường hợp thực sự là "strong/weak".

Fair bị nhận diện kém (recall chỉ 0.35), nghĩa là nhiều mẫu fair bị nhầm thành lớp khác.

📉 Confusion Matrix (Tóm tắt nhầm lẫn)
Thực tế ↓ / Dự đoán →	fair	medium	strong	weak
fair	112,029	61	207,910	0
medium	15,511	103,354	15,370	185,765
strong	39,571	415	280,014	0
weak	5,659	31,913	6,493	275,935

👉 Điểm đáng chú ý:

Rất nhiều fair bị nhầm sang strong (~208k).

Một lượng lớn medium bị nhầm sang weak (~186k).

Strong đa phần được dự đoán đúng (280k/320k).

Weak cũng được dự đoán khá chính xác (276k/320k).

📈 Chỉ số chung (Overall Metrics)

True Positive Rate (TPR / Recall): 60.26%

False Positive Rate (FPR): 13.25%

True Negative Rate (TNR / Specificity): 86.75%

False Negative Rate (FNR): 39.74%

👉 Nghĩa là:

Model dự đoán đúng khoảng 6/10 trường hợp.

Có khả năng phân biệt tương đối ổn (TNR ~87%), nhưng vẫn bỏ sót gần 40% mẫu (FNR cao).

📌 Kết luận & Hướng cải thiện

SVM cho kết quả tệ hơn Gradient Boosting (60% vs ~70% accuracy).

Sự nhầm lẫn chủ yếu giữa fair ↔ strong và medium ↔ weak.

Hướng cải thiện:

Thử kernel phi tuyến (RBF, polynomial) thay vì Linear SVM.

Điều chỉnh C và tăng max_iter để hội tụ tốt hơn.

Cân bằng dữ liệu hoặc sử dụng class_weight="balanced" trong SVM.

So sánh với các thuật toán boosting (XGBoost, LightGBM) để xem sự khác biệt.



### 4 Gradient boost

Model Evaluation Report (Gradient Boosting)
Dataset & Setup

Tổng số mẫu: 1,280,000

Các lớp (classes): fair, medium, strong, weak

Thuật toán: Gradient Boosting Classifier

✅ Kết quả tổng quan

Accuracy (Độ chính xác): 69.82%

Macro Average (trung bình trên tất cả lớp):

Precision: 0.71

Recall: 0.70

F1-score: 0.70

🔎 Classification Report chi tiết
Class	Precision	Recall	F1-score	Support
fair	0.65	0.62	0.63	320,000
medium	0.69	0.89	0.78	320,000
strong	0.64	0.70	0.67	320,000
weak	0.86	0.59	0.70	320,000

Precision cao nhất: lớp weak (0.86) → khi model dự đoán "weak" thì khá đáng tin.

Recall cao nhất: lớp medium (0.89) → model bắt được gần hết các trường hợp "medium".

Recall thấp nhất: lớp weak (0.59) → nhiều trường hợp "weak" bị nhầm sang lớp khác.

Các lớp fair và strong có hiệu năng trung bình, cần cải thiện thêm.

📉 Confusion Matrix (Tóm tắt nhầm lẫn)
Thực tế ↓ / Dự đoán →	fair	medium	strong	weak
fair	197,249	0	122,748	3
medium	4,157	283,265	2,848	29,730
strong	96,623	0	223,376	1
weak	3,319	124,408	2,475	189,798

👉 Điểm đáng chú ý:

Rất nhiều mẫu fair bị nhầm sang strong (~122k).

Một phần lớn weak bị nhầm thành medium (~124k).

📈 Chỉ số chung (Overall Metrics)

True Positive Rate (TPR / Recall): 69.82%

False Positive Rate (FPR): 10.06%

True Negative Rate (TNR / Specificity): 89.94%

False Negative Rate (FNR): 30.18%

👉 Nghĩa là:

Model dự đoán đúng khoảng 70% các trường hợp thực tế.

Khả năng phân biệt giữa positive/negative tương đối tốt (TNR ~90%), nhưng vẫn bỏ sót khoảng 30% mẫu (FNR).

📌 Kết luận & Hướng cải thiện

Model hoạt động khá ổn định, accuracy gần 70%, nhưng vẫn có nhiều nhầm lẫn giữa các lớp fair ↔ strong và weak ↔ medium.

Có thể cải thiện bằng cách:

Tăng cân bằng dữ liệu (class rebalancing / oversampling / focal loss).

Điều chỉnh hyperparameter cho Gradient Boosting (learning rate, max depth, n_estimators).

Thử các thuật toán boosting khác như XGBoost / LightGBM / CatBoost để so sánh.

### 5 MLP 
Model Evaluation Report (MLP Classifier)
Dataset & Setup

Tổng số mẫu: 1,280,000

Các lớp (classes): fair, medium, strong, weak

Đặc trưng sử dụng (features):
entropy, length, count_lower, ratio_lower, count_upper, ratio_upper, count_digits, ratio_digits, count_special_chars, ratio_special_chars, char_diversity, common_phrase

Thuật toán: Multi-Layer Perceptron (MLP Neural Network)

Thời gian huấn luyện: ~1684 giây (~28 phút)

✅ Kết quả tổng quan

Accuracy (Độ chính xác): 69.62%

Macro Average (trung bình trên tất cả lớp):

Precision: 0.70

Recall: 0.70

F1-score: 0.70

🔎 Classification Report chi tiết
Class	Precision	Recall	F1-score	Support
fair	0.65	0.63	0.64	320,000
medium	0.72	0.81	0.76	320,000
strong	0.64	0.69	0.66	320,000
weak	0.80	0.66	0.72	320,000

👉 Nhận xét:

Medium đạt recall cao nhất (0.81) → model bắt được nhiều trường hợp medium.

Weak có precision rất cao (0.80) → khi dự đoán "weak" thì đáng tin cậy.

Fair và Strong cân bằng nhưng chỉ ở mức trung bình (~0.64–0.66 F1).

📉 Confusion Matrix (Tóm tắt nhầm lẫn)
Thực tế ↓ / Dự đoán →	fair	medium	strong	weak
fair	200,779	0	119,182	39
medium	4,186	260,005	2,820	52,989
strong	100,615	0	219,364	21
weak	3,418	103,239	2,354	210,989

👉 Điểm đáng chú ý:

Rất nhiều fair bị nhầm sang strong (~119k).

Weak thường bị nhầm thành medium (~103k).

Medium được phân loại khá tốt (260k đúng trên 320k).

📈 Chỉ số chung (Overall Metrics)

True Positive Rate (TPR / Recall): 69.62%

False Positive Rate (FPR): 10.13%

True Negative Rate (TNR / Specificity): 89.87%

False Negative Rate (FNR): 30.38%

👉 Nghĩa là:

Model nhận diện đúng gần 70% mẫu thực tế.

Khả năng phân biệt tốt với TNR ~90%, nhưng vẫn bỏ sót khoảng 30% mẫu.

📌 Kết luận & Hướng cải thiện

MLP cho kết quả tương đương Gradient Boosting (~70% accuracy) và tốt hơn SVM (60%).

Điểm mạnh: nhận diện tốt medium và weak.

Điểm yếu: nhiều nhầm lẫn giữa fair ↔ strong và weak ↔ medium.

Có thể cải thiện bằng:

Thêm regularization (dropout, batch norm) để tránh overfitting.

Thử kiến trúc sâu hơn hoặc thay đổi số neurons mỗi layer.

So sánh thêm với các thuật toán boosting (XGBoost, LightGBM) để tìm trade-off tốc độ vs độ chính xác.