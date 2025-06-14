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

### 2.1 Logistic regression_OvR

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

### 2.2 Logistic Regression Softmax

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